mod attention;
mod beam_search;
mod decoder;
mod embeddings;
mod encoder;
mod feed_forward;
mod layer_normalization;
mod transformer;
mod utils;

use std::collections::{BTreeSet, VecDeque};
use std::error::Error;
use std::io::{stdin, stdout, BufRead, Write};
use std::rc::Rc;

use prima_undine::functions::BasicFunctions;
use prima_undine::{optimizers as O, shape, Device, Model, Node, Optimizer, OptimizerBase, Tensor};
use prima_undine_opencl::OpenCL;

use crate::beam_search::Hypothesis;
use crate::transformer::{Transformer, TransformerParams};
use crate::utils::{
    clean_corpus, load_corpus, load_msgpack, make_batch, make_batch_itr, make_wid_corpus,
    save_msgpack, to_slices, update_vocab, WordIds,
};

const SRC_VOCAB_SIZE: u32 = 5000;
const TRG_VOCAB_SIZE: u32 = 4000;
const N_UNITS: u32 = 512;
const N_HEADS: u32 = 8;
const N_LAYERS: u32 = 2;
const N_FF_UNITS_FACTOR: u32 = 4;
const DROPOUT: f32 = 0.1;
const MAX_LENGTH: u32 = 32;
const MAX_EPOCH: usize = 100000;
const MAX_BATCH_SENTS: usize = 100000;
const MAX_BATCH_TOKENS: usize = 4096;
const UPDATE_FREQ: usize = 8;
const WARMUP_STEPS: f32 = 8000.;
const MAX_RATIO: f32 = 2.0;
const BEAM_WIDTH: usize = 10;

fn make_sinusoidal_positional_encoding<'dev>(
    device: &'dev Device,
    n_units: u32,
    max_length: u32,
) -> Tensor<'dev> {
    let position = (0..max_length).map(|i| i as f32).collect::<Vec<f32>>();
    let position = device.new_tensor_by_slice(shape![max_length], &position);
    let num_timescales = n_units / 2;
    let log_timescale_increment = 10000f32.ln() / (num_timescales - 1) as f32;
    let timescale = (0..num_timescales).map(|i| i as f32).collect::<Vec<f32>>();
    let timescale = device.new_tensor_by_slice(shape![1, num_timescales], &timescale);
    let timescale = (timescale * -log_timescale_increment).exp();
    let v = position.broadcast(1, num_timescales) * timescale.broadcast(0, max_length);
    Tensor::concat(&[&v.sin(), &v.cos()], 1)
}

fn make_padding_position_mask<'dev>(
    device: &'dev Device,
    batch: &[&[u32]],
    padding: u32,
) -> Tensor<'dev> {
    let batch_size = batch[0].len();
    let sent_len = batch.len();
    let mut mask = vec![0.; sent_len * batch_size];
    for j in 0..batch_size {
        for i in 0..sent_len {
            if batch[i][j] == padding {
                mask[i + j * sent_len] = 1.;
            } else {
                mask[i + j * sent_len] = 0.;
            }
        }
    }
    device.new_tensor_by_slice(shape![1, sent_len as u32; batch_size as u32], &mask)
}

fn make_future_blinding_mask<'dev>(device: &'dev Device, sent_len: u32) -> Tensor<'dev> {
    let one = device.new_tensor_by_constant(shape![sent_len, sent_len], 1.);
    one.triangular_u(1)
}

fn main() -> Result<(), Box<dyn Error>> {
    let train = false;

    let dev = OpenCL::new(0, 0);

    let pe = make_sinusoidal_positional_encoding(&dev, N_UNITS, MAX_LENGTH);

    if train {
        let mut best_valid_loss = std::f32::MAX;

        let train_src = load_corpus("data/train.en")?;
        let train_trg = load_corpus("data/train.ja")?;
        let valid_src = load_corpus("data/dev.en")?;
        let valid_trg = load_corpus("data/dev.ja")?;

        let (train_src, train_trg) =
            clean_corpus(&train_src, &train_trg, MAX_LENGTH as usize - 2, MAX_RATIO);
        let (valid_src, valid_trg) =
            clean_corpus(&valid_src, &valid_trg, MAX_LENGTH as usize - 2, MAX_RATIO);

        let mut wids = WordIds::new();

        update_vocab(&train_src, SRC_VOCAB_SIZE, &mut wids);
        update_vocab(&train_trg, TRG_VOCAB_SIZE, &mut wids);

        save_msgpack(&wids, "wids.msgpack");

        let train_src = make_wid_corpus(&train_src, &wids);
        let train_trg = make_wid_corpus(&train_trg, &wids);
        let valid_src = make_wid_corpus(&valid_src, &wids);
        let valid_trg = make_wid_corpus(&valid_trg, &wids);

        let mut optimizer = O::Adam::new(1.0, 0.9, 0.98, 1e-9);

        let mut model =
            TransformerParams::new(&dev, wids.len(), N_UNITS, N_FF_UNITS_FACTOR, N_LAYERS);
        optimizer.configure_model(&mut model);
        optimizer.set_gradient_clipping(5.);

        let mut valid_fail_cnt = 0;

        for i in 0..MAX_EPOCH {
            println!("epoch: {}", i);
            let mut train_loss = 0.;
            let mut trained_sents = 0;
            for (step, ids_chunk) in make_batch_itr(
                &train_src,
                &train_trg,
                MAX_BATCH_SENTS,
                MAX_BATCH_TOKENS,
                true,
            )
            .iter()
            .enumerate()
            {
                print!("\t{} / {}\r", trained_sents, train_src.len());
                stdout().flush().unwrap();
                trained_sents += ids_chunk.len();
                let src_batch = make_batch(&to_slices(&train_src), &ids_chunk, &wids);
                let trg_batch = make_batch(&to_slices(&train_trg), &ids_chunk, &wids);
                let src_mask = make_padding_position_mask(&dev, &to_slices(&src_batch), wids.eos());
                let self_mask = make_padding_position_mask(
                    &dev,
                    &to_slices(&trg_batch[..trg_batch.len() - 1]),
                    wids.eos(),
                )
                .broadcast(0, trg_batch.len() as u32 - 1)
                    + make_future_blinding_mask(&dev, trg_batch.len() as u32 - 1);
                {
                    let transformer = Transformer::new(N_HEADS, DROPOUT, &mut model);
                    let loss = transformer.loss(
                        &to_slices(&src_batch),
                        &to_slices(&trg_batch),
                        &Node::from(&pe),
                        &Node::from(&src_mask),
                        &Node::from(&self_mask),
                        true,
                    );
                    let loss = loss / UPDATE_FREQ as f32;
                    loss.backward();
                    let loss_val = loss.to_float();
                    train_loss += loss_val * UPDATE_FREQ as f32 * ids_chunk.len() as f32;
                }

                if (step + 1) % UPDATE_FREQ == 0 {
                    let step_num = *optimizer.epoch() + 1;
                    let new_scale = (N_UNITS as f32).powf(-0.5)
                        * utils::minf(
                            (step_num as f32).powf(-0.5),
                            step_num as f32 * WARMUP_STEPS.powf(-1.5),
                        );
                    optimizer.set_learning_rate_scaling(new_scale);
                    optimizer.update_model(&mut model);
                }
            }
            println!(
                "\ttrain loss = {}, learning rate = {}",
                train_loss / train_src.len() as f32,
                optimizer.get_learning_rate_scaling()
            );

            let mut valid_loss = 0.;
            let mut sent_cnt = 0;
            for ids_chunk in make_batch_itr(
                &valid_src,
                &valid_trg,
                MAX_BATCH_SENTS,
                MAX_BATCH_TOKENS,
                false,
            ) {
                print!("\t{} / {}\r", sent_cnt, valid_src.len());
                stdout().flush().unwrap();
                sent_cnt += ids_chunk.len();
                let src_batch = make_batch(&to_slices(&valid_src), &ids_chunk, &wids);
                let trg_batch = make_batch(&to_slices(&valid_trg), &ids_chunk, &wids);
                let src_mask = make_padding_position_mask(&dev, &to_slices(&src_batch), wids.eos());
                let self_mask = make_padding_position_mask(
                    &dev,
                    &to_slices(&trg_batch[..trg_batch.len() - 1]),
                    wids.eos(),
                )
                .broadcast(0, trg_batch.len() as u32 - 1)
                    + make_future_blinding_mask(&dev, trg_batch.len() as u32 - 1);
                {
                    let transformer = Transformer::new(N_HEADS, DROPOUT, &mut model);
                    let loss = transformer.loss(
                        &to_slices(&src_batch),
                        &to_slices(&trg_batch),
                        &pe,
                        &src_mask,
                        &self_mask,
                        false,
                    );
                    let loss_val = loss.to_float();
                    valid_loss += loss_val * ids_chunk.len() as f32;
                }
            }
            println!("\tvalid loss = {}", valid_loss / valid_src.len() as f32);

            if valid_loss > best_valid_loss {
                valid_fail_cnt += 1;
                if valid_fail_cnt >= 4 {
                    break;
                } else {
                    continue;
                }
            }

            valid_fail_cnt = 0;

            best_valid_loss = valid_loss;

            save_msgpack(&model, "model.msgpack");
        }
        println!("finish!");
    } else {
        let wids: WordIds = load_msgpack("wids.msgpack");
        let mut model: TransformerParams = load_msgpack("model.msgpack");
        model.move_to_device(&dev);

        print!(">>> ");
        stdout().flush().unwrap();

        for line in stdin().lock().lines() {
            let transformer = Transformer::new(N_HEADS, DROPOUT, &mut model);
            let mut src = vec![vec![wids.bos()]];
            for word in line?.split_whitespace() {
                src.push(vec![wids.get_wid(word)]);
            }
            src.push(vec![wids.eos()]);
            let src_mask = make_padding_position_mask(&dev, &to_slices(&src), wids.eos());
            let src_encode = transformer.encode(&to_slices(&src), &pe, &src_mask, false);

            {
                println!("Greedy decode:");
                let mut trg = vec![vec![wids.bos()]];
                for _ in 0..MAX_LENGTH - 1 {
                    let self_mask = make_future_blinding_mask(&dev, trg.len() as u32);
                    let out = transformer.decode(
                        &src_encode,
                        &to_slices(&trg),
                        &pe,
                        &self_mask,
                        &src_mask,
                        false,
                    );
                    let y = out.pick(&[out.shape()[0] - 1], 0);
                    let y = y.argmax(1);
                    if y[0] == wids.eos() {
                        break;
                    }
                    print!("{} ", wids.get_word(y[0]));
                    trg.push(y);
                }
                println!();
                if trg.len() == MAX_LENGTH as usize {
                    eprintln!("<<< MAX_LEN reached >>>");
                }
            }
            {
                println!("Beam decode:");
                let mut eos_best_cost = std::f32::MAX;
                let mut eos_best_hyp = None;
                let mut trg = vec![vec![wids.bos()]];
                let mut prev_hyps = vec![None];
                for pos in 0..MAX_LENGTH - 1 {
                    let self_mask = make_future_blinding_mask(&dev, trg.len() as u32);
                    let out = transformer.decode(
                        &src_encode,
                        &to_slices(&trg),
                        &pe,
                        &self_mask,
                        &src_mask,
                        false,
                    );
                    let y = out.pick(&[out.shape()[0] - 1], 0).softmax(1);
                    let sorted_idx = (-&y).argsort(1);
                    let y_probs = y.to_vec();
                    let mut hyps = BTreeSet::new();
                    for batch in 0..y.shape().batch() as usize {
                        for i in 0..BEAM_WIDTH {
                            let idx = sorted_idx[batch * y.shape().volume() as usize + i];
                            let prob = y_probs[idx as usize];
                            let wid = idx % y.shape().volume();
                            let hyp = Hypothesis::new(&prev_hyps[batch], prob, wid);
                            if hyp.cost < eos_best_cost {
                                if wid == wids.eos() {
                                    eos_best_cost = hyp.cost;
                                    eos_best_hyp = Some(hyp);
                                // println!("eos_cost={}", eos_best_cost);
                                } else {
                                    hyps.insert(hyp);
                                }
                            }
                        }
                    }
                    if hyps.len() == 0 {
                        break;
                    }
                    let mut sentences = vec![];
                    prev_hyps = vec![];
                    for (i, hyp) in hyps.iter().enumerate() {
                        prev_hyps.push(Some(Rc::clone(hyp)));
                        if i >= BEAM_WIDTH {
                            break;
                        }
                        /*
                        println!(
                            "hyp {}-{}: {}, cost={}",
                            pos,
                            i,
                            wids.get_word(hyp.data),
                            hyp.cost
                        );
                        */
                        sentences.push(hyp.gen_vec(wids.bos()));
                    }
                    trg = make_batch(
                        &to_slices(&sentences),
                        &(0..sentences.len()).collect::<Vec<usize>>(),
                        &wids,
                    );
                    // println!("======");
                }
                if let Some(hyp) = eos_best_hyp {
                    let mut hyp = hyp;
                    let mut stack = VecDeque::new();
                    while let Some(prev_hyp) = hyp.prev.as_ref() {
                        stack.push_front(prev_hyp.data);
                        hyp = Rc::clone(prev_hyp);
                    }
                    for &wid in &stack {
                        print!("{} ", wids.get_word(wid));
                    }
                    println!();
                    if trg.len() == MAX_LENGTH as usize {
                        eprintln!("<<< MAX_LEN reached >>>");
                    }
                }
            }
            print!(">>> ");
            stdout().flush().unwrap();
        }
    }
    Ok(())
}
