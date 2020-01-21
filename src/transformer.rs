use prima_undine::functions::{ArithmeticFunctions, BasicFunctions};
use prima_undine::{Device, Model, Parameter};
use prima_undine_contrib::functions::ContribFunctions;

use serde::{Deserialize, Serialize};

use crate::decoder::{Decoder, DecoderParams};
use crate::embeddings::{TransformerEmbeddings, TransformerEmbeddingsParams};
use crate::encoder::{Encoder, EncoderParams};
use crate::utils::to_refs;

#[derive(Model, Serialize, Deserialize)]
pub struct TransformerParams<'dev> {
    p_embed: TransformerEmbeddingsParams<'dev>,
    p_encoder: EncoderParams<'dev>,
    p_decoder: DecoderParams<'dev>,
}

impl<'dev> TransformerParams<'dev> {
    pub fn new(
        device: &'dev Device,
        vocab: u32,
        n_units: u32,
        n_ff_units_factor: u32,
        n_layers: u32,
    ) -> Self {
        Self {
            p_embed: TransformerEmbeddingsParams::new(device, vocab, n_units),
            p_encoder: EncoderParams::new(device, n_units, n_ff_units_factor, n_layers),
            p_decoder: DecoderParams::new(device, n_units, n_ff_units_factor, n_layers),
        }
    }
}

pub struct Transformer<T> {
    embed: TransformerEmbeddings<T>,
    encoder: Encoder<T>,
    decoder: Decoder<T>,
}

impl<'arg, 'dev, T> Transformer<T>
where
    'dev: 'arg,
    T: From<&'arg mut Parameter<'dev>> + ArithmeticFunctions<T> + BasicFunctions + ContribFunctions,
    for<'a> &'a T: ArithmeticFunctions<T>,
{
    pub fn new(n_heads: u32, dropout: f32, params: &'arg mut TransformerParams<'dev>) -> Self {
        Self {
            embed: TransformerEmbeddings::new(dropout, &mut params.p_embed),
            encoder: Encoder::new(n_heads, dropout, &mut params.p_encoder),
            decoder: Decoder::new(n_heads, dropout, &mut params.p_decoder),
        }
    }

    pub fn encode(&self, src: &[&[u32]], pe: &T, src_mask: &T, train: bool) -> T {
        self.encoder
            .forward(self.embed.encode(src, pe, train), src_mask, train)
    }

    pub fn decode(
        &self,
        src: &T,
        trg: &[&[u32]],
        pe: &T,
        self_mask: &T,
        src_mask: &T,
        train: bool,
    ) -> T {
        let ret = self.decoder.forward(
            src,
            self.embed.encode(trg, pe, train),
            self_mask,
            src_mask,
            train,
        );
        self.embed.decode(&ret)
    }

    pub fn loss(
        &self,
        src: &[&[u32]],
        trg: &[&[u32]],
        pe: &T,
        src_mask: &T,
        self_mask: &T,
        train: bool,
    ) -> T {
        let output = self.decode(
            &self.encode(src, pe, src_mask, train),
            &trg[..trg.len() - 1],
            pe,
            self_mask,
            src_mask,
            train,
        );
        let mut losses = vec![];
        for (i, t) in trg[1..].iter().enumerate() {
            let y = output.pick(&[i as u32], 0);
            let loss = y.sparse_softmax_cross_entropy(&t, 1);
            losses.push(loss);
        }
        let loss = T::slice_sum(&to_refs(&losses)).batch_mean();
        loss
    }
}
