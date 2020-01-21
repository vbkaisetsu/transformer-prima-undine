use prima_undine::functions::{ArithmeticFunctions, BasicFunctions};
use prima_undine::{Device, Model, Parameter};
use prima_undine_contrib::functions::ContribFunctions;

use serde::{Deserialize, Serialize};

use crate::attention::{MultiHeadAttention, MultiHeadAttentionParams};
use crate::feed_forward::{PositionwiseFeedForward, PositionwiseFeedForwardParams};
use crate::layer_normalization::{LayerNormalization, LayerNormalizationParams};

#[derive(Model, Serialize, Deserialize)]
pub struct DecoderUnitParams<'dev> {
    p_ln1: LayerNormalizationParams<'dev>,
    p_self_attention: MultiHeadAttentionParams<'dev>,
    p_ln2: LayerNormalizationParams<'dev>,
    p_src_attention: MultiHeadAttentionParams<'dev>,
    p_ln3: LayerNormalizationParams<'dev>,
    p_feed_forward: PositionwiseFeedForwardParams<'dev>,
}

impl<'dev> DecoderUnitParams<'dev> {
    pub fn new(device: &'dev Device, n_units: u32, n_ff_units_factor: u32) -> Self {
        Self {
            p_ln1: LayerNormalizationParams::new(device, n_units),
            p_self_attention: MultiHeadAttentionParams::new(device, n_units),
            p_ln2: LayerNormalizationParams::new(device, n_units),
            p_src_attention: MultiHeadAttentionParams::new(device, n_units),
            p_ln3: LayerNormalizationParams::new(device, n_units),
            p_feed_forward: PositionwiseFeedForwardParams::new(device, n_units, n_ff_units_factor),
        }
    }
}

pub struct DecoderUnit<T> {
    dropout: f32,
    ln1: LayerNormalization<T>,
    self_attention: MultiHeadAttention<T>,
    ln2: LayerNormalization<T>,
    src_attention: MultiHeadAttention<T>,
    ln3: LayerNormalization<T>,
    feed_forward: PositionwiseFeedForward<T>,
}

impl<'arg, 'dev, T> DecoderUnit<T>
where
    'dev: 'arg,
    T: From<&'arg mut Parameter<'dev>> + ArithmeticFunctions<T> + BasicFunctions + ContribFunctions,
    for<'a> &'a T: ArithmeticFunctions<T>,
{
    pub fn new(n_heads: u32, dropout: f32, params: &'arg mut DecoderUnitParams<'dev>) -> Self {
        Self {
            dropout: dropout,
            ln1: LayerNormalization::new(1e-6, &mut params.p_ln1),
            self_attention: MultiHeadAttention::new(n_heads, dropout, &mut params.p_self_attention),
            ln2: LayerNormalization::new(1e-6, &mut params.p_ln2),
            src_attention: MultiHeadAttention::new(n_heads, dropout, &mut params.p_src_attention),
            ln3: LayerNormalization::new(1e-6, &mut params.p_ln3),
            feed_forward: PositionwiseFeedForward::new(dropout, &mut params.p_feed_forward),
        }
    }

    pub fn forward(&self, enc_inp: &T, dec_inp: &T, self_mask: &T, src_mask: &T, train: bool) -> T {
        let dec = dec_inp;

        let self_att = self
            .self_attention
            .forward(&dec, &dec, &dec, self_mask, train);
        let dec = dec + self_att.dropout(self.dropout, train);
        let dec = self.ln1.forward(&dec);

        let src_att = self
            .src_attention
            .forward(&dec, enc_inp, enc_inp, src_mask, train);
        let dec = dec + src_att.dropout(self.dropout, train);
        let dec = self.ln2.forward(&dec);

        let ff = self.feed_forward.forward(&dec, train);
        let dec = dec + ff.dropout(self.dropout, train);
        let dec = self.ln3.forward(&dec);

        dec
    }
}

#[derive(Model, Serialize, Deserialize)]
pub struct DecoderParams<'dev> {
    p_layers: Vec<DecoderUnitParams<'dev>>,
}

impl<'dev> DecoderParams<'dev> {
    pub fn new(device: &'dev Device, n_units: u32, n_ff_units_factor: u32, n_layers: u32) -> Self {
        Self {
            p_layers: (0..n_layers)
                .map(|_| DecoderUnitParams::new(device, n_units, n_ff_units_factor))
                .collect(),
        }
    }
}

pub struct Decoder<T> {
    layers: Vec<DecoderUnit<T>>,
}

impl<'arg, 'dev, T> Decoder<T>
where
    'dev: 'arg,
    T: From<&'arg mut Parameter<'dev>> + ArithmeticFunctions<T> + BasicFunctions + ContribFunctions,
    for<'a> &'a T: ArithmeticFunctions<T>,
{
    pub fn new(n_heads: u32, dropout: f32, params: &'arg mut DecoderParams<'dev>) -> Self {
        Self {
            layers: params
                .p_layers
                .iter_mut()
                .map(|p_layer| DecoderUnit::new(n_heads, dropout, p_layer))
                .collect(),
        }
    }

    pub fn forward(&self, src: &T, trg: T, self_mask: &T, src_mask: &T, train: bool) -> T {
        let mut trg = trg;
        for layer in &self.layers {
            trg = layer.forward(src, &trg, self_mask, src_mask, train);
        }
        trg
    }
}
