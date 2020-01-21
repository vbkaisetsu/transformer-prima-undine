use prima_undine::functions::{ArithmeticFunctions, BasicFunctions};
use prima_undine::{Device, Model, Parameter};
use prima_undine_contrib::functions::ContribFunctions;

use serde::{Deserialize, Serialize};

use crate::attention::{MultiHeadAttention, MultiHeadAttentionParams};
use crate::feed_forward::{PositionwiseFeedForward, PositionwiseFeedForwardParams};
use crate::layer_normalization::{LayerNormalization, LayerNormalizationParams};

#[derive(Model, Serialize, Deserialize)]
pub struct EncoderUnitParams<'dev> {
    p_self_attention: MultiHeadAttentionParams<'dev>,
    p_ln1: LayerNormalizationParams<'dev>,
    p_feed_forward: PositionwiseFeedForwardParams<'dev>,
    p_ln2: LayerNormalizationParams<'dev>,
}

impl<'dev> EncoderUnitParams<'dev> {
    pub fn new(device: &'dev Device, n_units: u32, n_ff_units_factor: u32) -> Self {
        Self {
            p_self_attention: MultiHeadAttentionParams::new(device, n_units),
            p_ln1: LayerNormalizationParams::new(device, n_units),
            p_feed_forward: PositionwiseFeedForwardParams::new(device, n_units, n_ff_units_factor),
            p_ln2: LayerNormalizationParams::new(device, n_units),
        }
    }
}

pub struct EncoderUnit<T> {
    dropout: f32,
    self_attention: MultiHeadAttention<T>,
    ln1: LayerNormalization<T>,
    feed_forward: PositionwiseFeedForward<T>,
    ln2: LayerNormalization<T>,
}

impl<'arg, 'dev, T> EncoderUnit<T>
where
    'dev: 'arg,
    T: From<&'arg mut Parameter<'dev>> + ArithmeticFunctions<T> + BasicFunctions + ContribFunctions,
    for<'a> &'a T: ArithmeticFunctions<T>,
{
    pub fn new(n_heads: u32, dropout: f32, params: &'arg mut EncoderUnitParams<'dev>) -> Self {
        Self {
            dropout: dropout,
            self_attention: MultiHeadAttention::new(n_heads, dropout, &mut params.p_self_attention),
            ln1: LayerNormalization::new(1e-6, &mut params.p_ln1),
            feed_forward: PositionwiseFeedForward::new(dropout, &mut params.p_feed_forward),
            ln2: LayerNormalization::new(1e-6, &mut params.p_ln2),
        }
    }

    pub fn forward(&self, src: &T, mask: &T, train: bool) -> T {
        let enc = src;

        let att = self.self_attention.forward(&enc, &enc, &enc, mask, train);
        let enc = enc + att.dropout(self.dropout, train);
        let enc = self.ln1.forward(&enc);

        let ff = self.feed_forward.forward(&enc, train);
        let enc = enc + ff.dropout(self.dropout, train);
        let enc = self.ln2.forward(&enc);

        enc
    }
}

#[derive(Model, Serialize, Deserialize)]
pub struct EncoderParams<'dev> {
    p_layers: Vec<EncoderUnitParams<'dev>>,
}

impl<'dev> EncoderParams<'dev> {
    pub fn new(device: &'dev Device, n_units: u32, n_ff_units_factor: u32, n_layers: u32) -> Self {
        Self {
            p_layers: (0..n_layers)
                .map(|_| EncoderUnitParams::new(device, n_units, n_ff_units_factor))
                .collect(),
        }
    }
}

pub struct Encoder<T> {
    layers: Vec<EncoderUnit<T>>,
}

impl<'arg, 'dev, T> Encoder<T>
where
    'dev: 'arg,
    T: From<&'arg mut Parameter<'dev>> + ArithmeticFunctions<T> + BasicFunctions + ContribFunctions,
    for<'a> &'a T: ArithmeticFunctions<T>,
{
    pub fn new(n_heads: u32, dropout: f32, params: &'arg mut EncoderParams<'dev>) -> Self {
        Self {
            layers: params
                .p_layers
                .iter_mut()
                .map(|p_layer| EncoderUnit::new(n_heads, dropout, p_layer))
                .collect(),
        }
    }

    pub fn forward(&self, src: T, mask: &T, train: bool) -> T {
        let mut src = src;
        for layer in &self.layers {
            src = layer.forward(&src, mask, train);
        }
        src
    }
}
