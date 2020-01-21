use prima_undine::functions::{ArithmeticFunctions, BasicFunctions};
use prima_undine::{initializers as I, shape, Device, Model, Parameter};
use prima_undine_contrib::functions::ContribFunctions;

use serde::{Deserialize, Serialize};

use crate::utils::to_refs;

pub struct ScaledDotProductAttention {
    dropout: f32,
}

impl ScaledDotProductAttention {
    pub fn new(dropout: f32) -> Self {
        Self { dropout: dropout }
    }

    pub fn forward<'arg, 'dev, T>(&self, query: &T, key: &T, value: &T, mask: &T, train: bool) -> T
    where
        'dev: 'arg,
        T: From<&'arg mut Parameter<'dev>>
            + ArithmeticFunctions<T>
            + BasicFunctions
            + ContribFunctions,
        for<'a> &'a T: ArithmeticFunctions<T>,
    {
        let d_k = query.shape()[1];
        let attn = query.matmul(key.transpose()) / (d_k as f32).sqrt();

        let attn = attn - mask;

        let attn_prob = attn.softmax(1).dropout(self.dropout, train);
        let out = attn_prob.matmul(value);

        out
    }
}

#[derive(Model, Serialize, Deserialize)]
pub struct MultiHeadAttentionParams<'dev> {
    pwq: Parameter<'dev>,
    pwk: Parameter<'dev>,
    pwv: Parameter<'dev>,
    pwo: Parameter<'dev>,
}

impl<'dev> MultiHeadAttentionParams<'dev> {
    pub fn new(device: &'dev Device, n_units: u32) -> Self {
        Self {
            pwq: device.new_parameter(shape![n_units, n_units], &I::XavierUniform::new(1.)),
            pwk: device.new_parameter(shape![n_units, n_units], &I::XavierUniform::new(1.)),
            pwv: device.new_parameter(shape![n_units, n_units], &I::XavierUniform::new(1.)),
            pwo: device.new_parameter(shape![n_units, n_units], &I::XavierUniform::new(1.)),
        }
    }
}

pub struct MultiHeadAttention<T> {
    n_heads: u32,
    wq: T,
    wk: T,
    wv: T,
    wo: T,
    attention: ScaledDotProductAttention,
}

impl<'arg, 'dev, T> MultiHeadAttention<T>
where
    'dev: 'arg,
    T: From<&'arg mut Parameter<'dev>> + ArithmeticFunctions<T> + BasicFunctions + ContribFunctions,
    for<'a> &'a T: ArithmeticFunctions<T>,
{
    pub fn new(
        n_heads: u32,
        dropout: f32,
        params: &'arg mut MultiHeadAttentionParams<'dev>,
    ) -> Self {
        Self {
            n_heads: n_heads,
            wq: T::from(&mut params.pwq),
            wk: T::from(&mut params.pwk),
            wv: T::from(&mut params.pwv),
            wo: T::from(&mut params.pwo),
            attention: ScaledDotProductAttention::new(dropout),
        }
    }

    pub fn forward(&self, query: &T, key: &T, value: &T, mask: &T, train: bool) -> T {
        let query = query.matmul(&self.wq);
        let key = key.matmul(&self.wk);
        let value = value.matmul(&self.wv);

        let query = query.split(1, self.n_heads);
        let key = key.split(1, self.n_heads);
        let value = value.split(1, self.n_heads);

        let mask = mask * 2000.;
        let mask = if mask.shape()[0] != query[0].shape()[0] {
            mask.broadcast(0, query[0].shape()[0])
        } else {
            mask
        };

        let mut heads = vec![];
        for i in 0..self.n_heads as usize {
            let head = self
                .attention
                .forward(&query[i], &key[i], &value[i], &mask, train);
            heads.push(head);
        }
        let heads = T::concat(&to_refs(&heads), 1);

        heads.matmul(&self.wo)
    }
}
