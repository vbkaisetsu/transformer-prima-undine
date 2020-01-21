use prima_undine::functions::{ArithmeticFunctions, BasicFunctions};
use prima_undine::{initializers as I, shape, Device, Model, Parameter};
use prima_undine_contrib::functions::ContribFunctions;

use serde::{Deserialize, Serialize};

use crate::utils::to_refs;

#[derive(Model, Serialize, Deserialize)]
pub struct TransformerEmbeddingsParams<'dev> {
    plookup: Parameter<'dev>,
    pby: Parameter<'dev>,
}

impl<'dev> TransformerEmbeddingsParams<'dev> {
    pub fn new(device: &'dev Device, vocab: u32, n_units: u32) -> Self {
        Self {
            plookup: device.new_parameter(shape![n_units, vocab], &I::XavierUniform::new(1.)),
            pby: device.new_parameter(shape![1, vocab], &I::XavierUniform::new(1.)),
        }
    }
}

pub struct TransformerEmbeddings<T> {
    dropout: f32,
    lookup: T,
    by: T,
}

impl<'arg, 'dev, T> TransformerEmbeddings<T>
where
    'dev: 'arg,
    T: From<&'arg mut Parameter<'dev>> + ArithmeticFunctions<T> + BasicFunctions + ContribFunctions,
    for<'a> &'a T: ArithmeticFunctions<T>,
{
    pub fn new(dropout: f32, params: &'arg mut TransformerEmbeddingsParams<'dev>) -> Self {
        Self {
            dropout: dropout,
            lookup: T::from(&mut params.plookup),
            by: T::from(&mut params.pby),
        }
    }

    pub fn encode(&self, seq: &[&[u32]], pe: &T, train: bool) -> T {
        let n_units = self.lookup.shape()[0];

        let mut embed = vec![];
        for w in seq {
            let e = self.lookup.pick(w, 1);
            embed.push(e);
        }
        let embed_tensor = T::concat(&to_refs(&embed), 1).transpose();

        let embed_tensor = embed_tensor * (n_units as f32).sqrt();
        let pos = pe.slice(0, 0, seq.len() as u32);
        (embed_tensor + pos).dropout(self.dropout, train)
    }

    pub fn decode(&self, x: &T) -> T {
        let by = self.by.broadcast(0, x.shape()[0]);
        x.matmul(&self.lookup) + by
    }
}
