use prima_undine::functions::{ArithmeticFunctions, BasicFunctions};
use prima_undine::{initializers as I, shape, Device, Model, Parameter};
use prima_undine_contrib::functions::ContribFunctions;

use serde::{Deserialize, Serialize};

#[derive(Model, Serialize, Deserialize)]
pub struct PositionwiseFeedForwardParams<'dev> {
    pw1: Parameter<'dev>,
    pb1: Parameter<'dev>,
    pw2: Parameter<'dev>,
    pb2: Parameter<'dev>,
}

impl<'dev> PositionwiseFeedForwardParams<'dev> {
    pub fn new(device: &'dev Device, n_units: u32, n_ff_units_factor: u32) -> Self {
        Self {
            pw1: device.new_parameter(
                shape![n_units, n_units * n_ff_units_factor],
                &I::XavierUniform::new(1.),
            ),
            pb1: device.new_parameter(
                shape![1, n_units * n_ff_units_factor],
                &I::XavierUniform::new(1.),
            ),
            pw2: device.new_parameter(
                shape![n_units * n_ff_units_factor, n_units],
                &I::XavierUniform::new(1.),
            ),
            pb2: device.new_parameter(shape![1, n_units], &I::XavierUniform::new(1.)),
        }
    }
}

pub struct PositionwiseFeedForward<T> {
    dropout: f32,
    w1: T,
    b1: T,
    w2: T,
    b2: T,
}

impl<'arg, 'dev, T> PositionwiseFeedForward<T>
where
    'dev: 'arg,
    T: From<&'arg mut Parameter<'dev>> + ArithmeticFunctions<T> + BasicFunctions + ContribFunctions,
    for<'a> &'a T: ArithmeticFunctions<T>,
{
    pub fn new(dropout: f32, params: &'arg mut PositionwiseFeedForwardParams<'dev>) -> Self {
        Self {
            dropout: dropout,
            w1: T::from(&mut params.pw1),
            b1: T::from(&mut params.pb1),
            w2: T::from(&mut params.pw2),
            b2: T::from(&mut params.pb2),
        }
    }

    pub fn forward(&self, x: &T, train: bool) -> T {
        let seq_len = x.shape()[0];
        let b1 = self.b1.broadcast(0, seq_len);
        let b2 = self.b2.broadcast(0, seq_len);

        let h = (x.matmul(&self.w1) + &b1)
            .relu()
            .dropout(self.dropout, train);
        h.matmul(&self.w2) + b2
    }
}
