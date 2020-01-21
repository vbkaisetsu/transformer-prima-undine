use prima_undine::functions::{ArithmeticFunctions, BasicFunctions};
use prima_undine::{initializers as I, shape, Device, Model, Parameter};
use prima_undine_contrib::functions::ContribFunctions;

use serde::{Deserialize, Serialize};

#[derive(Model, Serialize, Deserialize)]
pub struct LayerNormalizationParams<'dev> {
    pg: Parameter<'dev>,
    pb: Parameter<'dev>,
}

impl<'dev> LayerNormalizationParams<'dev> {
    pub fn new(device: &'dev Device, n_units: u32) -> Self {
        Self {
            pg: device.new_parameter(shape![1, n_units], &I::Constant::new(1.)),
            pb: device.new_parameter(shape![1, n_units], &I::Constant::new(0.)),
        }
    }
}

pub struct LayerNormalization<T> {
    eps: f32,
    g: T,
    b: T,
}

impl<'arg, 'dev, T> LayerNormalization<T>
where
    'dev: 'arg,
    T: From<&'arg mut Parameter<'dev>> + ArithmeticFunctions<T> + BasicFunctions + ContribFunctions,
    for<'a> &'a T: ArithmeticFunctions<T>,
{
    pub fn new(eps: f32, params: &'arg mut LayerNormalizationParams<'dev>) -> Self {
        Self {
            eps: eps,
            g: T::from(&mut params.pg),
            b: T::from(&mut params.pb),
        }
    }

    pub fn forward(&self, x: &T) -> T {
        let s0 = x.shape()[0];
        let s1 = x.shape()[1];

        let g = self.g.broadcast(0, s0);
        let b = self.b.broadcast(0, s0);

        let m = x.mean(1);
        let sd = ((x * x).mean(1) - &m * &m).sqrt();

        let m = m.broadcast(1, s1);
        let sd = sd.broadcast(1, s1);

        g * (x - m) / (sd + self.eps) + b
    }
}
