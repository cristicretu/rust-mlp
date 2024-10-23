use crate::value::Value;

#[derive(Debug)]
pub struct Neuron {
    weights: Vec<f64>,
    bias: Value,
}

impl Neuron {
    pub fn new(nin: i64) -> Neuron {
        Neuron {
            weights: (0..nin)
                .map(|_| rand::random::<f64>() * 2.0 - 1.0)
                .collect(),
            bias: Value::from(rand::random::<f64>() * 2.0 - 1.0),
        }
    }

    pub fn activation(&self, x: Vec<Value>) -> Value {
        let mut n = self.bias.clone();
        for i in 0..self.weights.len() {
            n = n + x[i].clone() * Value::from(self.weights[i]);
        }
        n.tanh()
    }
}

#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: i64, nout: i64) -> Layer {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(nin)).collect(),
        }
    }

    pub fn activation(&self, x: Vec<Value>) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|n| n.activation(x.clone()))
            .collect()
    }
}

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: i64, nouts: Vec<i64>) -> MLP {
        let mut layers = vec![];
        let mut prev_nout = nin;
        for nout in nouts {
            layers.push(Layer::new(prev_nout, nout));
            prev_nout = nout;
        }
        MLP { layers }
    }

    pub fn activation(&self, x: Vec<Value>) -> Vec<Value> {
        let mut x = x;
        for layer in &self.layers {
            x = layer.activation(x);
        }
        x
    }
}
