use crate::value::Value;

#[derive(Debug)]
pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: Value,
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

    pub fn forward(&self, x: Vec<Value>) -> Value {
        let mut n = self.bias.clone();
        for i in 0..self.weights.len() {
            n = n + x[i].clone() * Value::from(self.weights[i]);
        }
        n.tanh()
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = vec![self.bias.clone()];
        params.extend(self.weights.iter().map(|w| w.clone().into()));
        params
    }
}

#[derive(Debug)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: i64, nout: i64) -> Layer {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(nin)).collect(),
        }
    }

    pub fn forward(&self, x: Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x.clone())).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

#[derive(Debug)]
pub struct MLP {
    pub layers: Vec<Layer>,
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

    pub fn forward(&self, xs: &Vec<Vec<f64>>) -> Vec<Vec<Value>> {
        xs.iter()
            .map(|x| {
                let mut out = x.iter().map(|&xi| Value::from(xi)).collect::<Vec<Value>>();
                for layer in &self.layers {
                    out = layer.forward(out);
                }
                out
            })
            .collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
