use crate::operators::operators::*;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::thread_rng();
        let w = (0..nin)
            .map(|_| Value::new(rng.gen_range(-1.0..1.0), "w"))
            .collect::<Vec<Value>>();
        Neuron {
            bias: Value::new(0.0, "b"),
            weights: w, 
        }
    }

    pub fn forward(&self, xs: &[Value]) -> Value {
        let prods = std::iter::zip(&self.weights, xs)
            .map(|(a, b)| a.clone() * b.clone())
            .collect::<Vec<Value>>();

        let sum = prods
            .into_iter()
            .fold(self.bias.clone(), |acc, v| acc + v);
        sum.tanh()
    }
    
    pub fn parameters(&self) -> Vec<Value> {
        [self.bias.clone()]
            .into_iter()
            .chain(self.weights.clone())
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin:usize, nout:usize) -> Self {
        Layer {
            neurons: (0..nout)
                .map(|_| Neuron::new(nin))
                .collect()
        }
    }

    pub fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

#[derive(Debug, Clone)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nout: Vec<usize>) -> Self {
        let out_cnt = nout.len();
        let layer_size: Vec<usize> = [nin].into_iter().chain(nout).collect();

        MLP {
            layers: (0..out_cnt)
                .map(|i| Layer::new(layer_size[i], layer_size[i + 1]))
                .collect()
        }
    }

    pub fn forward(&self, mut xs: Vec<Value>) -> Vec<Value> {
        for layer in &self.layers {
            xs = layer.forward(&xs);
        }
        xs
    }

    
    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init() {
        let n = Neuron::new(2);
        println!("{:#?}", n);
    }
    

    #[test]
    fn forward_shape_and_grad() {
        let neuron = Neuron::new(2);
        let x1 = Value::new(1.0, "x1");
        let x2 = Value::new(2.0, "x2");
        let out = neuron.forward(&[x1, x2]);
        GraphNode::backward(&out);
        println!("out = {:?}", out);
    }

    #[test]
    fn simple_model() {
        let x = vec![2.0, 3.0, -1.0];
        let mlp = MLP::new(3, vec![4, 4, 1]);

        let xs = vec![
            vec![2.0, 3.0, -1.0],
            vec![3.0, -1.0, 0.5],
            vec![0.5, 1.0, 1.0],
            vec![1.0, 1.0, -1.0],
        ];

        let ys = vec![Value::new(1.0, ""), Value::new(-1.0, ""), Value::new(-1.0, ""), Value::new(1.0, "")];
        let ypred: Vec<Value> = xs
            .iter()
            .map(|x| mlp.forward(x.iter().map(|x| Value::from(*x)).collect())[0].clone())
            .collect();

        let ypred_floats: Vec<f64> = ypred.iter().map(|v| v.borrow().data).collect();

        let ygt = ys.iter().map(|y| Value::from(y.clone()));

        // Loss function
        // let loss: Value = ypred
        //     .into_iter()
        //     .zip(ygt)
        //     .map(|(yp, yg)| (yp - yg).powop(2.0))
        //     .sum();
    }
}
