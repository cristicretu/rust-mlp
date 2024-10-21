use core::fmt;
use std::{
    collections::HashSet,
    hash::{Hash, Hasher},
    ops,
};

#[derive(Clone)]
pub struct Value {
    data: f32,
    grad: f32,
    _backward: fn(&mut Value),
    _prev: HashSet<Value>,
    _op: Option<String>,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
    }
}

impl Value {
    pub fn new(data: f32, _children: Option<Vec<Value>>, _op: Option<String>) -> Value {
        Value {
            data,
            _prev: _children.unwrap_or_else(Vec::new).into_iter().collect(),
            _op,
            grad: 0.0,
            _backward: |_| {},
        }
    }

    pub fn set_grad(&mut self, grad: f32) {
        println!("Setting grad to {}", grad);
        self.grad = grad;
    }

    pub fn set_backward(&mut self, backward: fn(&mut Value)) {
        self._backward = backward;
    }

    pub fn backward(&mut self) {
        (self._backward)(self);
    }

    pub fn tanh(&self) -> Value {
        let temp =
            (f32::exp(2.0 * self.data as f32) - 1.0) / (f32::exp(2.0 * self.data as f32) + 1.0);

        fn backward(out: &mut Value) {
            if let Some(lhs) = out._prev.iter().next() {
                let mut lhs = lhs.clone();
                lhs.set_grad(out.grad * (1.0 - out.data * out.data));
            }
        }

        let mut out = Value::new(temp, Some(vec![self.clone()]), Some("tanh".to_string()));

        out.set_backward(backward);
        out
    }

    pub fn print_all_children(&self) {
        for child in self._prev.iter() {
            println!("{}", child);
            child.print_all_children();
        }
    }
}

impl ops::Add<Value> for Value {
    type Output = Value;
    fn add(self, _rhs: Value) -> Value {
        fn backward(out: &mut Value) {
            if let Some(lhs) = out._prev.iter().next() {
                let mut lhs = lhs.clone();
                lhs.set_grad(1.0 * out.grad);
            }

            if let Some(rhs) = out._prev.iter().nth(1) {
                let mut rhs = rhs.clone();
                rhs.set_grad(1.0 * out.grad);
            }
        }

        let mut out = Value::new(
            self.data + _rhs.data,
            Some(vec![self.clone(), _rhs.clone()]),
            Some("+".to_string()),
        );

        out.set_backward(backward);
        out
    }
}

impl ops::Mul<Value> for Value {
    type Output = Value;
    fn mul(self, rhs: Value) -> Value {
        fn backward(out: &mut Value) {
            if let Some(lhs) = out._prev.iter().next() {
                let mut lhs = lhs.clone();
                if let Some(rhs) = out._prev.iter().nth(1) {
                    lhs.set_grad(out.grad * rhs.data);
                }
            }

            if let Some(rhs) = out._prev.iter().nth(1) {
                let mut rhs = rhs.clone();
                if let Some(lhs) = out._prev.iter().next() {
                    rhs.set_grad(out.grad * lhs.data);
                }
            }
        }

        let mut out = Value::new(
            self.data * rhs.data,
            Some(vec![self.clone(), rhs.clone()]),
            Some("*".to_string()),
        );

        out.set_backward(backward);
        out
    }
}

impl ops::Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        Value::new(-self.data, Some(vec![self]), Some("neg".to_string()))
    }
}

impl ops::Sub<Value> for Value {
    type Output = Value;
    fn sub(self, rhs: Value) -> Value {
        self + (-rhs)
    }
}

impl ops::Div<Value> for Value {
    type Output = Value;
    fn div(self, rhs: Value) -> Value {
        fn backward(out: &mut Value) {
            if let Some(lhs) = out._prev.iter().next() {
                let mut lhs = lhs.clone();
                if let Some(rhs) = out._prev.iter().nth(1) {
                    lhs.set_grad(out.grad / rhs.data);
                }
            }

            if let Some(rhs) = out._prev.iter().nth(1) {
                let mut rhs = rhs.clone();
                if let Some(lhs) = out._prev.iter().next() {
                    rhs.set_grad(-out.grad * lhs.data / (rhs.data * rhs.data));
                }
            }
        };

        let mut out = Value::new(
            self.data / rhs.data,
            Some(vec![self.clone(), rhs.clone()]),
            Some("/".to_string()),
        );

        out.set_backward(backward);
        out
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Value({:.4}", self.data)?;
        if !self._prev.is_empty() {
            write!(f, ", prev=[")?;
            let mut iter = self._prev.iter();
            if let Some(first) = iter.next() {
                write!(f, "{:.4}", first.data)?;
                for value in iter {
                    write!(f, ", {:.4}", value.data)?;
                }
            }
            write!(f, "],")?;
        }
        write!(
            f,
            " op={:?}",
            self._op.as_ref().unwrap_or(&"None".to_string()),
        )?;
        write!(f, ", grad={:.4}", self.grad)?;
        write!(f, ")")
    }
}
