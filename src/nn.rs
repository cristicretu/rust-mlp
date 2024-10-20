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
        }
    }

    pub fn set_grad(&mut self, grad: f32) {
        self.grad = grad;
    }

    pub fn tanh(&self) -> Value {
        let temp =
            (f32::exp(2.0 * self.data as f32) - 1.0) / (f32::exp(2.0 * self.data as f32) + 1.0);
        return Value::new(temp, Some(vec![self.clone()]), Some("tanh".to_string()));
    }
}

impl ops::Add<Value> for Value {
    type Output = Value;
    fn add(self, _rhs: Value) -> Value {
        return Value::new(
            self.data + _rhs.data,
            Some(vec![self, _rhs]),
            Some("+".to_string()),
        );
    }
}

impl ops::Mul<Value> for Value {
    type Output = Value;
    fn mul(self, _rhs: Value) -> Value {
        return Value::new(
            self.data * _rhs.data,
            Some(vec![self, _rhs]),
            Some("*".to_string()),
        );
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
        Value::new(
            self.data / rhs.data,
            Some(vec![self, rhs]),
            Some("/".to_string()),
        )
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
