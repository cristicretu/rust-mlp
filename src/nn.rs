use core::fmt;
use std::ops;

pub struct Value {
    data: f32,
}

impl Value {
    pub fn new(data: f32) -> Value {
        Value { data }
    }
}

impl ops::Add<Value> for Value {
    type Output = Value;
    fn add(self, _rhs: Value) -> Value {
        return Value::new(self.data + _rhs.data);
    }
}

impl ops::Mul<Value> for Value {
    type Output = Value;
    fn mul(self, _rhs: Value) -> Value {
        return Value::new(self.data * _rhs.data);
    }
}

impl ops::Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        Value::new(-self.data)
    }
}

impl ops::Sub<Value> for Value {
    type Output = Value;
    fn sub(self, rhs: Value) -> Value {
        self + (-rhs)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}
