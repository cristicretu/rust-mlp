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

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}
