use core::fmt;
use std::collections::HashSet;
use std::ops::{self, Add, Div, Mul, Sub};
use std::{
    cell::RefCell,
    hash::{Hash, Hasher},
    rc::Rc,
};

pub struct ValueData {
    pub data: f64,
    pub grad: f64,
    pub backward: Option<fn(value: &ValueData)>,
    pub prev: Vec<Value>,
    pub op: Option<String>,
}

impl ValueData {
    fn new(data: f64) -> ValueData {
        ValueData {
            data,
            prev: Vec::new(),
            op: None,
            grad: 0.0,
            backward: None,
        }
    }
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueData>>);

impl ops::Deref for Value {
    type Target = Rc<RefCell<ValueData>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.borrow().data == other.borrow().data
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.borrow().data.to_bits().hash(state);
    }
}

impl Value {
    fn new(value: ValueData) -> Value {
        Value(Rc::new(RefCell::new(value)))
    }
}

impl<T: Into<f64>> From<T> for Value {
    fn from(t: T) -> Value {
        Value::new(ValueData::new(t.into()))
    }
}
impl Value {
    pub fn tanh(&self) -> Self {
        let mut out = ValueData::new(self.borrow().data.tanh());

        out.prev = vec![self.clone()];
        out.op = Some(String::from("tanh"));
        out.backward = Some(|value: &ValueData| {
            let prev = &value.prev[0];
            prev.borrow_mut().grad += (1.0 - value.data * value.data) * value.grad;
        });

        Value::new(out)
    }

    fn build_topo(&self) -> Vec<Value> {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<Value> = HashSet::new();
        self._build_topo(&mut topo, &mut visited);
        topo
    }

    fn _build_topo(&self, topo: &mut Vec<Value>, visited: &mut HashSet<Value>) {
        if visited.insert(self.clone()) {
            self.borrow().prev.iter().for_each(|child| {
                child._build_topo(topo, visited);
            });
            topo.push(self.clone());
        }
    }

    pub fn print_all(&self) {
        for child in self.borrow().prev.iter() {
            println!("{}", child);
            child.print_all();
        }
    }

    pub fn set_grad(&mut self, grad: f64) {
        self.borrow_mut().grad = grad;
    }

    pub fn zero_grad(&self) {
        self.borrow_mut().grad = 0.0;

        for child in self.borrow().prev.iter() {
            child.zero_grad();
        }
    }

    pub fn backward(&self) {
        let mut topo = self.build_topo();
        topo.reverse();

        self.borrow_mut().grad = 1.0;
        for v in topo {
            if let Some(backprop) = v.borrow().backward {
                backprop(&v.borrow());
            }
        }
    }
}

impl Add for Value {
    type Output = Self;
    fn add(self, _rhs: Self) -> Self {
        let mut out = ValueData::new(self.borrow().data + _rhs.borrow().data);
        out.prev = vec![self, _rhs];
        out.op = Some(String::from("+"));
        out.backward = Some(|value: &ValueData| {
            value.prev[0].borrow_mut().grad += 1.0 * value.grad;
            value.prev[1].borrow_mut().grad += 1.0 * value.grad;
        });
        Value::new(out)
    }
}

impl Add<f64> for Value {
    type Output = Value;
    fn add(self, rhs: f64) -> Value {
        self + Value::from(rhs)
    }
}

impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: &Value) -> Value {
        self.clone() + rhs.clone()
    }
}

impl Add<Value> for f64 {
    type Output = Value;
    fn add(self, rhs: Value) -> Value {
        Value::from(self) + rhs
    }
}

impl Add<i32> for Value {
    type Output = Value;
    fn add(self, rhs: i32) -> Value {
        self + Value::from(rhs as f64)
    }
}

impl Add<Value> for i32 {
    type Output = Value;
    fn add(self, rhs: Value) -> Value {
        Value::from(self as f64) + rhs
    }
}

impl Mul for Value {
    type Output = Self;
    fn mul(self, _rhs: Self) -> Self {
        let mut out = ValueData::new(self.borrow().data * _rhs.borrow().data);
        out.prev = vec![self.clone(), _rhs.clone()];
        out.op = Some(String::from("*"));
        out.backward = Some(|value: &ValueData| {
            let left = &value.prev[0];
            let right = &value.prev[1];
            left.borrow_mut().grad += value.grad * right.borrow().data;
            right.borrow_mut().grad += value.grad * left.borrow().data;
        });
        Value::new(out)
    }
}

impl Mul<f64> for Value {
    type Output = Value;
    fn mul(self, rhs: f64) -> Value {
        self * Value::from(rhs)
    }
}

impl Mul for &Value {
    type Output = Value;
    fn mul(self, rhs: &Value) -> Value {
        self.clone() * rhs.clone()
    }
}

impl Mul<Value> for f64 {
    type Output = Value;
    fn mul(self, rhs: Value) -> Value {
        Value::from(self) * rhs
    }
}

impl Mul<i32> for Value {
    type Output = Value;
    fn mul(self, rhs: i32) -> Value {
        self * Value::from(rhs as f64)
    }
}

impl Div for Value {
    type Output = Self;
    fn div(self, _rhs: Self) -> Self {
        let mut out = ValueData::new(self.borrow().data / _rhs.borrow().data);
        out.prev = vec![self.clone(), _rhs.clone()];
        out.op = Some(String::from("/"));
        out.backward = Some(|value: &ValueData| {
            value.prev[0].borrow_mut().grad += value.grad / value.prev[1].borrow().data;
            value.prev[1].borrow_mut().grad += -value.grad * value.prev[0].borrow().data
                / (value.prev[1].borrow().data * value.prev[1].borrow().data);
        });
        Value::new(out)
    }
}

impl Div<f64> for Value {
    type Output = Value;
    fn div(self, rhs: f64) -> Value {
        self / Value::from(rhs)
    }
}

impl Div for &Value {
    type Output = Value;
    fn div(self, rhs: &Value) -> Value {
        self.clone() / rhs.clone()
    }
}

impl Div<Value> for f64 {
    type Output = Value;
    fn div(self, rhs: Value) -> Value {
        Value::from(self) / rhs
    }
}

impl Div<i32> for Value {
    type Output = Value;
    fn div(self, rhs: i32) -> Value {
        self / Value::from(rhs as f64)
    }
}

impl ops::Neg for Value {
    type Output = Self;
    fn neg(self) -> Self {
        let mut out = ValueData::new(-self.borrow().data);
        out.prev = vec![self.clone()];
        out.op = Some(String::from("neg"));
        Value::new(out)
    }
}

impl Sub for Value {
    type Output = Self;
    fn sub(self, _rhs: Self) -> Self {
        self + (-_rhs)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Value({:.4}", self.borrow().data)?;
        if !self.borrow().prev.is_empty() {
            write!(f, ", prev=[")?;
            let prev = self.borrow();
            let mut iter = prev.prev.iter();
            if let Some(first) = iter.next() {
                write!(f, "{:.4}", first.borrow().data)?;
                for value in iter {
                    write!(f, ", {:.4}", value.borrow().data)?;
                }
            }
            write!(f, "],")?;
        }
        write!(
            f,
            " op={:?}",
            self.borrow().op.as_ref().unwrap_or(&"None".to_string()),
        )?;
        write!(f, ", grad={:.4}", self.borrow().grad)?;
        write!(f, ")")
    }
}
