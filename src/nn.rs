use core::fmt;
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

    pub fn print_all_children(&self) {
        for child in self.borrow().prev.iter() {
            println!("{}", child);
            child.print_all_children();
        }
    }

    pub fn set_grad(&mut self, grad: f64) {
        self.borrow_mut().grad = grad;
    }

    pub fn backward(&self) {
        (self.borrow().backward.unwrap())(&self.borrow());
    }
}

impl Add for Value {
    type Output = Self;
    fn add(self, _rhs: Self) -> Self {
        let mut out = ValueData::new(self.borrow().data + _rhs.borrow().data);
        out.prev = vec![self, _rhs];
        out.op = Some(String::from("+"));
        out.backward = Some(|value: &ValueData| {
            value.prev[0].borrow_mut().grad += value.grad;
            value.prev[1].borrow_mut().grad += value.grad;
        });
        Value::new(out)
    }
}

impl Mul for Value {
    type Output = Self;
    fn mul(self, _rhs: Self) -> Self {
        let mut out = ValueData::new(self.borrow().data * _rhs.borrow().data);
        out.prev = vec![self, _rhs];
        out.op = Some(String::from("*"));
        out.backward = Some(|value: &ValueData| {
            value.prev[0].borrow_mut().grad += value.grad * value.prev[1].borrow().data;
            value.prev[1].borrow_mut().grad += value.grad * value.prev[0].borrow().data;
        });
        Value::new(out)
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
