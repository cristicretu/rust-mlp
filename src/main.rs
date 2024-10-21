mod nn;
use nn::Value;

use ndarray::Array;

fn f(x: f32) -> f32 {
    return x.powf(2.0) * 3.0 - 4.0 * x + 5.0;
}

fn df(x: f32) -> f32 {
    return 6.0 * x - 4.0;
}

fn main() {
    let x1 = Value::new(2.0, None, None);
    let x2 = Value::new(0.0, None, None);

    let w1 = Value::new(-3.0, None, None);
    let w2 = Value::new(1.0, None, None);

    let b = Value::new(6.8813735870195432, None, None);

    let x1w1 = x1 * w1;
    let x2w2 = x2 * w2;
    let x1w1_x2w2 = x1w1 + x2w2;
    let n = x1w1_x2w2 + b;
    let mut o = n.tanh();
    o.set_grad(1.0);
    o.backward();
    println!("o = {}", o);
    o.print_all_children();

    // --------------

    // let a = Value::new(2.0, vec![], None);
    // let b = Value::new(-3.0, vec![], None);
    // let c = Value::new(10.0, vec![], None);
    // let e = a * b;
    // let d = e + c;
    // let f = Value::new(-2.0, vec![], None);
    // let mut L = d * f;
    // L.set_grad(1.0);
    // println!("L = {}", L);

    // ------------

    // let xs = Array::linspace(-5.0, 5.0, 42);
    // let ys = xs.mapv(f);

    // let mut x: f32 = 0.66;
    // let h = 0.001;

    // println!("df(3.0) = {}", (f(x + h) - f(x)) / (h));
    // println!("df(3.0) = {}", df(x));
}
