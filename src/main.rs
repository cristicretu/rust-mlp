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
    let mut a = Value::new(2.0);
    let mut b = Value::new(-3.0);
    println!("{}", a + b);

    // let xs = Array::linspace(-5.0, 5.0, 42);
    // let ys = xs.mapv(f);

    // let mut x: f32 = 0.66;
    // let h = 0.001;

    // println!("df(3.0) = {}", (f(x + h) - f(x)) / (h));
    // println!("df(3.0) = {}", df(x));
}
