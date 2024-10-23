mod value;
use std::borrow::BorrowMut;

use value::Value;

mod nn;
use nn::MLP;
// use ndarray::Array;

// fn f(x: f32) -> f32 {
//     return x.powf(2.0) * 3.0 - 4.0 * x + 5.0;
// }

// fn df(x: f32) -> f32 {
//     return 6.0 * x - 4.0;
// }

fn main() {
    // let x1 = Value::from(2.0);
    // let x2 = Value::from(0.0);

    // let w1 = Value::from(-3.0);
    // let w2 = Value::from(1.0);

    // let b = Value::from(6.8813735870195432);

    // let x1w1 = x1 * w1;
    // let x2w2 = x2 * w2;
    // let x1w1_x2w2 = x1w1 + x2w2;
    // let n = x1w1_x2w2 + b;
    // let o = n.tanh();
    // o.backward();
    // println!("o = {}", o);
    // o.print_all();

    let n = MLP::new(3, vec![4, 4, 1]);
    let xs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];
    let ys = vec![1.0, -1.0, -1.0, 1.0];

    // Forward pass
    let ypred = n.forward(&xs);
    println!("ypred = [");
    for pred in ypred.iter() {
        println!("  {:.4}", pred[0].borrow().data);
    }
    println!("]");

    // Calculate loss (e.g., mean squared error)
    let mut loss = Value::from(0.0);
    for (pred, y) in ypred.iter().zip(ys.iter()) {
        // loss = loss + (&*pred[0] - &Value::from(*y)).pow(2.0);
        println!("pred = {}", pred[0].borrow().data);
        println!("y = {}", y);

        loss = loss + (pred[0].borrow().data - y).powf(2.0);
    }
    // Backward pass
    loss.backward();

    // Print the loss
    println!("Loss: {}", loss.borrow().data);
    println!("Gradient: {}", n.layers[0].neurons[0].weights[0]);

    // Optionally, print gradients of parameters
    // n.print_gradients();

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
