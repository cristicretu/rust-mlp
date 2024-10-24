mod value;

use value::Value;

mod nn;
use nn::MLP;

fn calc_loss(n: &MLP, xs: &Vec<Vec<f64>>, ys: &Vec<f64>) -> Value {
    let ypred = n.forward(xs);
    let mut loss = Value::from(0.0);
    for (pred, y) in ypred.iter().zip(ys.iter()) {
        let y_val = Value::from(*y);
        // Clone or dereference appropriately
        let pred_val = pred[0].clone(); // assuming Value implements Clone
        let diff = pred_val - y_val; // now both are owned Values
        loss = loss + diff.pow(2.0);
    }
    loss
}

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

    let num_epochs = 1000;
    let step_size = 0.01;

    for epoch in 0..num_epochs {
        let loss = calc_loss(&n, &xs, &ys);
        println!("Epoch {}, loss = {}", epoch, loss.borrow().data);

        loss.backward();

        // println!("Gradients after backward:");
        // for (i, p) in n.parameters().iter().enumerate() {
        //     println!("  Parameter {}: grad = {}", i, p.borrow().grad);
        // }

        for p in n.parameters() {
            let grad = p.borrow().grad;
            p.borrow_mut().data += -step_size * grad;
        }

        // println!("Updated parameters:");
        // for (i, p) in n.parameters().iter().enumerate() {
        //     println!("  Parameter {}: data = {}", i, p.borrow().data);
        // }

        // Zero out gradients for the next iteration
        for p in n.parameters() {
            p.borrow_mut().grad = 0.0;
        }
    }

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
