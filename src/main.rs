use ndarray::{arr2, Array2, ArrayView2};

pub mod logic;

fn main() {
    let inputs = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
    let target = arr2(&[[0.0], [1.0], [1.0], [0.0]]);
    let mut network = logic::network::Layer::new()
    .add_layer(2, 3)
    .add_layer(3, 1);

    network.train(inputs, target, 100);

    // network.forward_propagate(inputs.row(0).clone().to_owned());
    // network.forward_propagate(inputs.row(1).clone().to_owned());
    // network.forward_propagate(inputs.row(2).clone().to_owned());
    // network.forward_propagate(inputs.row(3).clone().to_owned());
}
