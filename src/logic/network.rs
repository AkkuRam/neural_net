use ndarray::{s, Array, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};
use std::error::Error;
use std::f64::consts::E;
use std::{
    fs::File,
    io::{Read, Write},
};

pub struct Layer {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    data: Vec<Array2<f64>>,
    weights_transposed: bool, // New flag to track whether the weights are transposed or not
}

#[derive(Serialize, Deserialize)]
struct LayerData {
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<Vec<f64>>>,
    data: Vec<Vec<Vec<f64>>>,
    weights_transposed: bool,
}

trait ToVec2 {
    fn to_vec2(&self) -> Vec<Vec<f64>>;
}

impl ToVec2 for Array2<f64> {
    fn to_vec2(&self) -> Vec<Vec<f64>> {
        self.outer_iter().map(|row| row.to_vec()).collect()
    }
}

impl Layer {
    pub fn new() -> Layer {
        Layer {
            weights: Vec::new(),
            biases: Vec::new(),
            data: Vec::new(),
            weights_transposed: false, // Initialize the flag as false
        }
    }

    pub fn add_layer(mut self, n_inputs: usize, n_nodes: usize) -> Self {
        self.weights
            .push(Array::random((n_inputs, n_nodes), Uniform::new(-1., 1.)));
        self.biases
            .push(Array::random((1, n_nodes), Uniform::new(-1., 1.)));

        self
    }

    pub fn transpose_weights(&mut self) {
        for weight in &mut self.weights {
            *weight = weight.t().to_owned();
        }
        self.weights_transposed = true;
    }

    pub fn forward_propagate(&mut self, inputs: Array1<f64>) -> Array2<f64> {
        if !self.weights_transposed {
            self.transpose_weights();
        }

        let mut converted_inputs = inputs.clone().into_shape((1, inputs.len())).unwrap();
        converted_inputs.swap_axes(0, 1);

        self.data = vec![converted_inputs.clone()];
        for i in 0..self.weights.len() {
            converted_inputs =
                sigmoid((&self.weights[i].dot(&converted_inputs)) + &self.biases[i].t());
            self.data.push(converted_inputs.clone());
        }
        println!("{:?}", converted_inputs);
        converted_inputs
    }

    pub fn back_propagate(&mut self, inputs: Array2<f64>, target: Array1<f64>, learning_rate: f64) {
        let converted_targets = target.clone().into_shape((1, target.len())).unwrap();
        let mut errors = &converted_targets.t() - &inputs;
        let mut gradients = inputs.map(|x| x * (1.0 - x));
        for i in (0..self.weights.len()).rev() {
            gradients = gradients * &errors * learning_rate;

            self.weights[i] = &self.weights[i] + &gradients.dot(&self.data[i].t());
            self.biases[i] = &self.biases[i] + &gradients.t();

            errors = self.weights[i].t().dot(&errors);
            gradients = self.data[i].map(|x| x * (1.0 - x));
        }
    }

    pub fn train(&mut self, inputs: Array2<f64>, target: Array2<f64>, epochs: u16) {
        for epoch in 0..epochs {
            println!("Epoch {}", epoch);
            for i in 0..inputs.shape()[0] {
                let input = inputs.row(i).to_owned();
                let target = target.row(i).to_owned();
                println!("Target: {:?}", target);
                let output = self.forward_propagate(input);
                self.back_propagate(output, target, 0.5);
            }
            println!("\n");
        }
    }

    pub fn save(&self, file_path: &str) -> Result<(), Box<dyn Error>> {
        let layer_data = LayerData {
            weights: self.weights.iter().map(|array| array.to_vec2()).collect(),
            biases: self.biases.iter().map(|array| array.to_vec2()).collect(),
            data: self.data.iter().map(|array| array.to_vec2()).collect(),
            weights_transposed: self.weights_transposed,
        };

        let serialized_data = serde_json::to_string_pretty(&layer_data)?;

        let mut file = File::create(file_path)?;
        file.write_all(serialized_data.as_bytes())?;

        Ok(())
    }

    pub fn load(&mut self, file_path: &str) -> Result<(), Box<dyn Error>> {
        let mut file = File::open(file_path)?;
        let mut serialized_data = String::new();
        file.read_to_string(&mut serialized_data)?;

        let layer_data: LayerData = serde_json::from_str(&serialized_data)?;

      
        fn convert_data(data: &Vec<Vec<Vec<f64>>>) -> Result<Vec<Array2<f64>>, Box<dyn Error>> {
            let mut arrays = Vec::new();
            for item_data in data {
                let shape = (item_data.len(), item_data[0].len());
                let item_array = Array::from_shape_vec(shape, item_data.iter().flatten().cloned().collect())?;
                arrays.push(item_array);
            }
            Ok(arrays)
        }
       
        self.weights = convert_data(&layer_data.weights)?;

      
        self.biases = convert_data(&layer_data.biases)?;

        self.weights_transposed = layer_data.weights_transposed;

        Ok(())
    }

}

pub fn sigmoid(inputs: Array2<f64>) -> Array2<f64> {
    inputs.map(|x| (1.0 / (1.0 + E.powf(-*x))))
}
