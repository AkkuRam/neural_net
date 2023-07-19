use image::GenericImageView;
use ndarray::{s, Array, Array1, Array2, Array3};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::f64::consts::E;
use std::fs;
use std::ops::AddAssign;

pub struct Layer {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    data: Vec<Array2<f64>>,
    weights_transposed: bool, // New flag to track whether the weights are transposed or not
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
        let mut gradients = inputs.map(|x| (E.powf(-*x)) / (1.0 + E.powf(-*x)).powf(2.0));
        for i in (0..self.weights.len()).rev() {
            gradients = (gradients * &errors).map(&|x| x * learning_rate);

            self.weights[i] = &self.weights[i] +  &gradients.dot(&self.data[i].t());
            self.biases[i] = &self.biases[i] +  &gradients.t();

            errors = self.weights[i].t().dot(&errors);
            gradients = self.data[i].map(|x| (E.powf(-*x)) / (1.0 + E.powf(-*x)).powf(2.0));
        }
    }

    pub fn train(&mut self, inputs: Array2<f64>, target: Array2<f64>, epochs: u16){
    
        for epoch in 0..epochs {
            println!("Epoch {}", epoch);
            for i in 0..inputs.shape()[0] {
                let input = inputs.row(i).to_owned();
                let target = target.row(i).to_owned();
                println!("Target {:?}", target);
                let output = self.forward_propagate(input);
                self.back_propagate(output, target, 0.2);
            }
        }
    }
}

pub fn sigmoid(inputs: Array2<f64>) -> Array2<f64> {
    inputs.map(|x| (1.0 / (1.0 + E.powf(-*x))))
}
