use rllm_core::{
    error::{Result, RllmError},
    tensor::Tensor,
};
use std::collections::HashMap;

#[derive(Default)]
pub struct Weights {
    tensors: HashMap<String, Tensor>,
}

impl Weights {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, tensor: Tensor) {
        self.tensors.insert(name, tensor);
    }

    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| RllmError::ModelError(format!("Weight tensor '{}' not found", name)))
    }

    pub fn get_mut(&mut self, name: &str) -> Result<&mut Tensor> {
        self.tensors
            .get_mut(name)
            .ok_or_else(|| RllmError::ModelError(format!("Weight tensor '{}' not found", name)))
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}
