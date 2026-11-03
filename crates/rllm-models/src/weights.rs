//! In-memory parameter store and dictionary for model weights.
//!
//! Provides a key-value store [`Weights`] mapping parameter tensor names to [`Tensor`] instances,
//! with safe lookup and mutable borrowing for layer construction and weight loading.

use rllm_core::{
    error::{Result, RllmError},
    tensor::Tensor,
};
use std::collections::HashMap;

/// An in-memory dictionary mapping tensor names to tensor instances.
///
/// Used as an intermediate container when deserializing GGUF files or SafeTensors
/// before injecting weights into model layers.
#[derive(Default)]
pub struct Weights {
    tensors: HashMap<String, Tensor>,
}

impl Weights {
    /// Creates an empty weight dictionary.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }

    /// Inserts a named tensor into the dictionary.
    pub fn insert(&mut self, name: String, tensor: Tensor) {
        self.tensors.insert(name, tensor);
    }

    /// Retrieves an immutable reference to a named tensor.
    ///
    /// # Errors
    ///
    /// Returns [`RllmError::ModelError`] if the tensor name is not present.
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| RllmError::ModelError(format!("Weight tensor '{}' not found", name)))
    }

    /// Retrieves a mutable reference to a named tensor.
    ///
    /// # Errors
    ///
    /// Returns [`RllmError::ModelError`] if the tensor name is not present.
    pub fn get_mut(&mut self, name: &str) -> Result<&mut Tensor> {
        self.tensors
            .get_mut(name)
            .ok_or_else(|| RllmError::ModelError(format!("Weight tensor '{}' not found", name)))
    }

    /// Returns the total number of tensors in the dictionary.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns true if the dictionary contains no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}
