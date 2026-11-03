use crate::error::{Result, RllmError};
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn from_slice(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            0
        } else {
            self.dims.iter().product()
        }
    }

    pub fn dim(&self, index: usize) -> Result<usize> {
        self.dims.get(index).copied().ok_or_else(|| RllmError::DimOutOfRange {
            dim: index,
            shape: self.dims.clone(),
        })
    }

    /// Computes default contiguous row-major strides.
    pub fn default_strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.dims.len()];
        if self.dims.is_empty() {
            return strides;
        }
        for i in (0..self.dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Broadcast two shapes together following standard NumPy/PyTorch rules.
    pub fn broadcast(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
        let max_rank = lhs.rank().max(rhs.rank());
        let mut out_dims = vec![0; max_rank];

        for i in 0..max_rank {
            let l_dim = if i < lhs.rank() {
                lhs.dims[lhs.rank() - 1 - i]
            } else {
                1
            };
            let r_dim = if i < rhs.rank() {
                rhs.dims[rhs.rank() - 1 - i]
            } else {
                1
            };

            if l_dim == r_dim {
                out_dims[max_rank - 1 - i] = l_dim;
            } else if l_dim == 1 {
                out_dims[max_rank - 1 - i] = r_dim;
            } else if r_dim == 1 {
                out_dims[max_rank - 1 - i] = l_dim;
            } else {
                return Err(RllmError::BroadcastIncompatible {
                    lhs: lhs.dims.clone(),
                    rhs: rhs.dims.clone(),
                });
            }
        }

        Ok(Shape::new(out_dims))
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.dims)
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.dims)
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(arr: [usize; N]) -> Self {
        Shape::new(arr.to_vec())
    }
}

impl From<&[usize]> for Shape {
    fn from(s: &[usize]) -> Self {
        Shape::new(s.to_vec())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        Shape::new(v)
    }
}

