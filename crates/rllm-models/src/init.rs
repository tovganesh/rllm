//! Deterministic synthetic weight initialisation for tests and benchmarks.

use rllm_core::{device::Device, error::Result, shape::Shape, tensor::Tensor};

/// SplitMix64-based generator producing reproducible synthetic weights.
///
/// Weights are non-constant so that layout or indexing bugs change the output,
/// which constant-valued weights would hide.
pub struct SyntheticInit {
    state: u64,
}

impl SyntheticInit {
    /// Creates a generator from a seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform sample in `[-1, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// Tensor of the given shape with values uniform in `[-scale, scale)`.
    pub fn uniform(&mut self, dims: &[usize], scale: f32, device: Device) -> Result<Tensor> {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|_| self.next_f32() * scale).collect();
        Tensor::from_f32(&data, Shape::new(dims.to_vec()), device)
    }

    /// Linear layer weight `[out_features, in_features]` scaled by `1/sqrt(in_features)`.
    pub fn linear(&mut self, out_features: usize, in_features: usize, device: Device) -> Result<Tensor> {
        self.uniform(&[out_features, in_features], 1.0 / (in_features as f32).sqrt(), device)
    }

    /// Normalisation weight `[dim]` with values in `[0.9, 1.1)`.
    pub fn norm_weight(&mut self, dim: usize, device: Device) -> Result<Tensor> {
        let data: Vec<f32> = (0..dim).map(|_| 1.0 + 0.1 * self.next_f32()).collect();
        Tensor::from_f32(&data, Shape::new(vec![dim]), device)
    }
}
