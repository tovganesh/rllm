//! Activation functions (SiLU, GELU, SwiGLU).

use rayon::prelude::*;

/// Sigmoid Linear Unit (SiLU): $x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Fast GELU approximation:
/// $$0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)\right)\right)$$
#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// SwiGLU activation: $(x_{\text{gate}} \cdot \sigma(x_{\text{gate}})) \cdot x_{\text{up}}$.
pub fn swiglu_f32(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());

    out.par_iter_mut()
        .zip(gate.par_iter())
        .zip(up.par_iter())
        .for_each(|((o, &g), &u)| {
            *o = silu(g) * u;
        });
}

/// Applies GELU activation across a slice of elements in parallel.
pub fn gelu_f32(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());

    out.par_iter_mut()
        .zip(x.par_iter())
        .for_each(|(o, &val)| {
            *o = gelu(val);
        });
}
