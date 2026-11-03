use rayon::prelude::*;

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

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

pub fn gelu_f32(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());

    out.par_iter_mut()
        .zip(x.par_iter())
        .for_each(|(o, &val)| {
            *o = gelu(val);
        });
}

