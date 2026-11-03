//! Logit post-processing, temperature scaling, top-k/top-p filtering, and sampling.
//!
//! Implements greedy argmax, categorical sampling with temperature, nucleus (top-p),
//! top-k truncation, and frequency/repetition penalty.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Sampling parameters controlling token selection probability distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Temperature scaling factor ($T$). Values $\le 0.0$ trigger deterministic greedy argmax.
    pub temperature: f32,
    /// Nucleus sampling threshold ($0.0 < p \le 1.0$), capping cumulative probability mass.
    pub top_p: f32,
    /// Top-K sampling cutoff, retaining only the $K$ highest-probability tokens.
    pub top_k: usize,
    /// Repetition penalty factor applied to previously generated tokens.
    pub repetition_penalty: f32,
    /// Maximum number of tokens to generate before halting.
    pub max_tokens: usize,
    /// Set of token IDs that signal immediate end of generation when sampled.
    pub stop_tokens: Vec<u32>,
}

impl Default for SamplingParams {
    /// Sensible conversational default sampling hyperparameters.
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            max_tokens: 256,
            stop_tokens: vec![128001, 128009, 2], // Common EOS tokens
        }
    }
}

/// Token sampler that processes raw unnormalized logits into sampled token IDs.
pub struct Sampler;

impl Sampler {
    /// Samples a single token from raw model output logits according to [`SamplingParams`].
    ///
    /// Applies repetition penalty, temperature scaling, stable softmax, top-K filtering,
    /// top-p nucleus truncation, and categorical sampling.
    pub fn sample(
        logits: &[f32],
        params: &SamplingParams,
        context_tokens: &[u32],
    ) -> u32 {
        if params.temperature <= 0.0 || params.top_k == 1 {
            // Greedy argmax
            return logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
        }

        let mut scaled_logits = logits.to_vec();

        // 1. Repetition penalty
        if (params.repetition_penalty - 1.0).abs() > 1e-4 {
            for &token in context_tokens {
                let idx = token as usize;
                if idx < scaled_logits.len() {
                    if scaled_logits[idx] > 0.0 {
                        scaled_logits[idx] /= params.repetition_penalty;
                    } else {
                        scaled_logits[idx] *= params.repetition_penalty;
                    }
                }
            }
        }

        // 2. Temperature scaling
        let inv_temp = 1.0f32 / params.temperature;
        for val in scaled_logits.iter_mut() {
            *val *= inv_temp;
        }

        // 3. Softmax with max subtraction
        let max_val = scaled_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut probs: Vec<(u32, f32)> = scaled_logits
            .iter()
            .enumerate()
            .map(|(idx, &l)| {
                let p = (l - max_val).exp();
                sum_exp += p;
                (idx as u32, p)
            })
            .collect();

        let inv_sum = 1.0f32 / sum_exp.max(1e-12);
        for (_, p) in probs.iter_mut() {
            *p *= inv_sum;
        }

        // Sort descending by probability
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // 4. Top-K cutoff
        if params.top_k > 0 && params.top_k < probs.len() {
            probs.truncate(params.top_k);
        }

        // 5. Top-P (nucleus) cutoff
        if params.top_p < 1.0 {
            let mut cum_sum = 0.0f32;
            let mut cutoff_idx = probs.len();
            for (i, (_, p)) in probs.iter().enumerate() {
                cum_sum += *p;
                if cum_sum >= params.top_p {
                    cutoff_idx = (i + 1).min(probs.len());
                    break;
                }
            }
            probs.truncate(cutoff_idx);
        }

        // Re-normalize remaining probabilities
        let remaining_sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        let inv_rem = 1.0f32 / remaining_sum.max(1e-12);
        for (_, p) in probs.iter_mut() {
            *p *= inv_rem;
        }

        // 6. Categorical random sampling
        let mut rng = rand::thread_rng();
        let sample_val: f32 = rng.gen();
        let mut accum = 0.0f32;
        for (token_id, p) in probs {
            accum += p;
            if sample_val <= accum {
                return token_id;
            }
        }

        0
    }
}
