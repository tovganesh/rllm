//! Perplexity evaluation compatible with `llama-perplexity`.

use crate::llama::{ForwardBatch, LlamaModel};
use rllm_core::{
    backend::{Backend, BlockTable},
    dtype::DType,
    error::{Result, RllmError},
    shape::Shape,
    tensor::Tensor,
};

/// Perplexity result.
#[derive(Debug, Clone, Copy)]
pub struct Perplexity {
    /// `exp(mean negative log-likelihood)`.
    pub ppl: f64,
    /// Number of scored tokens.
    pub scored_tokens: usize,
    /// Number of chunks.
    pub chunks: usize,
}

/// Computes perplexity with llama.cpp's protocol: split `tokens` into non-overlapping
/// chunks of `ctx` tokens (replacing the first token of each chunk with `bos` if
/// given), run each chunk in one forward pass, and score the predictions for the
/// second half of every chunk.
pub fn perplexity<B: Backend>(model: &LlamaModel<B>, tokens: &[u32], ctx: usize, bos: Option<u32>) -> Result<Perplexity> {
    if ctx < 4 || ctx > model.config.max_position_embeddings {
        return Err(RllmError::ModelError(format!(
            "ctx must be between 4 and the model context ({})",
            model.config.max_position_embeddings
        )));
    }
    let chunks = tokens.len() / ctx;
    if chunks == 0 {
        return Err(RllmError::ModelError(format!("{} tokens is fewer than ctx {}", tokens.len(), ctx)));
    }
    let c = &model.config;
    let block = 16;
    let blocks = ctx.div_ceil(block);
    let shape = Shape::new(vec![blocks, c.num_key_value_heads, block, c.head_dim()]);
    let table = BlockTable { block_size: block, block_ids: (0..blocks).collect() };
    let tables = vec![table; ctx];
    let positions: Vec<usize> = (0..ctx).collect();
    let first = ctx / 2;
    let rows: Vec<usize> = (first..ctx - 1).collect();
    let dev = model.backend.device();

    let (mut nll, mut count) = (0f64, 0usize);
    for chunk in 0..chunks {
        let mut toks = tokens[chunk * ctx..(chunk + 1) * ctx].to_vec();
        if let Some(b) = bos {
            toks[0] = b;
        }
        let mut k: Vec<Tensor> = (0..c.num_hidden_layers)
            .map(|_| Tensor::zeros(shape.clone(), DType::F32, dev))
            .collect::<Result<_>>()?;
        let mut v = k.clone();
        let logits = model
            .forward(
                &ForwardBatch { tokens: &toks, positions: &positions, block_tables: &tables, logits_rows: &rows },
                &mut k,
                &mut v,
            )?
            .to_vec::<f32>()?;
        for (i, row) in logits.chunks(c.vocab_size).enumerate() {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = max + row.iter().map(|&x| (x as f64 - max).exp()).sum::<f64>().ln();
            nll += lse - row[toks[first + i + 1] as usize] as f64;
            count += 1;
        }
    }
    Ok(Perplexity { ppl: (nll / count as f64).exp(), scored_tokens: count, chunks })
}
