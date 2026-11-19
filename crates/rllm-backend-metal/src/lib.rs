//! # `rllm-backend-metal`
//!
//! Apple GPU executor for LLaMA-architecture models ([`MetalLlama`]), implementing
//! [`rllm_models::LanguageModel`] so it plugs into the continuous-batching scheduler.
//! The whole forward pass is encoded into a single Metal command buffer; weights and the
//! paged KV cache live in GPU-visible memory.
//!
//! Only available on macOS; on other platforms the crate is empty.

#[cfg(target_os = "macos")]
mod llama;
#[cfg(target_os = "macos")]
pub mod shaders;

#[cfg(target_os = "macos")]
pub use llama::MetalLlama;
