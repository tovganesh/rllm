//! # `rllm-backend-metal`
//!
//! Apple Silicon & iOS Metal backend for hardware-accelerated LLM execution.
//!
//! Exploits unified memory architecture (UMA) where CPU and GPU share physical RAM,
//! allowing zero-copy tensor passing and Metal Shading Language (MSL) compute shaders.

pub mod backend;
pub mod shaders;

pub use backend::MetalBackend;
