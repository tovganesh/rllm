//! Chat prompt formatting with the model's Jinja chat template.
//!
//! GGUF files carry the Hugging Face chat template in `tokenizer.chat_template`. It is
//! rendered with [minijinja] (plus Python-compatible string/dict methods), with the
//! variables HF templates expect: `messages`, `add_generation_prompt`, `bos_token`,
//! `eos_token` and the `raise_exception` function. Files without a template fall back
//! to ChatML.

use crate::bpe::BpeTokenizer;
use crate::gguf::{GgufFile, GgufValue};
use minijinja::{context, Environment, Error, ErrorKind};
use rllm_core::error::{Result, RllmError};
use serde::{Deserialize, Serialize};

/// A single chat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `system`, `user` or `assistant`.
    pub role: String,
    /// Message text.
    pub content: String,
}

impl ChatMessage {
    /// Convenience constructor.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// ChatML template used when a model provides none.
pub const CHATML_TEMPLATE: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

/// A compiled chat template.
pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate").finish_non_exhaustive()
    }
}

impl ChatTemplate {
    /// Compiles a template source.
    pub fn new(source: &str, bos_token: &str, eos_token: &str) -> Result<Self> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> std::result::Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_template_owned("chat", source.to_string())
            .map_err(|e| RllmError::ModelError(format!("Invalid chat template: {e}")))?;
        Ok(Self {
            env,
            bos_token: bos_token.to_string(),
            eos_token: eos_token.to_string(),
        })
    }

    /// The template from `tokenizer.chat_template`, or ChatML if absent.
    pub fn from_gguf(g: &GgufFile, tokenizer: &BpeTokenizer) -> Result<Self> {
        let source = g
            .get("tokenizer.chat_template")
            .and_then(GgufValue::as_str)
            .unwrap_or(CHATML_TEMPLATE);
        let text = |id: Option<u32>| id.map(|i| tokenizer.decode(&[i], false)).unwrap_or_default();
        Self::new(source, &text(tokenizer.bos_token_id), &text(tokenizer.eos_token_id))
    }

    /// Renders the conversation into a prompt string.
    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| RllmError::ModelError(e.to_string()))?;
        tmpl.render(context! {
            messages => minijinja::value::Serde(messages),
            add_generation_prompt => add_generation_prompt,
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
        })
        .map_err(|e| RllmError::ModelError(format!("Chat template rendering failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_fallback_renders_expected_prompt() {
        let t = ChatTemplate::new(CHATML_TEMPLATE, "", "").unwrap();
        let msgs = [ChatMessage::new("system", "Be brief."), ChatMessage::new("user", "Hi")];
        assert_eq!(
            t.render(&msgs, true).unwrap(),
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(!t.render(&msgs, false).unwrap().ends_with("assistant\n"));
    }

    #[test]
    fn supports_python_methods_bos_and_raise_exception() {
        let src = "{{ bos_token }}{% for m in messages %}{% if m['role'] not in ['user', 'assistant'] %}{{ raise_exception('bad role') }}{% endif %}[{{ m['role'].upper() }}] {{ m['content'].strip() }}\n{% endfor %}";
        let t = ChatTemplate::new(src, "<s>", "</s>").unwrap();
        let ok = t.render(&[ChatMessage::new("user", "  hi  ")], false).unwrap();
        assert_eq!(ok, "<s>[USER] hi\n");
        let err = t.render(&[ChatMessage::new("tool", "x")], false).unwrap_err();
        assert!(err.to_string().contains("bad role"));
    }
}
