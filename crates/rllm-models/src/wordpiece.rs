//! WordPiece tokenizer for BERT-family models, loaded from GGUF metadata
//! (`tokenizer.ggml.model = "bert"`).
//!
//! Mirrors llama.cpp's WPM tokenizer:
//! 1. Text is split around special tokens (`[CLS]`, `[SEP]`, ...).
//! 2. Ordinary text is NFD-normalised, lower-cased and stripped of combining marks;
//!    control characters are dropped; whitespace separates words; punctuation, ASCII
//!    symbols and CJK ideographs become single-character words.
//! 3. Each word is prefixed with `▁` (GGUF stores word-initial pieces as `▁piece` and
//!    continuation pieces bare) and split greedily by longest vocabulary match. A word
//!    that cannot be fully matched becomes `[UNK]`.

use crate::gguf::{GgufFile, GgufValue};
use rllm_core::error::{Result, RllmError};
use std::collections::HashMap;
use unicode_general_category::{get_general_category, GeneralCategory as Gc};
use unicode_normalization::UnicodeNormalization;

const WORD_START: char = '\u{2581}';

/// WordPiece tokenizer.
#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    specials: Vec<(String, u32)>,
    is_special: Vec<bool>,
    max_token_chars: usize,
    /// `[CLS]` / BOS token ID.
    pub cls_token_id: u32,
    /// `[SEP]` / EOS token ID.
    pub sep_token_id: u32,
    /// `[UNK]` token ID.
    pub unk_token_id: u32,
}

impl WordPieceTokenizer {
    /// Builds the tokenizer from GGUF metadata.
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let model = g.get_str("tokenizer.ggml.model")?;
        if model != "bert" {
            return Err(RllmError::ModelError(format!(
                "Expected a WordPiece ('bert') tokenizer, found '{model}'"
            )));
        }
        let tokens: Vec<String> = g
            .get("tokenizer.ggml.tokens")
            .and_then(GgufValue::as_array)
            .ok_or_else(|| RllmError::ModelError("tokenizer.ggml.tokens missing".to_string()))?
            .iter()
            .map(|v| v.as_str().unwrap_or_default().to_string())
            .collect();
        let types: Vec<u64> = g
            .get("tokenizer.ggml.token_type")
            .and_then(GgufValue::as_array)
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(1)).collect())
            .unwrap_or_else(|| vec![1; tokens.len()]);
        let id = |k: &str, default: u32| g.get(k).and_then(GgufValue::as_u64).map_or(default, |v| v as u32);
        Ok(Self::build(
            tokens,
            &types,
            id("tokenizer.ggml.cls_token_id", id("tokenizer.ggml.bos_token_id", 101)),
            id(
                "tokenizer.ggml.seperator_token_id",
                id("tokenizer.ggml.eos_token_id", 102),
            ),
            id("tokenizer.ggml.unknown_token_id", 100),
        ))
    }

    fn build(tokens: Vec<String>, types: &[u64], cls: u32, sep: u32, unk: u32) -> Self {
        let token_to_id = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        // GGUF token types: 3 = control, 4 = user-defined.
        let is_special: Vec<bool> = (0..tokens.len())
            .map(|i| matches!(types.get(i), Some(3) | Some(4)))
            .collect();
        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, t)| is_special[*i] && !t.is_empty())
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by_key(|s| std::cmp::Reverse(s.0.len()));
        let max_token_chars = tokens.iter().map(|t| t.chars().count()).max().unwrap_or(1);
        Self {
            tokens,
            token_to_id,
            specials,
            is_special,
            max_token_chars,
            cls_token_id: cls,
            sep_token_id: sep,
            unk_token_id: unk,
        }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Encodes text. With `add_special`, wraps the result in `[CLS] … [SEP]`.
    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_special {
            out.push(self.cls_token_id);
        }
        let mut rest = text;
        while !rest.is_empty() {
            let found = if parse_special { self.find_special(rest) } else { None };
            match found {
                Some((pos, len, id)) => {
                    self.encode_ordinary(&rest[..pos], &mut out);
                    out.push(id);
                    rest = &rest[pos + len..];
                }
                None => {
                    self.encode_ordinary(rest, &mut out);
                    break;
                }
            }
        }
        if add_special {
            out.push(self.sep_token_id);
        }
        out
    }

    fn find_special(&self, text: &str) -> Option<(usize, usize, u32)> {
        let mut best: Option<(usize, usize, u32)> = None;
        for (s, id) in &self.specials {
            if let Some(pos) = text.find(s.as_str()) {
                if best.is_none_or(|(bp, _, _)| pos < bp) {
                    best = Some((pos, s.len(), *id));
                }
            }
        }
        best
    }

    fn encode_ordinary(&self, text: &str, out: &mut Vec<u32>) {
        for word in pre_tokenize(text) {
            let chars: Vec<char> = std::iter::once(WORD_START).chain(word.chars()).collect();
            let start_len = out.len();
            let mut i = 0;
            while i < chars.len() {
                let mut matched = false;
                let upper = chars.len().min(i + self.max_token_chars);
                for j in (i + 1..=upper).rev() {
                    let piece: String = chars[i..j].iter().collect();
                    if let Some(&id) = self.token_to_id.get(&piece) {
                        out.push(id);
                        i = j;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    out.truncate(start_len);
                    break;
                }
            }
            if out.len() == start_len {
                out.push(self.unk_token_id);
            }
        }
    }

    /// Decodes IDs to text (word-initial pieces start a new space-separated word).
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut s = String::new();
        for &id in ids {
            let Some(tok) = self.tokens.get(id as usize) else {
                continue;
            };
            if self.is_special[id as usize] {
                if !skip_special {
                    s.push_str(tok);
                }
                continue;
            }
            match tok.strip_prefix(WORD_START) {
                Some(word) => {
                    if !s.is_empty() {
                        s.push(' ');
                    }
                    s.push_str(word);
                }
                None => s.push_str(tok),
            }
        }
        s
    }
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F | 0x2B820..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

fn is_punctuation(c: char) -> bool {
    matches!(
        get_general_category(c),
        Gc::ConnectorPunctuation
            | Gc::DashPunctuation
            | Gc::OpenPunctuation
            | Gc::ClosePunctuation
            | Gc::InitialPunctuation
            | Gc::FinalPunctuation
            | Gc::OtherPunctuation
    )
}

fn is_symbol(c: char) -> bool {
    matches!(
        get_general_category(c),
        Gc::MathSymbol | Gc::CurrencySymbol | Gc::ModifierSymbol | Gc::OtherSymbol
    )
}

/// BERT basic tokenization: normalise, lower-case, strip accents and split into words.
fn pre_tokenize(text: &str) -> Vec<String> {
    let mut words = vec![String::new()];
    for c in text.nfd() {
        if c.is_whitespace() {
            if !words.last().unwrap().is_empty() {
                words.push(String::new());
            }
            continue;
        }
        let cat = get_general_category(c);
        if c == '\0' || c == '\u{FFFD}' || matches!(cat, Gc::Control | Gc::Format) || cat == Gc::NonspacingMark {
            continue;
        }
        let lower: String = c.to_lowercase().collect();
        if is_punctuation(c) || ((c as u32) < 0x7F && is_symbol(c)) || is_cjk(c) {
            if !words.last().unwrap().is_empty() {
                words.push(String::new());
            }
            *words.last_mut().unwrap() = lower;
            words.push(String::new());
        } else {
            words.last_mut().unwrap().push_str(&lower);
        }
    }
    if words.last().is_some_and(String::is_empty) {
        words.pop();
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> WordPieceTokenizer {
        let tokens: Vec<String> = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "▁hello", "▁,", "▁world", "▁!", "▁una", "ffa", "ble", "▁naive", "▁日",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mut types = vec![1u64; tokens.len()];
        types[..4].fill(3);
        WordPieceTokenizer::build(tokens, &types, 2, 3, 1)
    }

    #[test]
    fn basic_tokenization_matches_bert_rules() {
        let t = toy();
        assert_eq!(t.encode("Hello, WORLD!", true, true), vec![2, 4, 5, 6, 7, 3]);
        assert_eq!(t.encode("unaffable", false, false), vec![8, 9, 10]);
        assert_eq!(t.encode("naïve", false, false), vec![11], "accents are stripped");
        assert_eq!(t.encode("日 unknownword", false, false), vec![12, 1]);
        assert_eq!(t.encode("[CLS]hello", false, true), vec![2, 4]);
        assert_eq!(t.decode(&[2, 4, 5, 8, 9, 10, 3], true), "hello , unaffable");
    }
}
