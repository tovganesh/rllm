//! Byte-level BPE tokenizer loaded from GGUF metadata (`tokenizer.ggml.model = "gpt2"`).
//!
//! Mirrors llama.cpp's BPE tokenizer:
//! 1. Text is split around special tokens (control / user-defined vocabulary entries).
//! 2. Ordinary text is pre-tokenized with the regex set selected by `tokenizer.ggml.pre`.
//!    Regexes are applied in sequence; each splits every current piece into matches and
//!    the unmatched text between them.
//! 3. Each piece is mapped through the GPT-2 byte-to-unicode table and merged greedily by
//!    merge rank.

use crate::gguf::{GgufFile, GgufValue};
use fancy_regex::Regex;
use rllm_core::error::{Result, RllmError};
use std::collections::HashMap;

/// GGUF `tokenizer.ggml.token_type` values.
mod token_type {
    pub const CONTROL: i64 = 3;
    pub const USER_DEFINED: i64 = 4;
}

const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";
const LLAMA3_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN2_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Pre-tokenizer regexes and options for a `tokenizer.ggml.pre` name.
fn pre_tokenizer(pre: &str) -> Option<(Vec<&'static str>, bool)> {
    // (regexes applied in order, ignore_merges)
    Some(match pre {
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "pixtral" => (vec![LLAMA3_PATTERN], true),
        "qwen2" | "deepseek-r1-qwen" | "megrez" => (vec![QWEN2_PATTERN], false),
        "smollm" | "starcoder" | "refact" | "codeshell" | "exaone" | "minerva-7b" => {
            (vec![r"\p{N}", GPT2_PATTERN], false)
        }
        "gpt-2" | "gpt2" | "mpt" | "olmo" | "jais" | "trillion" => (vec![GPT2_PATTERN], false),
        _ => return None,
    })
}

/// GPT-2 byte <-> printable unicode mapping.
fn byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..=255u32 {
        let printable = (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    table
}

/// Byte-level BPE tokenizer.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    is_special: Vec<bool>,
    /// Special token strings sorted longest-first for greedy matching.
    specials_by_len: Vec<(String, u32)>,
    merges: HashMap<(u32, u32), (u32, u32)>, // (left, right) -> (rank, merged id)
    pre_regexes: Vec<Regex>,
    ignore_merges: bool,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    byte_fallback_ids: [Option<u32>; 256],
    /// Beginning-of-sequence token ID.
    pub bos_token_id: Option<u32>,
    /// End-of-sequence token ID.
    pub eos_token_id: Option<u32>,
    /// Whether `encode(.., add_special = true)` prepends BOS.
    pub add_bos: bool,
    /// Whether `encode(.., add_special = true)` appends EOS.
    pub add_eos: bool,
    /// Tokens that end generation (EOS, EOT and known end-of-turn markers).
    pub stop_token_ids: Vec<u32>,
}

impl BpeTokenizer {
    /// Builds the tokenizer from GGUF metadata.
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let model = g.get_str("tokenizer.ggml.model")?;
        if model != "gpt2" {
            return Err(RllmError::ModelError(format!(
                "Unsupported tokenizer model '{model}' (only byte-level BPE 'gpt2' is supported)"
            )));
        }
        let pre = g
            .get("tokenizer.ggml.pre")
            .and_then(GgufValue::as_str)
            .unwrap_or("default");
        let (patterns, ignore_merges) = pre_tokenizer(pre).ok_or_else(|| {
            RllmError::ModelError(format!("Unsupported BPE pre-tokenizer '{pre}'"))
        })?;

        let tokens: Vec<String> = string_array(g, "tokenizer.ggml.tokens")?;
        let types: Vec<i64> = match g.get("tokenizer.ggml.token_type").and_then(GgufValue::as_array) {
            Some(arr) => arr.iter().map(|v| v.as_u64().map_or(1, |x| x as i64)).collect(),
            None => vec![1; tokens.len()],
        };
        let merges_raw = string_array(g, "tokenizer.ggml.merges")?;

        let mut tok = Self::build(tokens, &types, &merges_raw, &patterns, ignore_merges)?;
        let id = |k: &str| g.get(k).and_then(GgufValue::as_u64).map(|v| v as u32);
        let flag = |k: &str| g.get(k).and_then(GgufValue::as_bool);
        tok.bos_token_id = id("tokenizer.ggml.bos_token_id");
        tok.eos_token_id = id("tokenizer.ggml.eos_token_id");
        // Without explicit metadata, llama.cpp adds BOS for the Llama 3 pre-tokenizer
        // family (e.g. Llama 3.x GGUFs have no add_bos_token key but expect BOS).
        let llama3_family = matches!(pre, "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "pixtral");
        tok.add_bos = flag("tokenizer.ggml.add_bos_token").unwrap_or(llama3_family);
        tok.add_eos = flag("tokenizer.ggml.add_eos_token").unwrap_or(false);

        let mut stops: Vec<u32> = ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"]
            .iter()
            .filter_map(|k| id(k))
            .collect();
        for marker in ["<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|end_of_text|>", "<|eom_id|>"] {
            if let Some(&i) = tok.token_to_id.get(marker) {
                if tok.is_special[i as usize] {
                    stops.push(i);
                }
            }
        }
        stops.sort_unstable();
        stops.dedup();
        tok.stop_token_ids = stops;
        Ok(tok)
    }

    fn build(
        tokens: Vec<String>,
        types: &[i64],
        merges_raw: &[String],
        patterns: &[&str],
        ignore_merges: bool,
    ) -> Result<Self> {
        let token_to_id: HashMap<String, u32> =
            tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();

        let is_special: Vec<bool> = (0..tokens.len())
            .map(|i| matches!(types.get(i), Some(&token_type::CONTROL) | Some(&token_type::USER_DEFINED)))
            .collect();
        let mut specials_by_len: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, t)| is_special[*i] && !t.is_empty())
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials_by_len.sort_by_key(|s| std::cmp::Reverse(s.0.len()));

        let mut merges = HashMap::with_capacity(merges_raw.len());
        for (rank, m) in merges_raw.iter().enumerate() {
            // Merge entries are "left right"; the separator is the first space after the
            // first character (which may be multi-byte, e.g. 'Ġ').
            let Some(first) = m.chars().next() else { continue };
            let skip = first.len_utf8();
            let Some(split) = m[skip..].find(' ').map(|p| p + skip) else { continue };
            let (l, r) = (&m[..split], &m[split + 1..]);
            let merged = format!("{l}{r}");
            if let (Some(&li), Some(&ri), Some(&mi)) =
                (token_to_id.get(l), token_to_id.get(r), token_to_id.get(&merged))
            {
                merges.entry((li, ri)).or_insert((rank as u32, mi));
            }
        }

        let pre_regexes = patterns
            .iter()
            .map(|p| Regex::new(p).map_err(|e| RllmError::ModelError(format!("bad pre-tokenizer regex: {e}"))))
            .collect::<Result<Vec<_>>>()?;

        let byte_encoder = byte_to_unicode();
        let byte_decoder = byte_encoder.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let mut byte_fallback_ids = [None; 256];
        for (b, c) in byte_encoder.iter().enumerate() {
            byte_fallback_ids[b] = token_to_id.get(&c.to_string()).copied();
        }

        Ok(Self {
            tokens,
            token_to_id,
            is_special,
            specials_by_len,
            merges,
            pre_regexes,
            ignore_merges,
            byte_encoder,
            byte_decoder,
            byte_fallback_ids,
            bos_token_id: None,
            eos_token_id: None,
            add_bos: false,
            add_eos: false,
            stop_token_ids: Vec::new(),
        })
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Token ID of an exact vocabulary entry.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Encodes text to token IDs.
    ///
    /// * `add_special` — add BOS/EOS according to the model's `add_bos`/`add_eos` flags.
    /// * `parse_special` — recognise special-token strings (e.g. `<|im_start|>`) in the text.
    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_special && self.add_bos {
            out.extend(self.bos_token_id);
        }
        if parse_special {
            let mut rest = text;
            while !rest.is_empty() {
                match self.find_special(rest) {
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
        } else {
            self.encode_ordinary(text, &mut out);
        }
        if add_special && self.add_eos {
            out.extend(self.eos_token_id);
        }
        out
    }

    /// Earliest special-token occurrence: `(byte position, byte length, id)`.
    fn find_special(&self, text: &str) -> Option<(usize, usize, u32)> {
        let mut best: Option<(usize, usize, u32)> = None;
        for (s, id) in &self.specials_by_len {
            if let Some(pos) = text.find(s.as_str()) {
                // Earliest wins; on ties the longest (checked first) is kept.
                if best.is_none_or(|(bp, _, _)| pos < bp) {
                    best = Some((pos, s.len(), *id));
                }
            }
        }
        best
    }

    fn encode_ordinary(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        for piece in self.pre_tokenize(text) {
            self.bpe_piece(piece, out);
        }
    }

    /// Splits text with the pre-tokenizer regexes (matches and the gaps between them).
    fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut pieces = vec![text];
        for re in &self.pre_regexes {
            let mut next = Vec::with_capacity(pieces.len());
            for piece in pieces {
                let mut last = 0;
                for m in re.find_iter(piece).flatten() {
                    if m.start() > last {
                        next.push(&piece[last..m.start()]);
                    }
                    if m.end() > m.start() {
                        next.push(m.as_str());
                    }
                    last = m.end();
                }
                if last < piece.len() {
                    next.push(&piece[last..]);
                }
            }
            pieces = next;
        }
        pieces
    }

    fn bpe_piece(&self, piece: &str, out: &mut Vec<u32>) {
        let mapped: String = piece.bytes().map(|b| self.byte_encoder[b as usize]).collect();
        if self.ignore_merges {
            if let Some(&id) = self.token_to_id.get(&mapped) {
                out.push(id);
                return;
            }
        }

        // Start from one symbol per mapped character (i.e. per byte).
        let mut symbols: Vec<u32> = Vec::with_capacity(piece.len());
        for (ch, b) in mapped.chars().zip(piece.bytes()) {
            match self.token_to_id.get(ch.encode_utf8(&mut [0; 4]) as &str) {
                Some(&id) => symbols.push(id),
                None => symbols.extend(self.byte_fallback_ids[b as usize]),
            }
        }

        // Repeatedly apply the lowest-ranked available merge.
        loop {
            let mut best: Option<(u32, usize, u32)> = None; // (rank, index, merged id)
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(&(rank, merged)) = self.merges.get(&(symbols[i], symbols[i + 1])) {
                    if best.is_none_or(|(r, _, _)| rank < r) {
                        best = Some((rank, i, merged));
                    }
                }
            }
            let Some((_, i, merged)) = best else { break };
            symbols[i] = merged;
            symbols.remove(i + 1);
        }
        out.extend_from_slice(&symbols);
    }

    /// Raw bytes of one token (special tokens yield their literal text).
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(tok) = self.tokens.get(id as usize) else { return Vec::new() };
        if self.is_special[id as usize] {
            return tok.as_bytes().to_vec();
        }
        let mut bytes = Vec::with_capacity(tok.len());
        for ch in tok.chars() {
            match self.byte_decoder.get(&ch) {
                Some(&b) => bytes.push(b),
                None => bytes.extend_from_slice(ch.encode_utf8(&mut [0; 4]).as_bytes()),
            }
        }
        bytes
    }

    /// Whether a token is a control / user-defined special token.
    pub fn is_special(&self, id: u32) -> bool {
        self.is_special.get(id as usize).copied().unwrap_or(false)
    }

    /// Decodes token IDs to text. With `skip_special`, special tokens are omitted.
    /// Invalid UTF-8 sequences are replaced with U+FFFD.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if skip_special && self.is_special(id) {
                continue;
            }
            bytes.extend(self.token_bytes(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Incremental decoder that only emits complete UTF-8 characters, for streaming.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    /// Adds one token and returns any newly completed text.
    pub fn push(&mut self, tok: &BpeTokenizer, id: u32, skip_special: bool) -> String {
        if skip_special && tok.is_special(id) {
            return String::new();
        }
        self.pending.extend(tok.token_bytes(id));
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(s) => s.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(), // incomplete tail
            Err(_) => {
                // Genuinely invalid bytes: flush lossily.
                let s = String::from_utf8_lossy(&self.pending).into_owned();
                self.pending.clear();
                return s;
            }
        };
        let text = String::from_utf8(self.pending[..valid].to_vec()).unwrap_or_default();
        self.pending.drain(..valid);
        text
    }

    /// Flushes any buffered bytes (lossily).
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

fn string_array(g: &GgufFile, key: &str) -> Result<Vec<String>> {
    let arr = g
        .get(key)
        .and_then(GgufValue::as_array)
        .ok_or_else(|| RllmError::ModelError(format!("GGUF metadata '{key}' missing or not an array")))?;
    arr.iter()
        .map(|v| {
            v.as_str()
                .map(str::to_string)
                .ok_or_else(|| RllmError::ModelError(format!("'{key}' contains a non-string")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny vocabulary: all single bytes as tokens plus a few merges.
    fn toy() -> BpeTokenizer {
        let enc = byte_to_unicode();
        let mut tokens: Vec<String> = vec!["<|special|>".into()];
        tokens.extend(enc.iter().map(|c| c.to_string()));
        let g = |s: &str| s.bytes().map(|b| enc[b as usize]).collect::<String>();
        for t in ["he", "ll", "hell", "hello", " w", " wo"] {
            tokens.push(g(t));
        }
        let mut types = vec![1i64; tokens.len()];
        types[0] = token_type::CONTROL;
        let merges: Vec<String> = [("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o"), (" ", "w"), (" w", "o")]
            .iter()
            .map(|(a, b)| format!("{} {}", g(a), g(b)))
            .collect();
        BpeTokenizer::build(tokens, &types, &merges, &[GPT2_PATTERN], false).unwrap()
    }

    #[test]
    fn merges_by_rank_and_round_trips() {
        let t = toy();
        let ids = t.encode("hello world", false, false);
        // "hello" fully merges; " world" -> " wo" + "r" + "l" + "d".
        assert_eq!(t.decode(&ids, false), "hello world");
        assert_eq!(ids[0], t.token_id("hello").unwrap());
        assert_eq!(ids.len(), 1 + 4);
    }

    #[test]
    fn special_tokens_are_parsed_only_when_requested() {
        let t = toy();
        let with = t.encode("a<|special|>b", false, true);
        assert_eq!(with, vec![t.token_id("a").unwrap(), 0, t.token_id("b").unwrap()]);
        let without = t.encode("a<|special|>b", false, false);
        assert!(!without.contains(&0));
        assert_eq!(t.decode(&with, true), "ab");
        assert_eq!(t.decode(&with, false), "a<|special|>b");
    }

    #[test]
    fn utf8_round_trip_and_streaming() {
        let t = toy();
        let text = "naïve café 🦀 — ok";
        let ids = t.encode(text, false, false);
        assert_eq!(t.decode(&ids, false), text);
        let mut sd = StreamDecoder::default();
        let streamed: String = ids.iter().map(|&i| sd.push(&t, i, true)).collect::<String>() + &sd.finish();
        assert_eq!(streamed, text);
    }
}
