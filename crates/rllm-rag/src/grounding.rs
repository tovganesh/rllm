//! Precise passage grounding, sentence extraction, and relevance verification.
//!
//! Provides utilities to extract the exact grounded sentences from document passages
//! relevant to a query, filter stop words, and detect low-relevance or unrelated queries.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Standard English stop words filtered during content keyword extraction.
pub const STOP_WORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "should", "shouldn't", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "use", "used", "uses", "using", "very", "was", "wasn't", "we", "were", "weren't", "what", "when", "where",
    "which", "while", "who", "whom", "why", "with", "won't", "would", "wouldn't", "you",
    "your", "yours", "yourself", "yourselves",
];

/// Result of extracting a grounded quotation from a document passage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundedQuote {
    /// Exact quotation / sentence from the source document that answers or matches the query.
    pub quote: String,
    /// List of content keywords that appear in both the query and the quotation.
    pub matched_keywords: Vec<String>,
    /// Sentence index within the passage (0-based).
    pub sentence_index: usize,
    /// Grounding relevance score between 0.0 and 1.0.
    pub score: f32,
}

/// Extracts meaningful content keywords from text, excluding common English stop words.
pub fn extract_content_keywords(text: &str) -> Vec<String> {
    let stop_set: HashSet<&'static str> = STOP_WORDS.iter().copied().collect();
    let mut words: Vec<String> = text
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2 && !stop_set.contains(*w))
        .map(|w| w.to_string())
        .collect();

    // Deduplicate while preserving order
    let mut seen = HashSet::new();
    words.retain(|w| seen.insert(w.clone()));
    words
}

/// Splits a document passage into individual sentences based on standard delimiters.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);
        if c == '.' || c == '!' || c == '?' || c == '\n' || c == ';' {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_string());
            }
            current.clear();
        }
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_string());
    }

    if sentences.is_empty() && !text.trim().is_empty() {
        sentences.push(text.trim().to_string());
    }
    sentences
}

/// Extracts the most precise and relevant grounded sentence quotation from a passage.
///
/// If multiple consecutive sentences provide complementary context and match query keywords,
/// combines them into a coherent grounded excerpt (up to ~300 characters).
pub fn extract_grounded_quote(query: &str, passage_text: &str) -> GroundedQuote {
    let query_keywords = extract_content_keywords(query);
    let sentences = split_sentences(passage_text);

    if sentences.is_empty() {
        return GroundedQuote {
            quote: passage_text.to_string(),
            matched_keywords: Vec::new(),
            sentence_index: 0,
            score: 0.0,
        };
    }

    if query_keywords.is_empty() {
        // Query has no content keywords; default to first sentence
        return GroundedQuote {
            quote: sentences[0].clone(),
            matched_keywords: Vec::new(),
            sentence_index: 0,
            score: 0.0,
        };
    }

    let mut best_score = -1.0f32;
    let mut best_idx = 0;

    for (idx, sentence) in sentences.iter().enumerate() {
        let sent_words: Vec<String> = sentence
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(|w| w.to_string())
            .collect();

        let mut matched = Vec::new();
        for qk in &query_keywords {
            if sent_words.iter().any(|sw| sw == qk || (qk.len() >= 4 && sw.len() >= 4 && (sw.starts_with(qk) || qk.starts_with(sw)))) {
                matched.push(qk.clone());
            }
        }
        matched.sort();
        matched.dedup();

        let coverage = matched.len() as f32 / query_keywords.len() as f32;
        let density = matched.len() as f32 / (sent_words.len() as f32 + 5.0);

        // Check exact phrase matching bonus
        let mut phrase_bonus = 0.0f32;
        if query_keywords.len() >= 2 {
            let sent_lower = sentence.to_lowercase();
            let query_lower = query.to_lowercase();
            if sent_lower.contains(&query_lower) {
                phrase_bonus = 0.25;
            }
        }

        let score = (coverage * 0.65 + density * 0.20 + phrase_bonus).min(1.0);

        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }

    // Check if next sentence also adds valuable context
    let mut quote = sentences[best_idx].clone();
    if best_idx + 1 < sentences.len() {
        let next_sent = &sentences[best_idx + 1];
        let next_words: Vec<String> = next_sent
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(|w| w.to_string())
            .collect();

        let next_has_keyword = query_keywords
            .iter()
            .any(|qk| next_words.iter().any(|nw| nw == qk || nw.starts_with(qk)));

        if next_has_keyword && (quote.len() + next_sent.len() <= 320) {
            quote.push(' ');
            quote.push_str(next_sent);
        }
    }

    // Re-verify all matched keywords within the final quote
    let quote_lower = quote.to_lowercase();
    let final_matched: Vec<String> = query_keywords
        .into_iter()
        .filter(|qk| quote_lower.contains(qk.as_str()))
        .collect();

    GroundedQuote {
        quote,
        matched_keywords: final_matched,
        sentence_index: best_idx,
        score: best_score.max(0.0),
    }
}
