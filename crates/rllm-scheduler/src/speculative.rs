//! Prompt-lookup speculative decoding (opt-in).
//!
//! Many generations copy spans of their own context: a summary quotes the source, an
//! edit repeats the code it changes, a RAG answer reuses the passage. Prompt lookup
//! (n-gram speculation) exploits that without a draft model: find the most recent
//! earlier occurrence of the sequence's last `n` tokens and propose the tokens that
//! followed it as a draft.
//!
//! The scheduler appends the draft after the newest token and runs all of them in the
//! same forward pass. Row `i` of the logits is the distribution of the token that
//! follows draft token `i - 1`, so the tokens are sampled one by one exactly as plain
//! decoding would sample them; while a sampled token equals the next draft token, the
//! draft's KV entry is already valid and the next row can be used too. The first
//! mismatch ends the step (its sampled token is still emitted, so every step produces
//! at least one token). Because each emitted token is drawn from the model's own
//! distribution given the accepted prefix, the output distribution is unchanged for any
//! sampler — greedy, temperature, top-k/top-p and repetition penalty alike. Only the
//! arithmetic differs: a verification pass runs batched kernels, so greedy decoding can
//! break a near-tie differently than one-token-at-a-time decoding would.
//!
//! Verification is not free, and its price depends on the executor and weight format
//! (on Metal 2–8 rows cost about 1.3× one row; CPU K-quant kernels pay ~3× for 8 rows).
//! So `max_draft_tokens` is only an upper bound: the scheduler times decode passes by
//! row count ([`PassCosts`]), tracks each sequence's acceptance rate ([`DraftStats`])
//! and keeps the draft length with the best expected tokens per second, which may be
//! no draft at all.

/// Settings for prompt-lookup speculation. Off unless `max_draft_tokens > 0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculationConfig {
    /// Most draft tokens proposed per sequence per step; `0` disables speculation.
    pub max_draft_tokens: usize,
    /// Shortest suffix n-gram that may be matched.
    pub min_ngram: usize,
    /// Longest suffix n-gram tried first.
    pub max_ngram: usize,
    /// Pick each draft's length from measured pass costs and acceptance (default).
    /// When false, every lookup match is verified at full length (up to
    /// `max_draft_tokens`), which is deterministic but can be slower than not drafting.
    pub adaptive: bool,
}

impl Default for SpeculationConfig {
    fn default() -> Self {
        Self {
            max_draft_tokens: 0,
            min_ngram: 1,
            max_ngram: 3,
            adaptive: true,
        }
    }
}

impl SpeculationConfig {
    /// Prompt lookup with `max_draft_tokens` drafts and the default n-gram range.
    pub fn prompt_lookup(max_draft_tokens: usize) -> Self {
        Self {
            max_draft_tokens,
            ..Self::default()
        }
    }

    /// Whether speculation is enabled.
    pub fn enabled(&self) -> bool {
        self.max_draft_tokens > 0 && self.min_ngram > 0 && self.min_ngram <= self.max_ngram
    }
}

/// Proposes up to `max_draft` tokens to follow `tokens`.
///
/// Tries suffix n-grams from `max_ngram` down to `min_ngram` tokens; for the first
/// length with an earlier occurrence, returns the tokens that followed the **most
/// recent** occurrence. Returns an empty draft when nothing matches.
pub fn lookup_draft(tokens: &[u32], min_ngram: usize, max_ngram: usize, max_draft: usize) -> Vec<u32> {
    let len = tokens.len();
    if max_draft == 0 || min_ngram == 0 {
        return Vec::new();
    }
    for n in (min_ngram..=max_ngram).rev() {
        if len <= n {
            continue;
        }
        let pattern = &tokens[len - n..];
        // Candidate starts end before the suffix itself; scan newest first.
        for start in (0..len - n).rev() {
            if &tokens[start..start + n] == pattern {
                let from = start + n;
                let to = (from + max_draft).min(len);
                return tokens[from..to].to_vec();
            }
        }
    }
    Vec::new()
}

/// Decayed per-sequence counts of draft acceptance, modelling each draft token as
/// accepted with probability `a` given that the previous one was.
#[derive(Debug, Clone, Default)]
pub struct DraftStats {
    accepted: f32,
    trials: f32,
}

impl DraftStats {
    const PRIOR_ACCEPTED: f32 = 1.0;
    const PRIOR_TRIALS: f32 = 2.0;
    const DECAY: f32 = 0.8;
    /// Slower fade for steps that skipped drafting: a probe after a bad stretch costs
    /// a verification pass, so it should come rarely.
    const IDLE_DECAY: f32 = 0.97;

    /// Estimated probability that the next draft token is accepted.
    pub fn acceptance(&self) -> f32 {
        (self.accepted + Self::PRIOR_ACCEPTED) / (self.trials + Self::PRIOR_TRIALS)
    }

    /// Records a verification of `drafted` tokens of which the first `accepted` matched.
    pub fn record(&mut self, drafted: usize, accepted: usize) {
        if drafted == 0 {
            return;
        }
        self.accepted = self.accepted * Self::DECAY + accepted as f32;
        self.trials = self.trials * Self::DECAY + accepted as f32 + f32::from(accepted < drafted);
    }

    /// Lets old evidence fade while the sequence is not drafting, so a sequence that
    /// stopped drafting after a bad stretch eventually tries again.
    pub fn decay(&mut self) {
        self.accepted *= Self::IDLE_DECAY;
        self.trials *= Self::IDLE_DECAY;
    }
}

/// Expected tokens a step yields with `d` drafts at per-token acceptance `a`:
/// the sampled token plus each draft prefix that survives, `1 + a + ... + a^d`.
pub fn expected_tokens(a: f32, d: usize) -> f32 {
    let mut sum = 0.0;
    let mut p = 1.0;
    for _ in 0..=d {
        sum += p;
        p *= a;
    }
    sum
}

/// Moving averages of forward-pass time for decode-only passes, by number of rows.
#[derive(Debug, Clone)]
pub struct PassCosts {
    /// `(average microseconds, pass index of last update)`, if measured.
    entries: Vec<Option<(f64, u64)>>,
    passes: u64,
}

impl PassCosts {
    /// Measurements older than this many passes are ignored, so a row count measured
    /// during a busy moment is retried eventually.
    const STALE_AFTER: u64 = 512;
    const ALPHA: f64 = 0.3;

    /// Tracks row counts `1..=max_rows`.
    pub fn new(max_rows: usize) -> Self {
        Self {
            entries: vec![None; max_rows + 1],
            passes: 0,
        }
    }

    /// Records a decode-only pass of `rows` rows that took `micros`.
    pub fn record(&mut self, rows: usize, micros: f64) {
        self.passes += 1;
        let passes = self.passes;
        let Some(e) = self.entries.get_mut(rows) else { return };
        let avg = match *e {
            Some((old, at)) if passes - at <= Self::STALE_AFTER => {
                (1.0 - Self::ALPHA) * old + Self::ALPHA * micros
            }
            _ => micros,
        };
        *e = Some((avg, passes));
    }

    fn known(&self, rows: usize) -> Option<f64> {
        let (us, at) = (*self.entries.get(rows)?)?;
        (self.passes - at <= Self::STALE_AFTER).then_some(us)
    }

    /// Cost estimate for `rows` rows: its own measurement, else a linear interpolation
    /// between the nearest measured sizes around it, else (past the largest measured
    /// size) optimistically the largest measured smaller size, so bigger drafts get
    /// tried once.
    pub fn estimate(&self, rows: usize) -> Option<f64> {
        if let Some(c) = self.known(rows) {
            return Some(c);
        }
        let n = self.entries.len();
        let lo = (1..rows.min(n)).rev().find_map(|r| self.known(r).map(|c| (r, c)));
        let hi = (rows + 1..n).find_map(|r| self.known(r).map(|c| (r, c)));
        match (lo, hi) {
            (Some((r0, c0)), Some((r1, c1))) => {
                Some(c0 + (c1 - c0) * (rows - r0) as f64 / (r1 - r0) as f64)
            }
            (Some((_, c0)), None) => Some(c0),
            (None, hi) => hi.map(|(_, c)| c),
        }
    }

    /// Picks how many of `max_draft` draft tokens to verify for one sequence.
    ///
    /// The step so far has `rows` rows expected to yield `tokens` tokens (this sequence
    /// included, at one token). Returns the `d` maximising
    /// `(tokens + E(d) - 1) / cost(rows + d)`, preferring fewer drafts on ties. Without
    /// a measurement for `rows` itself it returns 0, so the plain pass gets measured.
    pub fn choose_draft_len(&self, rows: usize, tokens: f32, acceptance: f32, max_draft: usize) -> usize {
        let Some(base) = self.known(rows) else { return 0 };
        let mut best = (tokens as f64 / base, 0);
        for d in 1..=max_draft {
            let Some(cost) = self.estimate(rows + d) else { break };
            let value = (tokens + expected_tokens(acceptance, d) - 1.0) as f64 / cost;
            if value > best.0 * 1.0001 {
                best = (value, d);
            }
        }
        best.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chooses_draft_length_from_measured_costs() {
        let mut c = PassCosts::new(32);
        // Nothing measured for one row: measure the plain pass first.
        assert_eq!(c.choose_draft_len(1, 1.0, 0.75, 7), 0);
        c.record(1, 1000.0);
        // Unmeasured larger sizes are assumed to cost the same: try the longest draft.
        assert_eq!(c.choose_draft_len(1, 1.0, 0.75, 7), 7);
        // Metal-like: 2-8 rows cost 1.3x, 9-16 rows 1.8x.
        for r in 2..=8 {
            c.record(r, 1300.0);
        }
        for r in 9..=16 {
            c.record(r, 1800.0);
        }
        assert_eq!(c.choose_draft_len(1, 1.0, 0.75, 15), 7);
        // Low acceptance: drafting doesn't pay.
        assert_eq!(c.choose_draft_len(1, 1.0, 0.2, 15), 0);
        // CPU K-quant-like: 8 rows cost 3x one row.
        let mut c = PassCosts::new(32);
        c.record(1, 1000.0);
        for r in 2..=8 {
            c.record(r, 1000.0 + 290.0 * (r - 1) as f64);
        }
        let d = c.choose_draft_len(1, 1.0, 0.6, 7);
        assert!((1..=3).contains(&d), "{d}");
        assert_eq!(c.choose_draft_len(1, 1.0, 0.25, 7), 0);
        // One expensive large pass: sizes in between are interpolated, not re-explored.
        let mut c = PassCosts::new(32);
        c.record(1, 1000.0);
        c.record(16, 5000.0);
        assert!((c.estimate(6).unwrap() - 2333.3).abs() < 1.0);
        assert!(c.choose_draft_len(1, 1.0, 0.6, 15) <= 2);
    }

    #[test]
    fn acceptance_tracks_recent_verifications() {
        let mut s = DraftStats::default();
        assert!((s.acceptance() - 0.5).abs() < 1e-6);
        for _ in 0..10 {
            s.record(7, 0);
        }
        assert!(s.acceptance() < 0.25, "{}", s.acceptance());
        for _ in 0..10 {
            s.decay();
        }
        assert!(s.acceptance() < 0.3, "{}", s.acceptance());
        for _ in 0..100 {
            s.decay();
        }
        assert!(s.acceptance() > 0.45);
        for _ in 0..10 {
            s.record(7, 7);
        }
        assert!(s.acceptance() > 0.95);
        assert!((expected_tokens(0.5, 2) - 1.75).abs() < 1e-6);
    }

    #[test]
    fn finds_most_recent_longest_match() {
        // Suffix [1, 2] appears at 0 (followed by 3, 4) and at 4 (followed by 7).
        let t = [1, 2, 3, 4, 1, 2, 7, 9, 1, 2];
        assert_eq!(lookup_draft(&t, 1, 3, 4), vec![7, 9, 1, 2]);
        // A longer suffix wins over a more recent short one.
        let t = [5, 6, 7, 8, 9, 7, 0, 5, 6, 7];
        assert_eq!(lookup_draft(&t, 1, 3, 2), vec![8, 9]);
    }

    #[test]
    fn falls_back_to_shorter_ngrams_and_stops_at_the_end() {
        let t = [4, 1, 2, 9, 3, 2];
        assert_eq!(lookup_draft(&t, 1, 3, 8), vec![9, 3, 2]);
        assert!(lookup_draft(&t, 2, 3, 8).is_empty());
        assert!(lookup_draft(&[1, 2, 3], 1, 3, 4).is_empty());
        assert!(lookup_draft(&[1, 1], 1, 3, 4) == vec![1]);
        assert!(lookup_draft(&[1, 1, 1], 1, 3, 0).is_empty());
    }
}
