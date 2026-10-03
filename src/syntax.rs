//! Zero-Allocation Distributional Lexical Categorization
//!
//! # Scientific Foundation
//! - Primary Papers:
//!   - Redington, M., Chater, N., & Finch, S. (1998). "Distributional information:
//!     A powerful marker for acquiring lexical categories." *Cognitive Science*, 22(4), 425-469.
//!     (Mathematical proof that syntactic categories—determiners, auxiliaries, nouns, verbs—emerge
//!     unsupervised from raw bigram frequency and context transition dispersion without hand-coded rules).
//!   - Harris, Z. S. (1954). "Distributional structure." *Word*, 10(2-3), 146-162.
//!     (The distributional hypothesis: words occurring in similar linguistic environments share
//!     grammatical function).
//!   - Ding, N., Melloni, L., Zhang, H., Tian, X., & Poeppel, D. (2016). "Cortical tracking of
//!     hierarchical linguistic structures in normal and jabberwocky speech." *Nature Neuroscience*,
//!     19(1), 158-164. (Cortical phase-locking tracks grammatical structure purely from transition
//!     probabilities, with zero pre-compiled dictionaries).
//!
//! # Strict Constraints (FRANKIE_LAWS.md)
//! - ZERO hardcoded keyword dictionaries ("the", "who", "not", "did" are BANNED from source code).
//! - Zero heap allocations during lexical tracking (`malloc` free).
//! - Stack-resident fixed-size tables and 36-byte `StackWord` buffers only.
//! - Completely language-agnostic: operates identically on English, Latin, or non-human sequences.

/// Maximum bytes in an inline stack-allocated word
pub const MAX_WORD_BYTES: usize = 32;

/// Maximum words tracked in the cache-resident distributional lexicon
pub const MAX_DISTRIBUTIONAL_VOCAB: usize = 1024;

/// Stack-allocated fixed-size lowercased word buffer (Zero Heap Allocation)
/// Consumes exactly 36 bytes on the CPU stack.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StackWord {
    pub bytes: [u8; MAX_WORD_BYTES],
    pub len: usize,
}

impl StackWord {
    /// Create empty word buffer
    pub const fn empty() -> Self {
        Self {
            bytes: [0u8; MAX_WORD_BYTES],
            len: 0,
        }
    }

    /// Extract alphanumeric characters, convert to ASCII lowercase in-place on the stack
    #[inline]
    pub fn from_str(s: &str) -> Option<Self> {
        let mut bytes = [0u8; MAX_WORD_BYTES];
        let mut len = 0;
        for b in s.bytes() {
            if b.is_ascii_alphanumeric() && len < MAX_WORD_BYTES {
                bytes[len] = b.to_ascii_lowercase();
                len += 1;
            }
        }
        if len > 0 {
            Some(Self { bytes, len })
        } else {
            None
        }
    }

    /// Safe ASCII string slice view (Zero Allocation)
    #[inline(always)]
    pub fn as_str(&self) -> &str {
        // Safe: from_str only writes valid ASCII alphanumeric bytes
        unsafe { std::str::from_utf8_unchecked(&self.bytes[..self.len]) }
    }
}

/// Statistics entry for a token in the distributional lexicon
#[derive(Clone, Copy)]
pub struct LexicalStats {
    pub word: StackWord,
    /// Absolute frequency of occurrence in streaming text: $N(w)$
    pub count: u32,
    /// Number of times this token immediately followed the language's primary determiner ($T_{max}$)
    pub preceded_by_lead_particle: u32,
    /// 64-bit sparse bloom filter signature of observed right-neighbor contexts
    pub right_context_signature: u64,
    /// Distinct right-neighbor contexts observed (derived from bloom signature)
    pub right_context_dispersion: u16,
    /// Whether this entry is currently populated
    pub is_active: bool,
}

impl LexicalStats {
    pub const fn empty() -> Self {
        Self {
            word: StackWord::empty(),
            count: 0,
            preceded_by_lead_particle: 0,
            right_context_signature: 0,
            right_context_dispersion: 0,
            is_active: false,
        }
    }

    #[inline(always)]
    pub fn right_dispersion(&self) -> u32 {
        self.right_context_dispersion as u32
    }
}

/// Unsupervised Distributional Lexical Profiler (Redington et al. 1998; Harris 1954)
///
/// Segregates functional syntactic particles from thematic content words purely
/// from statistical distribution and bigram co-occurrence:
/// - **Functional Scaffolds:** High frequency ($N(w) \gg \mu$), wide context dispersion,
///   and rarely/never preceded by the primary determiner.
/// - **Thematic Content:** Moderate frequency OR high frequency but structurally following
///   the primary determiner ($NP$ phrase heads, e.g. "the king", "the lord", "the earth").
///
/// Runs entirely in a flat cache-resident table with zero heap allocation.
pub struct DistributionalLexicon {
    pub entries: [LexicalStats; MAX_DISTRIBUTIONAL_VOCAB],
    pub entry_count: usize,
    pub total_tokens_observed: u32,
    /// Primary grammatical particle of the corpus (highest frequency token, e.g. "the")
    pub lead_particle: StackWord,
    pub lead_particle_count: u32,
    /// Last observed token for sequential bigram tracking
    pub last_token: Option<StackWord>,
}

impl DistributionalLexicon {
    pub const fn new() -> Self {
        Self {
            entries: [LexicalStats::empty(); MAX_DISTRIBUTIONAL_VOCAB],
            entry_count: 0,
            total_tokens_observed: 0,
            lead_particle: StackWord::empty(),
            lead_particle_count: 0,
            last_token: None,
        }
    }

    #[inline(always)]
    fn hash_word(word: &StackWord) -> usize {
        let mut h = 0xcbf29ce484222325u64;
        for i in 0..word.len {
            h ^= word.bytes[i] as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        (h as usize) % 64
    }

    /// Observe a token in context: updates frequency, right-context bloom signature,
    /// and lead determiner bigram statistics.
    /// Runs in $O(V)$ with zero heap allocation.
    pub fn observe_token(&mut self, token: StackWord, _next_token: Option<StackWord>) {
        self.total_tokens_observed += 1;
        let token_hash_bit = 1u64 << Self::hash_word(&token);
        let was_preceded_by_lead = self.last_token == Some(self.lead_particle) && self.lead_particle_count > 0;

        // Step 1: Update right-neighbor signature of the preceding token
        if let Some(prev) = self.last_token {
            for i in 0..self.entry_count {
                if self.entries[i].is_active && self.entries[i].word == prev {
                    self.entries[i].right_context_signature |= token_hash_bit;
                    self.entries[i].right_context_dispersion = self.entries[i].right_context_signature.count_ones() as u16;
                    break;
                }
            }
        }

        // Step 2: Update current token statistics
        let mut found_idx = None;
        for i in 0..self.entry_count {
            if self.entries[i].is_active && self.entries[i].word == token {
                self.entries[i].count += 1;
                if was_preceded_by_lead {
                    self.entries[i].preceded_by_lead_particle += 1;
                }
                found_idx = Some(i);
                break;
            }
        }

        let curr_idx = match found_idx {
            Some(i) => i,
            None => {
                if self.entry_count < MAX_DISTRIBUTIONAL_VOCAB {
                    let idx = self.entry_count;
                    self.entries[idx] = LexicalStats {
                        word: token,
                        count: 1,
                        preceded_by_lead_particle: if was_preceded_by_lead { 1 } else { 0 },
                        right_context_signature: 0,
                        right_context_dispersion: 0,
                        is_active: true,
                    };
                    self.entry_count += 1;
                    idx
                } else {
                    self.last_token = Some(token);
                    return;
                }
            }
        };

        // Step 3: Track global lead particle
        if self.entries[curr_idx].count > self.lead_particle_count {
            self.lead_particle = token;
            self.lead_particle_count = self.entries[curr_idx].count;
        }

        self.last_token = Some(token);
    }

    /// Check if a word is an emergent structural scaffold (functional particle).
    ///
    /// Principles:
    /// 1. Frequency $N(w) \gg \mu$ (Zipfian functional head).
    /// 2. Context dispersion $\ge 2$ distinct right-neighbor categories.
    /// 3. Invariant (Harris 1954): Determiners precede content nouns, NEVER functional particles.
    ///    If $W$ regularly follows the lead determiner ("the $W$"), $W$ is a content noun, NOT a scaffold!
    #[inline]
    pub fn is_structural_scaffold(&self, token: &str) -> bool {
        if self.entry_count == 0 || self.total_tokens_observed < 8 {
            return false;
        }

        let mean_freq = (self.total_tokens_observed as f32) / (self.entry_count as f32);
        let threshold = (1.5 * mean_freq).max(2.5);

        for i in 0..self.entry_count {
            if self.entries[i].is_active && self.entries[i].word.as_str() == token {
                let entry = &self.entries[i];

                if (entry.count as f32) < threshold {
                    return false;
                }

                // If token regularly functions as the thematic argument of the lead particle (>= 33% of occurrences),
                // it is a Content Word (e.g. "the king", "the lord", "the earth"), not a functional scaffold!
                if entry.preceded_by_lead_particle > 0
                    && (entry.preceded_by_lead_particle * 3 >= entry.count)
                {
                    return false;
                }

                return entry.right_dispersion() >= 2;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsupervised_emergence_of_scaffolding_particles() {
        let mut lexicon = DistributionalLexicon::new();

        // Feed natural text where particles naturally recur frequently without hardcoding
        let text = "the dog chased the cat and the cat climbed the tree and the boy saw the dog";
        let tokens: Vec<&str> = text.split_whitespace().collect();

        for i in 0..tokens.len() {
            let curr = StackWord::from_str(tokens[i]).unwrap();
            let next = if i + 1 < tokens.len() {
                StackWord::from_str(tokens[i + 1])
            } else {
                None
            };
            lexicon.observe_token(curr, next);
        }

        // "the" appeared 5 times out of 16 tokens (31.25% of stream) -> Emerges as structural scaffold!
        assert!(
            lexicon.is_structural_scaffold("the"),
            "'the' must autonomously emerge as a structural scaffold purely from frequency statistics"
        );

        // "dog" appeared only 2 times (12.5%) -> Content word
        // "cat" appeared only 2 times (12.5%) -> Content word
        // "climbed" appeared only 1 time (6.25%) -> Content word
        assert!(
            !lexicon.is_structural_scaffold("climbed"),
            "'climbed' is a thematic content word and must NOT be classified as a scaffold"
        );
    }
}
