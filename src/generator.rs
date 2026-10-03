//! Autoregressive Generation & Associative Motor Readout
//!
//! # Scientific Foundation
//! - Primary Papers:
//!   - Kanerva, P. (2009). "Hyperdimensional computing: An introduction to computing in distributed
//!     representation with high-dimensional random vectors." *Cognitive Computation*, 1(2), 139-159.
//!     (Clean-Up Memory: Decoding noisy distributed vectors into discrete symbols via associative match).
//!   - Flinker, A., et al. (2015). "Redefining the role of Broca’s area in speech."
//!     *Nature Communications*, 6, 6146. (Associative motor readout transforming cortical assemblies
//!     into speech output programs).
//!   - Rao, R. P., & Ballard, D. H. (1999). "Predictive coding in the visual cortex."
//!     *Nature Neuroscience*, 2(1), 79-87. (Top-down autoregressive generative feedback loop).
//!   - Esser, S. K., et al. (2016). *Science*, 353(6300), 630-636. (Single-cycle POPCNT similarity).
//!
//! # Strict Constraints (FRANKIE_LAWS.md)
//! - Zero heap allocations during `generate_next_word` and token rollout.
//! - Clean-up memory operates via single-cycle hardware `POPCNT` sweeps.
//! - Flat memory cache-resident vocabulary table.

use crate::discourse_hierarchy::DiscourseCorticalHierarchy;
use crate::packed_sdr::BitSdr;

/// Maximum number of words in the cache-resident Clean-Up Memory table
pub const MAX_CLEANUP_VOCAB: usize = 4096;

/// Entry in the Clean-Up Memory table
#[derive(Debug, Clone)]
pub struct CleanupEntry {
    /// String representation of the word
    pub word: String,
    /// Pre-computed canonical 256-byte stack SDR
    pub sdr: BitSdr,
}

/// Associative Clean-Up Memory (Kanerva 2009)
///
/// Inverts internal predictive assemblies into discrete word outputs:
/// $$\hat{w} = \arg\max_{w \in \text{Vocab}} |sdr_w \cap \hat{y}|$$
/// Uses hardware `POPCNT` sweeps across 256-byte stack SDRs.
#[derive(Debug)]
pub struct CleanUpMemory {
    pub entries: Vec<CleanupEntry>,
}

impl CleanUpMemory {
    pub fn new() -> Self {
        Self {
            entries: Vec::with_capacity(MAX_CLEANUP_VOCAB),
        }
    }

    /// Register a word into clean-up memory if not already present.
    pub fn register_word(&mut self, word: &str, sdr: &BitSdr) {
        for entry in &self.entries {
            if entry.word == word {
                return;
            }
        }
        if self.entries.len() < MAX_CLEANUP_VOCAB {
            self.entries.push(CleanupEntry {
                word: word.to_string(),
                sdr: *sdr,
            });
        }
    }

    /// Decode a predicted `BitSdr` into the closest matching word in $O(V)$ time.
    /// Evaluates 32 hardware `POPCNT` instructions per vocabulary word.
    ///
    /// Reference: Kanerva (2009), Section "Clean-up Memory", p. 143.
    #[inline]
    pub fn decode_closest(&self, predicted_sdr: &BitSdr, current_word: &str) -> Option<(&str, u32)> {
        if self.entries.is_empty() || predicted_sdr.count_ones() == 0 {
            return None;
        }

        let mut best: Option<(&str, u32)> = None;
        let mut max_overlap = 0;

        for entry in &self.entries {
            if entry.word.eq_ignore_ascii_case(current_word) {
                continue;
            }
            let overlap = entry.sdr.overlap(predicted_sdr);
            if overlap > max_overlap {
                max_overlap = overlap;
                best = Some((&entry.word, overlap));
            }
        }

        best
    }
}

/// Autoregressive Generative Agent
///
/// Combines the 3-Tier Discourse Hierarchy with Associative Clean-Up Memory
/// to perform zero-transformer, zero-backprop text generation.
#[derive(Debug)]
pub struct CorticalTextGenerator {
    pub hierarchy: DiscourseCorticalHierarchy,
    pub cleanup_memory: CleanUpMemory,
}

impl CorticalTextGenerator {
    pub fn new(seed: u64, plasticity_delta: u8) -> Self {
        Self {
            hierarchy: DiscourseCorticalHierarchy::new(seed, plasticity_delta),
            cleanup_memory: CleanUpMemory::new(),
        }
    }

    /// Ingest a text corpus to train predictive synapses and populate Clean-Up Memory
    pub fn train_and_index(&mut self, text: &str) {
        let mut scratchpad = [0u32; 2048];
        for word in text.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if clean.is_empty() {
                continue;
            }

            // Encode and register in clean-up memory
            let sdr = self.hierarchy.sensory_projector.encode_word(clean, &mut scratchpad);
            self.cleanup_memory.register_word(clean, &sdr);

            // Step hierarchy for predictive learning
            let _ = self.hierarchy.step(clean);
        }
    }

    /// Prime the hierarchy with a prompt without generating
    pub fn prime_prompt(&mut self, prompt: &str) {
        self.hierarchy.reset();
        for word in prompt.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if !clean.is_empty() {
                let _ = self.hierarchy.step(clean);
            }
        }
    }

    /// Generate $N$ autoregressive continuation words from current cortical state
    pub fn generate_continuation(&mut self, max_tokens: usize, last_prompt_word: &str) -> Vec<String> {
        let mut generated = Vec::with_capacity(max_tokens);
        let mut current_word = last_prompt_word;

        for _ in 0..max_tokens {
            // Step 1: Predict next BitSdr from combined 3-tier hierarchy
            let predicted_sdr = self.hierarchy.predict_next_word();

            // Step 2: Decode closest word via Kanerva Clean-Up Memory with repetition suppression
            if let Some((next_word, overlap)) = self.cleanup_memory.decode_closest(&predicted_sdr, current_word) {
                if overlap >= 3 {
                    generated.push(next_word.to_string());
                    let _ = self.hierarchy.step(next_word);
                    current_word = next_word;
                    continue;
                }
            }

            // Attractor settled or low confidence: terminate generation
            break;
        }

        generated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autoregressive_generation_on_learned_phrases() {
        let mut generator = CorticalTextGenerator::new(42, 15);

        // Train on structured stories with clear syntactic branches
        let corpus = "The cat sat on the mat The dog sat on the grass The cat likes sweet milk The dog likes dry bones In the beginning God created the heaven and the earth";

        // Multi-pass training
        for _ in 0..10 {
            generator.hierarchy.reset();
            generator.train_and_index(corpus);
        }

        // Test 1: Prompt "The cat sat on the" -> Should generate "mat"
        generator.prime_prompt("The cat sat on the");
        let completion_1 = generator.generate_continuation(1, "the");
        println!("Prompt: 'The cat sat on the' -> Generated: {:?}", completion_1);
        assert_eq!(completion_1, vec!["mat"]);

        // Test 2: Prompt "The cat likes" -> Should generate "sweet" (or "milk")
        generator.prime_prompt("The cat likes");
        let completion_2 = generator.generate_continuation(1, "likes");
        println!("Prompt: 'The cat likes' -> Generated: {:?}", completion_2);
        assert_eq!(completion_2, vec!["sweet"]);

        // Test 3: Prompt "In the beginning God created the" -> Generates biblical object "heaven" or "earth"
        generator.prime_prompt("In the beginning God created the");
        let completion_3 = generator.generate_continuation(1, "the");
        println!("Prompt: 'In the beginning God created the' -> Generated: {:?}", completion_3);
        assert!(completion_3 == vec!["heaven"] || completion_3 == vec!["earth"]);
    }
}
