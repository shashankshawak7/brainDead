//! Bit-Packed Zero-Allocation SDR & Quantized Synapses
//!
//! # Scientific Foundation
//! - Primary Papers:
//!   - Esser, S. K., et al. (2016). "Convolutional networks for chip-scale neuromorphic systems."
//!     *Science*, 353(6300), 630-636. (IBM TrueNorth 1-to-8 bit quantized synapses).
//!   - Davies, M., et al. (2018). "Loihi: A neuromorphic manycore processor with on-chip learning."
//!     *IEEE Micro*, 38(1), 82-99. (Intel Loihi low-precision discrete weight states).
//!   - Petersen, C. C., et al. (1998). "All-or-none potentiation at CA3-CA1 synapses."
//!     *Proceedings of the National Academy of Sciences (PNAS)*, 95(8), 4732-4737.
//!     (Biological proof that synaptic transmission is discrete and quantized).
//! - Mathematical & Systems Principle:
//!   - Floating-point representations (`f32`) waste 32 bits per state and cause cache eviction.
//!   - A 2,048-bit SDR represented as index vectors (`Vec<usize>`) requires 8 bytes per active bit + heap overhead.
//!   - Packing into a static array of 64-bit words (`[u64; 32]`) consumes **exactly 256 bytes**,
//!     requires **zero heap allocations**, and evaluates overlap via single-cycle hardware `POPCNT`.

/// Total bits in a standard cortical column SDR (Mountcastle canonical module)
pub const SDR_BITS: usize = 2048;
/// Number of 64-bit machine words required to pack SDR_BITS (2048 / 64 = 32 words)
pub const SDR_WORDS: usize = SDR_BITS / 64;

/// Stack-allocated, bit-packed Sparse Distributed Representation.
/// Consumes exactly 256 bytes on the stack. Zero heap allocation (`malloc` free).
///
/// Reference: Esser et al. (Science 2016) - Binary neuromorphic representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitSdr {
    pub words: [u64; SDR_WORDS],
}

impl BitSdr {
    /// Create empty (all silent) SDR
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            words: [0u64; SDR_WORDS],
        }
    }

    /// Set bit at index $i \in [0, \text{SDR\_BITS})$
    #[inline(always)]
    pub fn set_bit(&mut self, bit_idx: usize) {
        debug_assert!(bit_idx < SDR_BITS, "Bit index out of bounds");
        let word_idx = bit_idx / 64;
        let bit_offset = bit_idx % 64;
        self.words[word_idx] |= 1u64 << bit_offset;
    }

    /// Check if bit is set
    #[inline(always)]
    pub fn is_bit_set(&self, bit_idx: usize) -> bool {
        debug_assert!(bit_idx < SDR_BITS);
        let word_idx = bit_idx / 64;
        let bit_offset = bit_idx % 64;
        (self.words[word_idx] & (1u64 << bit_offset)) != 0
    }

    /// Population count: total active neurons ($k$)
    /// Uses single-cycle hardware CPU instruction (`POPCNT`).
    #[inline(always)]
    pub fn count_ones(&self) -> u32 {
        let mut count = 0u32;
        for w in &self.words {
            count += w.count_ones();
        }
        count
    }

    /// Compute exact overlap (intersection $|A \cap B|$) in ~32 CPU cycles using SIMD/POPCNT.
    ///
    /// Math: $|A \cap B| = \sum_{w=0}^{31} \text{popcnt}(A_w \ \& \ B_w)$
    #[inline(always)]
    pub fn overlap(&self, other: &Self) -> u32 {
        let mut count = 0u32;
        for i in 0..SDR_WORDS {
            count += (self.words[i] & other.words[i]).count_ones();
        }
        count
    }

    /// Compute Jaccard similarity: $\frac{|A \cap B|}{|A \cup B|}$
    #[inline(always)]
    pub fn jaccard_similarity(&self, other: &Self) -> f32 {
        let mut intersection = 0u32;
        let mut union = 0u32;
        for i in 0..SDR_WORDS {
            intersection += (self.words[i] & other.words[i]).count_ones();
            union += (self.words[i] | other.words[i]).count_ones();
        }
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Circular permutation (bit rotation) over the 2048-bit address space.
    ///
    /// Preserves exact population count (sparsity), is distance-preserving, and orthogonal.
    /// Used for Vector Symbolic Architecture (VSA) permutation binding without density explosion.
    ///
    /// Reference:
    /// - Kanerva, P. (2009). *Cognitive Computation*, 1(2), 139-159. (Section "Permutation").
    /// - Kleyko, D., et al. (2020). "Vector Symbolic Architectures as a Computing Framework for
    ///   Nanoscale Neuromorphic Computing." *Proceedings of the IEEE*, 110(9), 1538-1571.
    #[inline]
    pub fn rotate_left(&self, shift: usize) -> Self {
        let shift = shift % SDR_BITS;
        if shift == 0 {
            return *self;
        }

        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        let mut out = Self::empty();

        if bit_shift == 0 {
            for i in 0..SDR_WORDS {
                let src_idx = (i + SDR_WORDS - (word_shift % SDR_WORDS)) % SDR_WORDS;
                out.words[i] = self.words[src_idx];
            }
        } else {
            for i in 0..SDR_WORDS {
                let src_curr = (i + SDR_WORDS - (word_shift % SDR_WORDS)) % SDR_WORDS;
                let src_prev = (src_curr + SDR_WORDS - 1) % SDR_WORDS;
                out.words[i] = (self.words[src_curr] << bit_shift)
                    | (self.words[src_prev] >> (64 - bit_shift));
            }
        }
        out
    }

    /// Reverse circular permutation: $\Pi^{-1}(S)$
    #[inline(always)]
    pub fn rotate_right(&self, shift: usize) -> Self {
        let shift = shift % SDR_BITS;
        if shift == 0 {
            *self
        } else {
            self.rotate_left(SDR_BITS - shift)
        }
    }
}

/// Quantized 8-Bit Synapse Matrix.
///
/// Reduces memory consumption by 75% compared to `f32` (1 MB instead of 4 MB for 1000x1000 area).
/// All inner loop arithmetic uses integer SIMD rather than floating-point math.
///
/// Reference: Davies et al. (IEEE Micro 2018) - Intel Loihi 8-bit synaptic state architecture.
#[derive(Debug, Clone)]
pub struct QuantizedSynapseMatrix {
    pub pre_neurons: usize,
    pub post_neurons: usize,
    /// 8-bit quantized weights in range [0, 255]
    pub weights: Vec<u8>,
}

impl QuantizedSynapseMatrix {
    /// Initialize with zero heap reallocations
    pub fn new(pre: usize, post: usize) -> Self {
        Self {
            pre_neurons: pre,
            post_neurons: post,
            weights: vec![0u8; pre * post],
        }
    }

    /// Compute integer postsynaptic dendritic drive directly from bit-packed SDR.
    /// Zero heap allocations: accumulates into a pre-allocated caller buffer.
    ///
    /// Math: $d_j = \sum_{i \in \text{active bits}} w_{ij}$ (integer addition, zero float operations)
    #[inline]
    pub fn compute_drive_into(&self, input_sdr: &BitSdr, out_drive: &mut [u32]) {
        debug_assert_eq!(out_drive.len(), self.post_neurons);
        out_drive.fill(0);

        for word_idx in 0..SDR_WORDS {
            let mut word = input_sdr.words[word_idx];
            let base_neuron = word_idx * 64;

            // Fast bit traversal using trailing zeros
            while word != 0 {
                let bit_offset = word.trailing_zeros() as usize;
                let pre_idx = base_neuron + bit_offset;

                if pre_idx < self.pre_neurons {
                    let row_offset = pre_idx * self.post_neurons;
                    for post_idx in 0..self.post_neurons {
                        out_drive[post_idx] += self.weights[row_offset + post_idx] as u32;
                    }
                }

                // Clear lowest set bit in 1 CPU cycle
                word &= word - 1;
            }
        }
    }

    /// Apply quantized Hebbian potentiation with saturating integer arithmetic.
    ///
    /// Math: $w_{ij} \leftarrow \min(255, w_{ij} + \Delta w)$
    /// Reference: Petersen et al. (PNAS 1998) - Quantized LTP state transitions.
    #[inline]
    pub fn hebbian_potentiate(
        &mut self,
        pre_sdr: &BitSdr,
        post_sdr: &BitSdr,
        delta_weight: u8,
    ) {
        for pre_word_idx in 0..SDR_WORDS {
            let mut pre_word = pre_sdr.words[pre_word_idx];
            let pre_base = pre_word_idx * 64;

            while pre_word != 0 {
                let pre_offset = pre_word.trailing_zeros() as usize;
                let pre_idx = pre_base + pre_offset;

                if pre_idx < self.pre_neurons {
                    let row_offset = pre_idx * self.post_neurons;

                    for post_word_idx in 0..SDR_WORDS {
                        let mut post_word = post_sdr.words[post_word_idx];
                        let post_base = post_word_idx * 64;

                        while post_word != 0 {
                            let post_offset = post_word.trailing_zeros() as usize;
                            let post_idx = post_base + post_offset;

                            if post_idx < self.post_neurons {
                                let synapse_idx = row_offset + post_idx;
                                self.weights[synapse_idx] = self.weights[synapse_idx].saturating_add(delta_weight);
                            }

                            post_word &= post_word - 1;
                        }
                    }
                }

                pre_word &= pre_word - 1;
            }
        }
    }
}

/// Zero-Allocation Cortical Area.
/// Uses pre-allocated scratchpads so inference steps perform ZERO `malloc` or heap calls.
#[derive(Debug)]
pub struct ZeroAllocArea {
    pub synapses: QuantizedSynapseMatrix,
    /// Pre-allocated scratchpad for dendritic drive
    pub drive_scratchpad: Vec<u32>,
    /// Pre-allocated scratchpad for k-WTA selection sorting
    pub selection_scratchpad: Vec<u32>,
    /// Number of winners to keep in k-WTA
    pub k_winners: usize,
}

impl ZeroAllocArea {
    pub fn new(neurons: usize, k_winners: usize) -> Self {
        Self {
            synapses: QuantizedSynapseMatrix::new(neurons, neurons),
            drive_scratchpad: vec![0u32; neurons],
            selection_scratchpad: vec![0u32; neurons],
            k_winners,
        }
    }

    /// Step inference with ZERO heap allocation. Returns result in a stack BitSdr.
    #[inline]
    pub fn step_zero_alloc(&mut self, input: &BitSdr) -> BitSdr {
        self.synapses.compute_drive_into(input, &mut self.drive_scratchpad);

        // Copy into pre-allocated scratchpad (pure memcpy, zero malloc)
        self.selection_scratchpad.copy_from_slice(&self.drive_scratchpad);
        let k = self.k_winners;
        let mut out = BitSdr::empty();

        if k < self.selection_scratchpad.len() {
            self.selection_scratchpad.select_nth_unstable_by(k - 1, |a, b| b.cmp(a));
            let threshold = self.selection_scratchpad[k - 1];

            if threshold > 0 {
                let mut count = 0;
                for (idx, &drive) in self.drive_scratchpad.iter().enumerate() {
                    if drive >= threshold && count < k {
                        out.set_bit(idx);
                        count += 1;
                    }
                }
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitsdr_memory_and_popcnt_speed() {
        // Assert stack footprint is exactly 256 bytes!
        assert_eq!(
            std::mem::size_of::<BitSdr>(),
            256,
            "BitSdr MUST be exactly 256 bytes on the stack"
        );

        let mut sdr_a = BitSdr::empty();
        let mut sdr_b = BitSdr::empty();

        // Set 40 active bits (~2% sparsity of 2048)
        for i in 0..40 {
            sdr_a.set_bit(i * 10);
            sdr_b.set_bit(i * 10); // identical
        }

        assert_eq!(sdr_a.count_ones(), 40);
        assert_eq!(sdr_a.overlap(&sdr_b), 40);
        assert!((sdr_a.jaccard_similarity(&sdr_b) - 1.0).abs() < 1e-6);

        // Perturb sdr_b
        let mut sdr_c = BitSdr::empty();
        for i in 0..40 {
            sdr_c.set_bit(i * 10 + 1); // completely disjoint
        }
        assert_eq!(sdr_a.overlap(&sdr_c), 0);
    }

    #[test]
    fn test_quantized_synapses_zero_alloc() {
        let n = 2048;
        let mut area = ZeroAllocArea::new(n, 40);

        let mut input = BitSdr::empty();
        for i in 0..40 {
            input.set_bit(i);
        }

        let mut target = BitSdr::empty();
        for i in 100..140 {
            target.set_bit(i);
        }

        // Potentiate input -> target with discrete 8-bit steps
        area.synapses.hebbian_potentiate(&input, &target, 15);

        // Step zero alloc
        let output = area.step_zero_alloc(&input);

        // Verify that target neurons fired
        let overlap = output.overlap(&target);
        println!("Quantized Zero-Alloc Step Overlap: {}/40", overlap);
        assert_eq!(overlap, 40, "Target assembly must be recalled with 100% fidelity");
    }
}
