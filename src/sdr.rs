//! Pillar 1: Semantic Sparse Distributed Representations (SDR)
//!
//! # Scientific Foundation
//! - Paper: Dasgupta, S., Stevens, C. F., & Navlakha, S. (2017).
//!   "A neural algorithm for a fundamental computing problem." Science, 358(6364), 793-796.
//! - Biological Mechanism: Olfactory bulb / Mushroom body Kenyon cells.
//!   Dense sensory inputs (Projection Neurons) project via random sparse synapses to a large
//!   layer of Kenyon cells ($m \gg d$), followed by global feedforward inhibition (APL neuron)
//!   producing strict k-Winners-Take-All (k-WTA) sparsity (~2-5%).
//! - Mathematical Guarantee: Provable Locality-Sensitive Hashing (LSH) that preserves
//!   cosine/angular similarity without backpropagation or learned projection matrices.

/// Deterministic, zero-dependency pseudo-random number generator (XorShift64)
/// Used to generate biologically static random synaptic wiring (Kenyon cell connections).
#[derive(Debug, Clone)]
pub struct DeterministicPrng {
    state: u64,
}

impl DeterministicPrng {
    #[inline]
    pub const fn new(seed: u64) -> Self {
        // Non-zero seed required for xorshift
        Self { state: if seed == 0 { 0x853c49e6748fea9b } else { seed } }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    pub fn next_bounded(&mut self, bound: usize) -> usize {
        (self.next_u64() % (bound as u64)) as usize
    }
}

/// Binary Sparse Distributed Representation (SDR).
/// Invariant: Exactly `active_bits.len()` indices are active, sorted in ascending order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sdr {
    /// Total dimensionality of the cortical area ($m$)
    pub total_bits: usize,
    /// Indices of active neurons (columns)
    pub active_indices: Vec<usize>,
}

impl Sdr {
    /// Overlap score (dot product) between two binary SDRs.
    /// Since indices are sorted, computed in O(|A| + |B|) time via linear merge.
    pub fn overlap(&self, other: &Self) -> usize {
        debug_assert_eq!(self.total_bits, other.total_bits, "Dimension mismatch in SDR overlap");
        let mut i = 0;
        let mut j = 0;
        let mut count = 0;

        while i < self.active_indices.len() && j < other.active_indices.len() {
            let a = self.active_indices[i];
            let b = other.active_indices[j];
            if a == b {
                count += 1;
                i += 1;
                j += 1;
            } else if a < b {
                i += 1;
            } else {
                j += 1;
            }
        }
        count
    }

    /// Jaccard similarity: |A ∩ B| / |A ∪ B|
    pub fn jaccard_similarity(&self, other: &Self) -> f32 {
        let intersection = self.overlap(other);
        let union = self.active_indices.len() + other.active_indices.len() - intersection;
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

/// Fly-Algorithm Sparse Random Projector (Dasgupta et al., 2017)
/// Maps dense input vectors to sparse binary SDRs.
#[derive(Debug, Clone)]
pub struct FlyProjector {
    /// Input dimension (e.g. 64 or 128 context dimensions)
    pub input_dim: usize,
    /// Expanded cortical dimension (m, e.g. 2048 or 10000)
    pub expanded_dim: usize,
    /// Number of active connections per expanded neuron (c, typically ~6 in fruit fly)
    pub connections_per_neuron: usize,
    /// Number of winners to keep in k-WTA (k, typically 2% to 5% of expanded_dim)
    pub k_winners: usize,
    /// Static sparse synaptic connectivity table: [expanded_dim][connections_per_neuron]
    pub connections: Vec<Vec<usize>>,
}

impl FlyProjector {
    /// Initialize with deterministic random wiring
    pub fn new(
        input_dim: usize,
        expanded_dim: usize,
        connections_per_neuron: usize,
        k_winners: usize,
        seed: u64,
    ) -> Self {
        assert!(connections_per_neuron <= input_dim, "Connections cannot exceed input dimension");
        assert!(k_winners <= expanded_dim, "Winners cannot exceed expanded dimension");

        let mut rng = DeterministicPrng::new(seed);
        let mut connections = Vec::with_capacity(expanded_dim);

        for _ in 0..expanded_dim {
            let mut neuron_inputs = Vec::with_capacity(connections_per_neuron);
            while neuron_inputs.len() < connections_per_neuron {
                let idx = rng.next_bounded(input_dim);
                if !neuron_inputs.contains(&idx) {
                    neuron_inputs.push(idx);
                }
            }
            neuron_inputs.sort_unstable();
            connections.push(neuron_inputs);
        }

        Self {
            input_dim,
            expanded_dim,
            connections_per_neuron,
            k_winners,
            connections,
        }
    }

    /// Project a dense input vector into a sparse binary SDR via k-WTA inhibition
    pub fn project(&self, input: &[f32]) -> Sdr {
        assert_eq!(input.len(), self.input_dim, "Input vector size mismatch");

        // Step 1: Compute dendritic sum for each expanded neuron: a_i = sum_{j in conn(i)} x_j
        let mut activations: Vec<(usize, f32)> = Vec::with_capacity(self.expanded_dim);
        for (i, conns) in self.connections.iter().enumerate() {
            let mut sum = 0.0f32;
            for &c in conns {
                sum += input[c];
            }
            activations.push((i, sum));
        }

        // Step 2: Global lateral inhibition (k-Winners-Take-All)
        // Find top k indices with highest activation.
        // Using select_nth_unstable_by for O(M) expected time instead of O(M log M) sort.
        let k = self.k_winners;
        if k < self.expanded_dim {
            activations.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            activations.truncate(k);
        }

        let mut winners: Vec<usize> = activations.into_iter().map(|(idx, _)| idx).collect();
        winners.sort_unstable();

        Sdr {
            total_bits: self.expanded_dim,
            active_indices: winners,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strict_k_wta_sparsity() {
        let projector = FlyProjector::new(64, 1024, 6, 25, 42);
        let input = vec![1.0f32; 64];
        let sdr = projector.project(&input);

        assert_eq!(sdr.total_bits, 1024);
        assert_eq!(sdr.active_indices.len(), 25);
    }

    #[test]
    fn test_similarity_preservation_dasgupta() {
        let projector = FlyProjector::new(64, 2048, 6, 40, 1337);

        // Vector A
        let mut vec_a = vec![0.0f32; 64];
        for i in 0..32 {
            vec_a[i] = 1.0;
        }

        // Vector B: High similarity to A (perturbed slightly)
        let mut vec_b = vec_a.clone();
        vec_b[0] = 0.8;
        vec_b[1] = 0.9;
        vec_b[32] = 0.2; // slight noise

        // Vector C: Orthogonal to A
        let mut vec_c = vec![0.0f32; 64];
        for i in 32..64 {
            vec_c[i] = 1.0;
        }

        let sdr_a = projector.project(&vec_a);
        let sdr_b = projector.project(&vec_b);
        let sdr_c = projector.project(&vec_c);

        let sim_ab = sdr_a.jaccard_similarity(&sdr_b);
        let sim_ac = sdr_a.jaccard_similarity(&sdr_c);

        println!("Sim(A, B) = {:.4}, Sim(A, C) = {:.4}", sim_ab, sim_ac);

        // Dasgupta theorem: Similar inputs must produce significantly higher overlap than orthogonal ones
        assert!(sim_ab > 0.5, "Similar vectors must have high SDR overlap");
        assert!(sim_ac < 0.15, "Orthogonal vectors must have low SDR overlap");
    }
}
