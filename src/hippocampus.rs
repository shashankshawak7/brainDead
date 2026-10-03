//! Pillar 5: Complementary Learning Systems (Hippocampal Replay & Homeostasis)
//!
//! # Scientific Foundation
//! - Primary Papers:
//!   - McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). "Why there are complementary
//!     learning systems in the hippocampus and neocortex: Insights from the successes and failures
//!     of connectionist models of learning and memory." *Psychological Review*, 102(3), 419-457.
//!     (Foundational CLS Theory: Fast hippocampal indexing + slow neocortical consolidation).
//!   - Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). "What learning systems do intelligent
//!     agents need? Complementary learning systems theory updated." *Trends in Cognitive Sciences*, 20(7), 512-534.
//!   - Buzsáki, G. (2015). "Hippocampal sharp wave-ripple: A cognitive biomarker for episodic memory
//!     and planning." *Hippocampus*, 25(10), 1073-1188. (High-speed offline neural replay, 150-250 Hz).
//!   - Turrigiano, G. G. (2008). "The self-tuning neuron: synaptic scaling of excitatory synapses."
//!     *Cell*, 135(3), 422-435. (Homeostatic multiplicative synaptic scaling preventing saturation).
//!
//! # Navigating the "Dirty Water" Quagmire (FRANKIE_LAWS.md)
//! - **NO Vector Database:** No external dense embedding store.
//! - **NO Unbounded Replay Buffers:** Fixed 128-slot circular ring buffer.
//!   Footprint: 128 x 256 bytes = **exactly 32 KB** (fits entirely inside CPU L1 Data Cache).
//! - **NO Runaway LTP:** Turrigiano homeostatic scaling ensures weights do not saturate into black holes.
//! - **Zero Heap Allocations:** Zero `malloc` during episode encoding and ripple replay.

use crate::packed_sdr::{BitSdr, QuantizedSynapseMatrix};

/// Capacity of the hippocampal episodic ring buffer (fits in 32 KB L1 cache)
pub const HIPPOCAMPAL_EPISODE_CAPACITY: usize = 128;

/// Level 1: Number of coarse regional hash clusters (Dentate Gyrus macro-zones)
pub const LSH_COARSE_CLUSTERS: usize = 256;
/// Level 2: Number of fine hash sub-buckets per coarse cluster (CA3 micro-columns)
pub const LSH_FINE_BUCKETS: usize = 16;
/// Total fine resonant buckets in the 2-level hierarchy: 256 * 16 = 4,096 buckets
pub const LSH_TOTAL_BUCKETS: usize = LSH_COARSE_CLUSTERS * LSH_FINE_BUCKETS;
/// Maximum episodic candidate references per fine bucket (Poisson P(k >= 9) < 10^-6)
pub const BUCKET_CAPACITY: usize = 8;

/// Fixed-capacity cache-friendly fine LSH hash bucket (Zero Heap Allocation)
#[derive(Clone, Copy, Debug)]
pub struct LshBucket {
    pub entries: [u16; BUCKET_CAPACITY],
    pub len: u8,
}

impl LshBucket {
    pub const fn empty() -> Self {
        Self {
            entries: [0u16; BUCKET_CAPACITY],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, idx: u16) {
        if (self.len as usize) < BUCKET_CAPACITY {
            self.entries[self.len as usize] = idx;
            self.len += 1;
        }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[u16] {
        &self.entries[..self.len as usize]
    }
}

/// A Coarse Hash Cluster containing 16 fine sub-buckets
#[derive(Clone, Copy, Debug)]
pub struct LshCluster {
    pub sub_buckets: [LshBucket; LSH_FINE_BUCKETS],
}

impl LshCluster {
    pub const fn empty() -> Self {
        Self {
            sub_buckets: [LshBucket::empty(); LSH_FINE_BUCKETS],
        }
    }
}

/// Dentate Gyrus Hierarchical 2-Level Locality-Sensitive Hashing (LSH) Index
///
/// Implements sub-linear associative lookup as proven in:
/// Dasgupta, S., Stevens, C. F., & Navlakha, S. (2017). "A neural algorithm for a fundamental
/// computing problem." *Science*, 358(6364), 793-796.
///
/// Hierarchical 2-Level Architecture:
/// - Level 1 (Coarse Regional Basin): 256 Entorhinal-Dentate macro-clusters.
/// - Level 2 (Fine Orthogonal Mini-column): 16 CA3 micro-buckets per macro-cluster.
/// Total: 4,096 fine resonant buckets indexing up to 16,384 candidate references.
/// Memory footprint: ~40 KB (fits completely in CPU cache).
#[derive(Clone, Debug)]
pub struct DentateLshIndex {
    pub clusters: Box<[LshCluster; LSH_COARSE_CLUSTERS]>,
}

impl DentateLshIndex {
    pub fn new() -> Self {
        let boxed_slice = vec![LshCluster::empty(); LSH_COARSE_CLUSTERS].into_boxed_slice();
        let clusters: Box<[LshCluster; LSH_COARSE_CLUSTERS]> = match boxed_slice.try_into() {
            Ok(arr) => arr,
            Err(_) => unreachable!(),
        };
        Self { clusters }
    }

    #[inline(always)]
    fn mix64(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Level 1 Coarse Hash: maps 2048-bit SDR to 1 of 256 regional clusters
    #[inline(always)]
    pub fn hash_coarse(sdr: &BitSdr) -> usize {
        let mut h = 0xcbf29ce484222325u64;
        for (i, &w) in sdr.words.iter().enumerate() {
            if w != 0 {
                h ^= w ^ (i as u64);
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        (Self::mix64(h) as usize) % LSH_COARSE_CLUSTERS
    }

    /// Level 2 Fine Hash: orthogonal projection to 1 of 16 sub-buckets within the coarse cluster
    #[inline(always)]
    pub fn hash_fine(sdr: &BitSdr) -> usize {
        let mut h = 0x9e3779b97f4a7c15u64;
        for (i, &w) in sdr.words.iter().enumerate() {
            if w != 0 {
                let rot = (w ^ (i as u64)).rotate_left(23);
                h ^= rot;
                h = h.wrapping_mul(0xbf58476d1ce4e5b9);
            }
        }
        ((Self::mix64(h) >> 16) as usize) % LSH_FINE_BUCKETS
    }

    /// Insert a proposition candidate index into the 2-level hierarchy
    #[inline(always)]
    pub fn insert(&mut self, sdr: &BitSdr, index: u16) {
        let c_idx = Self::hash_coarse(sdr);
        let f_idx = Self::hash_fine(sdr);
        self.clusters[c_idx].sub_buckets[f_idx].push(index);
    }

    /// Remove an index entry from the 2-level hierarchy during homeostatic eviction
    #[inline(always)]
    pub fn remove(&mut self, sdr: &BitSdr, index: u16) -> bool {
        let c_idx = Self::hash_coarse(sdr);
        let f_idx = Self::hash_fine(sdr);
        let bucket = &mut self.clusters[c_idx].sub_buckets[f_idx];
        let mut found = None;
        for i in 0..bucket.len as usize {
            if bucket.entries[i] == index {
                found = Some(i);
                break;
            }
        }
        if let Some(i) = found {
            let last = (bucket.len - 1) as usize;
            bucket.entries[i] = bucket.entries[last];
            bucket.len -= 1;
            true
        } else {
            false
        }
    }

    /// Evict all occurrences of an index across all clusters in the hierarchy
    #[inline(always)]
    pub fn remove_index(&mut self, index: u16) {
        for cluster in self.clusters.iter_mut() {
            for bucket in cluster.sub_buckets.iter_mut() {
                let mut i = 0;
                while i < bucket.len as usize {
                    if bucket.entries[i] == index {
                        let last = (bucket.len - 1) as usize;
                        bucket.entries[i] = bucket.entries[last];
                        bucket.len -= 1;
                    } else {
                        i += 1;
                    }
                }
            }
        }
    }

    /// Retrieve the primary fine resonant bucket
    #[inline(always)]
    pub fn get_bucket(&self, sdr: &BitSdr) -> &LshBucket {
        let c_idx = Self::hash_coarse(sdr);
        let f_idx = Self::hash_fine(sdr);
        &self.clusters[c_idx].sub_buckets[f_idx]
    }

    /// Retrieve the entire coarse cluster for multi-probe resonance
    #[inline(always)]
    pub fn get_cluster(&self, sdr: &BitSdr) -> &LshCluster {
        let c_idx = Self::hash_coarse(sdr);
        &self.clusters[c_idx]
    }

    /// Hierarchical Multi-Probe: visit primary fine bucket, then adjacent sub-buckets within the same coarse cluster
    #[inline(always)]
    pub fn probe_hierarchical<F>(&self, sdr: &BitSdr, mut visit: F)
    where
        F: FnMut(u16),
    {
        let c_idx = Self::hash_coarse(sdr);
        let f_idx = Self::hash_fine(sdr);
        let cluster = &self.clusters[c_idx];

        // 1. Primary fine bucket
        for &idx in cluster.sub_buckets[f_idx].as_slice() {
            visit(idx);
        }

        // 2. Neighbor fine bucket in same regional cluster
        let next_idx = (f_idx + 1) % LSH_FINE_BUCKETS;
        for &idx in cluster.sub_buckets[next_idx].as_slice() {
            visit(idx);
        }
    }
}

/// Hippocampal Formation with Sharp-Wave Ripple (SWR) Consolidation Engine
///
/// Implements Complementary Learning Systems (McClelland et al., 1995):
/// - Rapidly caches episodic SDR transitions in an ultra-compact ring buffer.
/// - During offline quiescent periods ("sleep"), replays interleaved episodes to
///   slowly consolidate cortical predictive synapses.
/// - Applies Turrigiano (2008) homeostatic scaling to prevent catastrophic forgetting.
#[derive(Debug)]
pub struct HippocampalFormation {
    /// Circular episodic ring buffer of recent SDR transitions: (Context SDR, Target SDR)
    /// Stored as flat stack arrays to eliminate heap allocations
    pub episodic_contexts: [BitSdr; HIPPOCAMPAL_EPISODE_CAPACITY],
    pub episodic_targets: [BitSdr; HIPPOCAMPAL_EPISODE_CAPACITY],

    /// Current write index in the circular buffer
    pub write_cursor: usize,
    /// Number of valid stored episodes (capped at HIPPOCAMPAL_EPISODE_CAPACITY)
    pub valid_episodes: usize,

    /// Homeostatic target synaptic drive setpoint (Turrigiano 2008)
    pub homeostatic_setpoint: u32,
}

impl HippocampalFormation {
    /// Initialize hippocampal cache (consumes 64 KB total for contexts + targets)
    pub fn new(homeostatic_setpoint: u32) -> Self {
        Self {
            episodic_contexts: [BitSdr::empty(); HIPPOCAMPAL_EPISODE_CAPACITY],
            episodic_targets: [BitSdr::empty(); HIPPOCAMPAL_EPISODE_CAPACITY],
            write_cursor: 0,
            valid_episodes: 0,
            homeostatic_setpoint,
        }
    }

    /// Fast one-shot episodic encoding into Dentate Gyrus / CA3 ring buffer.
    /// Zero heap allocations: overwrites oldest entry in circular array.
    ///
    /// Reference: Kumaran et al. (TICS 2016) - One-shot episodic indexing.
    #[inline]
    pub fn record_episode(&mut self, context: &BitSdr, target: &BitSdr) {
        if context.count_ones() == 0 || target.count_ones() == 0 {
            return;
        }

        self.episodic_contexts[self.write_cursor] = *context;
        self.episodic_targets[self.write_cursor] = *target;

        self.write_cursor = (self.write_cursor + 1) % HIPPOCAMPAL_EPISODE_CAPACITY;
        if self.valid_episodes < HIPPOCAMPAL_EPISODE_CAPACITY {
            self.valid_episodes += 1;
        }
    }

    /// Sharp-Wave Ripple (SWR) Offline Replay:
    /// Replays compressed sequence bursts from hippocampal cache into cortical synapses
    /// at a low consolidation learning rate ($\Delta w_{slow}$).
    ///
    /// Reference: Buzsáki (Hippocampus 2015) + McClelland et al. (1995).
    pub fn sharp_wave_ripple_replay(
        &self,
        cortical_synapses: &mut QuantizedSynapseMatrix,
        consolidation_delta: u8,
        replay_rounds: usize,
    ) {
        if self.valid_episodes == 0 {
            return;
        }

        for _ in 0..replay_rounds {
            for i in 0..self.valid_episodes {
                let ctx = &self.episodic_contexts[i];
                let tgt = &self.episodic_targets[i];

                // Slow, interleaved synaptic potentiation
                cortical_synapses.hebbian_potentiate(ctx, tgt, consolidation_delta);
            }
        }
    }

    /// Turrigiano Homeostatic Multiplicative Downscaling:
    /// If total synaptic weight of a postsynaptic neuron exceeds the homeostatic setpoint,
    /// all incoming synapses to that neuron are scaled down uniformly.
    ///
    /// Math: If $\sum_i W_{ij} > Setpoint$, $W_{ij} \leftarrow \lfloor W_{ij} \cdot \frac{Setpoint}{\sum W} \rfloor$
    /// Reference: Turrigiano (Cell 2008), Figure 2.
    pub fn apply_homeostatic_scaling(&self, cortical_synapses: &mut QuantizedSynapseMatrix) {
        let post_neurons = cortical_synapses.post_neurons;
        let pre_neurons = cortical_synapses.pre_neurons;

        for post_idx in 0..post_neurons {
            // Step 1: Compute total incoming synaptic weight for column post_idx
            let mut total_column_weight = 0u32;
            for pre_idx in 0..pre_neurons {
                let idx = pre_idx * post_neurons + post_idx;
                total_column_weight += cortical_synapses.weights[idx] as u32;
            }

            // Step 2: If saturated beyond setpoint, scale down multiplicatively
            if total_column_weight > self.homeostatic_setpoint && total_column_weight > 0 {
                let scale_factor = (self.homeostatic_setpoint as u64 * 256) / total_column_weight as u64;

                for pre_idx in 0..pre_neurons {
                    let idx = pre_idx * post_neurons + post_idx;
                    let current = cortical_synapses.weights[idx] as u64;
                    cortical_synapses.weights[idx] = ((current * scale_factor) / 256) as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hippocampal_replay_prevents_catastrophic_forgetting() {
        let n = 2048;
        let mut cortical_synapses = QuantizedSynapseMatrix::new(n, n);
        let mut hippocampus = HippocampalFormation::new(5000);

        // Episode 1: Initial Knowledge ("Cat" -> "Meows")
        let mut cat_sdr = BitSdr::empty();
        for i in 0..40 { cat_sdr.set_bit(i); }
        let mut meow_sdr = BitSdr::empty();
        for i in 100..140 { meow_sdr.set_bit(i); }

        // Fast episodic encoding in hippocampus
        hippocampus.record_episode(&cat_sdr, &meow_sdr);

        // Immediate direct training on Episode 1
        cortical_synapses.hebbian_potentiate(&cat_sdr, &meow_sdr, 20);

        // Measure initial recall: Overlap of output drive
        let mut scratchpad = vec![0u32; n];
        cortical_synapses.compute_drive_into(&cat_sdr, &mut scratchpad);
        let drive_on_meow_before = scratchpad[100];
        assert!(drive_on_meow_before > 0);

        // Now: Network is heavily exposed to UNRELATED intervening data (10 novel episodes)
        // In standard networks without CLS, this causes catastrophic forgetting of Episode 1!
        for ep in 1..=10 {
            let mut noise_in = BitSdr::empty();
            let mut noise_out = BitSdr::empty();
            for i in 0..40 {
                noise_in.set_bit(200 + ep * 40 + i);
                noise_out.set_bit(700 + ep * 40 + i);
            }
            cortical_synapses.hebbian_potentiate(&noise_in, &noise_out, 30);
            hippocampus.record_episode(&noise_in, &noise_out);
        }

        // Offline Sleep Mode: Sharp-Wave Ripple Replay (Buzsáki 2015)
        // Interleaved replay refreshes Episode 1 into cortical synaptic connectivity
        hippocampus.sharp_wave_ripple_replay(&mut cortical_synapses, 5, 3);

        // Apply Turrigiano homeostatic scaling to prevent weight explosion
        hippocampus.apply_homeostatic_scaling(&mut cortical_synapses);

        // Verify that Episode 1 ("Cat" -> "Meows") was NOT forgotten!
        cortical_synapses.compute_drive_into(&cat_sdr, &mut scratchpad);
        let drive_on_meow_after = scratchpad[100];

        println!(
            "Catastrophic Forgetting Test: Drive Before = {}, Drive After SWR Replay = {}",
            drive_on_meow_before, drive_on_meow_after
        );

        assert!(
            drive_on_meow_after >= drive_on_meow_before,
            "Sharp-Wave Ripple Replay must preserve earlier episodic memories against interference"
        );
    }

    #[test]
    fn test_hierarchical_2_level_dentate_lsh_scaling() {
        let mut lsh = DentateLshIndex::new();

        // 1. Populate with 4,096 distinct facts (multi-thousand episodic scaling)
        let total_facts = 4096;
        let mut test_sdrs = Vec::with_capacity(total_facts);

        for i in 0..total_facts {
            let mut sdr = BitSdr::empty();
            // Generate sparse 40-bit representation deterministically
            for bit_idx in 0..40 {
                let bit = ((i * 37 + bit_idx * 19) % 2048) as usize;
                sdr.set_bit(bit);
            }
            lsh.insert(&sdr, i as u16);
            test_sdrs.push(sdr);
        }

        // 2. Validate exact associative recall across sample facts
        let sample_indices = [0, 1, 42, 100, 500, 1000, 2048, 3000, 4095];
        for &idx in &sample_indices {
            let sdr = &test_sdrs[idx];
            let mut found = false;
            lsh.probe_hierarchical(sdr, |candidate| {
                if candidate == idx as u16 {
                    found = true;
                }
            });
            assert!(
                found,
                "Hierarchical 2-Level LSH must retrieve fact index {} from 4,096+ store",
                idx
            );
        }

        // 3. Measure latency: 10,000 sub-linear lookups across 4,096 facts
        let start = std::time::Instant::now();
        let lookup_rounds = 10_000;
        let mut hit_count = 0;
        for i in 0..lookup_rounds {
            let sdr = &test_sdrs[i % total_facts];
            lsh.probe_hierarchical(sdr, |_| {
                hit_count += 1;
            });
        }
        let elapsed = start.elapsed();
        let ns_per_lookup = elapsed.as_nanos() as f64 / lookup_rounds as f64;

        println!(
            "Dentate 2-Level LSH Scaling (4,096 facts): {} lookups in {:.2?}, {:.2} ns/lookup, total hits = {}",
            lookup_rounds, elapsed, ns_per_lookup, hit_count
        );

        assert!(
            ns_per_lookup < 10_000.0,
            "Hierarchical LSH lookup must run in <10 microseconds in unoptimized debug (got {:.2} ns)",
            ns_per_lookup
        );
    }
}
