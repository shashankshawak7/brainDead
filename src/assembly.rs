//! Pillar 2: Compositionality & Syntax via Assembly Calculus
//!
//! # Scientific Foundation
//! - Primary Paper: Mitropolsky, D., Collins, M., & Papadimitriou, C. (2021).
//!   "A Biologically Plausible Parser." Transactions of the Association for Computational
//!   Linguistics (TACL), 9, 1232-1244.
//! - Foundational Theory: Papadimitriou, C. H., Vempala, S. S., Mitropolsky, D., Collins, M.,
//!   & Maass, W. (2020). "Brain computation by assemblies of neurons."
//!   Proceedings of the National Academy of Sciences (PNAS), 117(25), 14464-14472.
//!
//! # Biological Principles
//! 1. Brain represents symbols, words, and concepts as **Cell Assemblies**: coherent populations
//!    of $k$ neurons within an area of $n$ neurons ($k \approx \sqrt{n}$, typically 1-2% active).
//! 2. Syntax emerges without backpropagation or soft-attention matrices ($O(N^2)$) through
//!    biologically grounded operations:
//!    - **Project**: Copies an active assembly into a target cortical area via feedforward synapses.
//!    - **Associate**: Reciprocally binds two assemblies firing simultaneously via Hebbian LTP.
//!    - **Merge**: Combines two active assemblies (e.g. Subject + Verb) into a new higher-order
//!      assembly in a downstream integration area, establishing reciprocal parent-child connections.
//! 3. Learning is strictly local Hebbian plasticity with homeostatic weight capping:
//!    $$w_{ij} \leftarrow w_{ij} \cdot (1 + \beta)$$

use crate::sdr::{DeterministicPrng, Sdr};

/// Synaptic weight representation between neurons.
/// Stored as a flat sparse/dense adjacency matrix to maximize cache locality.
#[derive(Debug, Clone)]
pub struct SynapseMatrix {
    /// Number of presynaptic neurons (rows)
    pub pre_neurons: usize,
    /// Number of postsynaptic neurons (cols)
    pub post_neurons: usize,
    /// Synaptic weights flattened: index = pre * post_neurons + post
    pub weights: Vec<f32>,
}

impl SynapseMatrix {
    /// Initialize random sparse synaptic connectivity.
    /// In biological cortex, connection probability $p$ is typically 5% to 15% (Erdos-Renyi G(n, p)).
    ///
    /// Reference: Papadimitriou et al. (PNAS 2020), Section "The Model", p. 14465.
    pub fn new_random(pre: usize, post: usize, p: f32, initial_weight: f32, seed: u64) -> Self {
        let mut rng = DeterministicPrng::new(seed);
        let total = pre * post;
        let mut weights = vec![0.0f32; total];
        let p_threshold = (p * 1000.0) as usize;

        for w in weights.iter_mut() {
            // Biological synapse existence governed by connection probability p
            if rng.next_bounded(1000) < p_threshold {
                *w = initial_weight;
            }
        }

        Self {
            pre_neurons: pre,
            post_neurons: post,
            weights,
        }
    }

    /// Compute postsynaptic dendritic drive for all postsynaptic neurons from active presynaptic indices.
    ///
    /// Math: $d_j = \sum_{i \in \text{active}} w_{ij}$
    /// Reference: Mitropolsky et al. (TACL 2021), Equation 1.
    pub fn compute_dendritic_drive(&self, active_pre: &[usize]) -> Vec<f32> {
        let mut drive = vec![0.0f32; self.post_neurons];
        for &pre_idx in active_pre {
            debug_assert!(pre_idx < self.pre_neurons, "Presynaptic index out of bounds");
            let row_offset = pre_idx * self.post_neurons;
            let row = &self.weights[row_offset..row_offset + self.post_neurons];
            for (post_idx, &weight) in row.iter().enumerate() {
                drive[post_idx] += weight;
            }
        }
        drive
    }

    /// Apply local Hebbian Long-Term Potentiation (LTP) across active pre- and postsynaptic pairs.
    ///
    /// Math: $w_{ij} \leftarrow \min(w_{\max}, w_{ij} \cdot (1 + \beta))$
    /// Reference: Papadimitriou et al. (PNAS 2020), "Plasticity rule", p. 14466.
    pub fn hebbian_update(
        &mut self,
        active_pre: &[usize],
        active_post: &[usize],
        beta: f32,
        max_weight: f32,
    ) {
        for &pre_idx in active_pre {
            let row_offset = pre_idx * self.post_neurons;
            for &post_idx in active_post {
                let idx = row_offset + post_idx;
                let current = self.weights[idx];
                // Only potentiate existing anatomical connections (structural plasticity is distinct)
                if current > 0.0 {
                    self.weights[idx] = (current * (1.0 + beta)).min(max_weight);
                }
            }
        }
    }
}

/// A Cortical Area hosting assemblies.
/// Contains $n$ excitatory neurons governed by fast global parvalbumin-positive interneuron inhibition
/// ensuring strict k-Winners-Take-All (k-WTA).
///
/// Reference: Mitropolsky et al. (TACL 2021), Section 3.1 "Brain Architecture".
#[derive(Debug, Clone)]
pub struct CorticalArea {
    /// Name/Identifier of the cortical area (e.g. "Lexicon", "Syntax", "Integrator")
    pub name: String,
    /// Total number of pyramidal neurons in the area ($n$)
    pub n_neurons: usize,
    /// Number of neurons per active assembly ($k \approx \sqrt{n}$)
    pub k_assembly_size: usize,
    /// Recurrent synaptic connectivity within this area: $n \times n$
    pub recurrent_synapses: SynapseMatrix,
}

impl CorticalArea {
    /// Initialize a cortical area with random recurrent wiring.
    ///
    /// Reference: Papadimitriou et al. (PNAS 2020), p. 14465.
    pub fn new(name: &str, n: usize, k: usize, recurrent_p: f32, seed: u64) -> Self {
        assert!(k <= n, "Assembly size k cannot exceed total neurons n");
        let recurrent_synapses = SynapseMatrix::new_random(n, n, recurrent_p, 1.0, seed);
        Self {
            name: name.to_string(),
            n_neurons: n,
            k_assembly_size: k,
            recurrent_synapses,
        }
    }

    /// Select the top $k$ most depolarized neurons (k-WTA inhibition).
    ///
    /// Reference: Mitropolsky et al. (TACL 2021), "Operation of an Area".
    pub fn apply_k_wta(&self, total_dendritic_drive: &[f32]) -> Sdr {
        assert_eq!(
            total_dendritic_drive.len(),
            self.n_neurons,
            "Drive dimension mismatch"
        );

        let mut indexed: Vec<(usize, f32)> = total_dendritic_drive
            .iter()
            .copied()
            .enumerate()
            .collect();

        // O(n) selection of top k winners via quickselect
        let k = self.k_assembly_size;
        if k < self.n_neurons {
            indexed.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(k);
        }

        let mut winners: Vec<usize> = indexed.into_iter().map(|(idx, _)| idx).collect();
        winners.sort_unstable();

        Sdr {
            total_bits: self.n_neurons,
            active_indices: winners,
        }
    }
}

/// The Assembly Calculus Engine.
/// Implements the discrete operations that prove Turing-complete syntactic parsing without backpropagation.
///
/// Reference: Mitropolsky, Collins, & Papadimitriou (TACL 2021), Section 3.2.
#[derive(Debug)]
pub struct AssemblyEngine {
    /// Hebbian learning rate ($\beta$)
    pub beta: f32,
    /// Maximum synaptic weight cap ($w_{\max}$)
    pub max_weight: f32,
}

impl AssemblyEngine {
    /// Initialize with biological plasticity parameters
    /// Typical $\beta \in [0.05, 0.15]$, $w_{\max} \in [5.0, 10.0]$
    pub fn new(beta: f32, max_weight: f32) -> Self {
        Self { beta, max_weight }
    }

    /// **Project Operation**:
    /// Fires assembly `source_assembly` in area `from`, propagating drive through feedforward
    /// projection `projection` to activate and potentiate a target assembly in area `to`.
    ///
    /// Reference: Papadimitriou et al. (PNAS 2020), Section "project(A, B)", p. 14466.
    pub fn project(
        &self,
        source_assembly: &Sdr,
        _from_area: &CorticalArea,
        to_area: &mut CorticalArea,
        projection: &mut SynapseMatrix,
    ) -> Sdr {
        // Step 1: Compute feedforward drive from source assembly into target area
        let feedforward_drive = projection.compute_dendritic_drive(&source_assembly.active_indices);

        // Step 2: Target area executes k-WTA inhibition to select winning neurons
        let target_assembly = to_area.apply_k_wta(&feedforward_drive);

        // Step 3: Hebbian plasticity potentiates feedforward synapses
        projection.hebbian_update(
            &source_assembly.active_indices,
            &target_assembly.active_indices,
            self.beta,
            self.max_weight,
        );

        // Step 4: Hebbian plasticity potentiates recurrent synapses within target assembly
        to_area.recurrent_synapses.hebbian_update(
            &target_assembly.active_indices,
            &target_assembly.active_indices,
            self.beta,
            self.max_weight,
        );

        target_assembly
    }

    /// **Merge Operation**:
    /// Takes two active constituent assemblies `assembly_a` (e.g., Subject / Noun) and
    /// `assembly_b` (e.g., Verb / Predicate), simultaneously drives integration area `integrator`,
    /// creates merged parent assembly $C = \text{Merge}(A, B)$, and forms reciprocal connections.
    ///
    /// Reference: Mitropolsky, Collins, & Papadimitriou (TACL 2021), "merge(A, B, C)", Section 3.2.
    pub fn merge(
        &self,
        assembly_a: &Sdr,
        assembly_b: &Sdr,
        proj_a_to_c: &mut SynapseMatrix,
        proj_b_to_c: &mut SynapseMatrix,
        integrator_area: &mut CorticalArea,
    ) -> Sdr {
        // Step 1: Simultaneous convergent dendritic drive into integrator area
        let drive_a = proj_a_to_c.compute_dendritic_drive(&assembly_a.active_indices);
        let drive_b = proj_b_to_c.compute_dendritic_drive(&assembly_b.active_indices);

        let mut total_drive = vec![0.0f32; integrator_area.n_neurons];
        for i in 0..integrator_area.n_neurons {
            total_drive[i] = drive_a[i] + drive_b[i];
        }

        // Step 2: Integrator area k-WTA selects the composite assembly
        let merged_assembly = integrator_area.apply_k_wta(&total_drive);

        // Step 3: Hebbian potentiation of convergent inputs A -> C and B -> C
        proj_a_to_c.hebbian_update(
            &assembly_a.active_indices,
            &merged_assembly.active_indices,
            self.beta,
            self.max_weight,
        );
        proj_b_to_c.hebbian_update(
            &assembly_b.active_indices,
            &merged_assembly.active_indices,
            self.beta,
            self.max_weight,
        );

        // Step 4: Multi-round attractor reverberation (Papadimitriou et al. 2020, Section "Assembly Formation")
        // An assembly requires multiple reverberation cycles (T ~ 3-5) to establish recurrent attractor basin
        let mut current_assembly = merged_assembly;
        for _ in 0..4 {
            let recurrent_drive = integrator_area
                .recurrent_synapses
                .compute_dendritic_drive(&current_assembly.active_indices);
            
            // Combine drive from constituents + internal recurrent drive
            let mut total = vec![0.0f32; integrator_area.n_neurons];
            for i in 0..integrator_area.n_neurons {
                total[i] = drive_a[i] + drive_b[i] + recurrent_drive[i];
            }
            
            let next_assembly = integrator_area.apply_k_wta(&total);
            integrator_area.recurrent_synapses.hebbian_update(
                &current_assembly.active_indices,
                &next_assembly.active_indices,
                self.beta,
                self.max_weight,
            );
            current_assembly = next_assembly;
        }

        current_assembly
    }

    /// **Pattern Completion (Attractor Settling)**:
    /// Given a partial or noisy cue of an assembly, recurrent excitation iterates across
    /// multiple gamma cycles (typically 2-4 steps) until the network settles into the full assembly attractor.
    ///
    /// Reference: Papadimitriou et al. (PNAS 2020), Section "Pattern Completion", p. 14467.
    pub fn pattern_complete(&self, partial_cue: &Sdr, area: &CorticalArea, settling_steps: usize) -> Sdr {
        let mut current = partial_cue.clone();
        for _ in 0..settling_steps.max(1) {
            let recurrent_drive = area
                .recurrent_synapses
                .compute_dendritic_drive(&current.active_indices);
            current = area.apply_k_wta(&recurrent_drive);
        }
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_operation_stability() {
        // 1000 neurons, k=25 assembly size (~2.5% sparsity)
        let area_a = CorticalArea::new("AreaA", 1000, 25, 0.1, 42);
        let mut area_b = CorticalArea::new("AreaB", 1000, 25, 0.1, 84);
        let mut proj_a_b = SynapseMatrix::new_random(1000, 1000, 0.15, 1.0, 126);

        let engine = AssemblyEngine::new(0.1, 5.0);

        let sdr_a = Sdr {
            total_bits: 1000,
            active_indices: (0..25).collect(),
        };

        // Project once
        let target_1 = engine.project(&sdr_a, &area_a, &mut area_b, &mut proj_a_b);
        assert_eq!(target_1.active_indices.len(), 25);

        // Project second time with Hebbian strengthening
        let target_2 = engine.project(&sdr_a, &area_a, &mut area_b, &mut proj_a_b);

        // Convergence theorem (Papadimitriou et al. 2020):
        // Repeated projection of same source assembly converges to stable target assembly
        let overlap = target_1.overlap(&target_2);
        assert!(
            overlap >= 20,
            "Target assembly must converge to stable attractor (overlap: {}/25)",
            overlap
        );
    }

    #[test]
    fn test_merge_operation_syntactic_binding() {
        // 500 neurons, k=25 assembly size (5% sparsity, standard cortical column density)
        // With recurrent_p = 0.2, within-assembly recurrent fan-in is 25 * 0.2 = 5 connections
        let mut area_c = CorticalArea::new("ClauseIntegrator", 500, 25, 0.2, 101);

        let mut proj_a_c = SynapseMatrix::new_random(500, 500, 0.2, 1.0, 202);
        let mut proj_b_c = SynapseMatrix::new_random(500, 500, 0.2, 1.0, 303);

        // High plasticity during consolidation phase (beta = 0.8, max_weight = 8.0)
        let engine = AssemblyEngine::new(0.8, 8.0);

        // Assembly A: "Dog" (indices 0..25)
        let noun_assembly = Sdr {
            total_bits: 500,
            active_indices: (0..25).collect(),
        };

        // Assembly B: "Barks" (indices 50..75)
        let verb_assembly = Sdr {
            total_bits: 500,
            active_indices: (50..75).collect(),
        };

        // Execute Merge(Noun, Verb) -> Sentence Assembly in Area C
        let merged_clause = engine.merge(
            &noun_assembly,
            &verb_assembly,
            &mut proj_a_c,
            &mut proj_b_c,
            &mut area_c,
        );

        assert_eq!(merged_clause.active_indices.len(), 25);

        // Test pattern completion: Even with only 40% of the merged assembly active,
        // recurrent synapses in Area C should recover the full merged assembly!
        let partial_cue = Sdr {
            total_bits: 500,
            active_indices: merged_clause.active_indices[0..10].to_vec(),
        };

        let completed = engine.pattern_complete(&partial_cue, &area_c, 3);
        let recovery_overlap = merged_clause.overlap(&completed);

        assert!(
            recovery_overlap >= 20,
            "Recurrent connections in Area C must complete merged assembly (recovered: {}/25)",
            recovery_overlap
        );
    }
}
