//! Pillar 3: Activity-Silent Working Memory via Short-Term Synaptic Plasticity (STP)
//!
//! # Scientific Foundation
//! - Primary Paper: Mongillo, G., Barak, O., & Tsodyks, M. (2008).
//!   "Synaptic theory of working memory." Science, 319(5869), 1543-1546.
//! - Supporting Paper: Barak, O., & Tsodyks, M. (2014).
//!   "Working models of working memory." Current Opinion in Neurobiology, 25, 20-24.
//!
//! # Biological Principles
//! 1. Brain does **not** maintain working memory through continuous energetic spiking (persistent reverberation)
//!    nor through artificial dense float buffers with arbitrary decay ($0.95^t$).
//! 2. Context is retained in the **synaptic state** itself via calcium-mediated facilitation:
//!    - Each action potential triggers an influx of presynaptic calcium $u(t)$, increasing release probability.
//!    - Available neurotransmitter vesicle pool $x(t)$ is depleted upon vesicle release and slowly recovers.
//! 3. **Activity-Silent Retention**: When network spikes cease, the memory persists in elevated $u(t)$
//!    over seconds ($\tau_F \sim 1500\text{ ms}$) at zero metabolic spiking cost.
//! 4. **Memory Readout**: A non-specific baseline retrieval cue automatically reactivates the stored assembly
//!    because facilitated synapses provide enhanced transmission.

use crate::sdr::Sdr;

/// Tsodyks-Markram Short-Term Plasticity (STP) synapse array.
///
/// Implements differential equations from Mongillo et al. (Science 2008), p. 1544:
/// - Calcium facilitation factor $u(t) \in [0, 1]$
/// - Neurotransmitter vesicle availability $x(t) \in [0, 1]$
/// - Baseline utilization parameter $U \in (0, 1)$
/// - Facilitation time constant $\tau_F$ (typically ~1500 ms)
/// - Depression time constant $\tau_D$ (typically ~200 ms)
#[derive(Debug, Clone)]
pub struct StpSynapseArray {
    pub pre_neurons: usize,
    pub post_neurons: usize,
    /// Baseline synaptic weight (anatomical wiring)
    pub baseline_weights: Vec<f32>,
    /// Presynaptic calcium release probability $u \in [0, 1]$ for each presynaptic neuron
    pub u_facilitation: Vec<f32>,
    /// Available neurotransmitter resource $x \in [0, 1]$ for each presynaptic neuron
    pub x_depression: Vec<f32>,
    /// Baseline release probability ($U$)
    pub baseline_u: f32,
    /// Facilitation time constant ($\tau_F$) in milliseconds
    pub tau_f_ms: f32,
    /// Depression time constant ($\tau_D$) in milliseconds
    pub tau_d_ms: f32,
}

impl StpSynapseArray {
    /// Initialize synapse array with resting biophysical equilibrium:
    /// At rest, $u = U$ and $x = 1.0$.
    ///
    /// Reference: Mongillo et al. (Science 2008), Equations 1-2.
    pub fn new(
        pre: usize,
        post: usize,
        baseline_weights: Vec<f32>,
        baseline_u: f32,
        tau_f_ms: f32,
        tau_d_ms: f32,
    ) -> Self {
        assert_eq!(baseline_weights.len(), pre * post);
        Self {
            pre_neurons: pre,
            post_neurons: post,
            baseline_weights,
            u_facilitation: vec![baseline_u; pre],
            x_depression: vec![1.0f32; pre],
            baseline_u,
            tau_f_ms,
            tau_d_ms,
        }
    }

    /// Advance time by $\Delta t$ milliseconds without spikes (decay towards rest).
    ///
    /// Math:
    /// $\frac{du}{dt} = \frac{U - u}{\tau_F} \implies u(t + \Delta t) = U + (u(t) - U) e^{-\Delta t / \tau_F}$
    /// $\frac{dx}{dt} = \frac{1 - x}{\tau_D} \implies x(t + \Delta t) = 1 + (x(t) - 1) e^{-\Delta t / \tau_D}$
    ///
    /// Reference: Mongillo et al. (Science 2008), p. 1544.
    pub fn decay_silent_period(&mut self, dt_ms: f32) {
        let decay_u = (-dt_ms / self.tau_f_ms).exp();
        let decay_x = (-dt_ms / self.tau_d_ms).exp();

        for u in self.u_facilitation.iter_mut() {
            *u = self.baseline_u + (*u - self.baseline_u) * decay_u;
        }

        for x in self.x_depression.iter_mut() {
            *x = 1.0 + (*x - 1.0) * decay_x;
        }
    }

    /// Process incoming action potentials (spikes) from presynaptic neurons.
    /// Updates $u$ and $x$, and returns effective postsynaptic current:
    ///
    /// Math:
    /// $u \leftarrow u + U (1 - u)$ (Calcium influx)
    /// $I_j = \sum_{i \in \text{spikes}} W_{ij} \cdot u_i \cdot x_i$ (Synaptic transmission)
    /// $x \leftarrow x - u \cdot x$ (Vesicle depletion)
    ///
    /// Reference: Mongillo et al. (Science 2008), Equations 1-2.
    pub fn transmit_and_update(&mut self, presynaptic_spikes: &[usize]) -> Vec<f32> {
        let mut postsynaptic_current = vec![0.0f32; self.post_neurons];

        for &pre_idx in presynaptic_spikes {
            debug_assert!(pre_idx < self.pre_neurons);

            // Influx of calcium facilitates release probability
            let u_prev = self.u_facilitation[pre_idx];
            let u_new = u_prev + self.baseline_u * (1.0 - u_prev);
            self.u_facilitation[pre_idx] = u_new;

            // Transmitted current is proportional to effective efficacy: W * u * x
            let x_current = self.x_depression[pre_idx];
            let efficacy = u_new * x_current;

            let row_offset = pre_idx * self.post_neurons;
            for post_idx in 0..self.post_neurons {
                let w = self.baseline_weights[row_offset + post_idx];
                postsynaptic_current[post_idx] += w * efficacy;
            }

            // Deplete available neurotransmitter resource
            self.x_depression[pre_idx] = (x_current - u_new * x_current).max(0.0);
        }

        postsynaptic_current
    }

    /// Check if presynaptic neuron has active facilitated memory trace: $u_i > 1.5 \cdot U$.
    pub fn is_facilitated(&self, pre_idx: usize) -> bool {
        self.u_facilitation[pre_idx] > self.baseline_u * 1.5
    }
}

/// Working Memory Buffer operating via Activity-Silent Synaptic Facilitation
#[derive(Debug)]
pub struct ActivitySilentMemory {
    pub synapses: StpSynapseArray,
}

impl ActivitySilentMemory {
    /// Create working memory module with Mongillo et al. (2008) biological parameters:
    /// $U = 0.2$, $\tau_F = 1500\text{ ms}$, $\tau_D = 200\text{ ms}$.
    pub fn new(n_neurons: usize, baseline_weights: Vec<f32>) -> Self {
        Self {
            synapses: StpSynapseArray::new(
                n_neurons,
                n_neurons,
                baseline_weights,
                0.2,    // U: baseline calcium release probability
                1500.0, // tau_F: 1.5 seconds facilitation decay
                200.0,  // tau_D: 200 ms depression recovery
            ),
        }
    }

    /// Encode an incoming SDR token into short-term synaptic memory trace
    pub fn encode(&mut self, sdr: &Sdr) {
        let _ = self.synapses.transmit_and_update(&sdr.active_indices);
    }

    /// Advance time during inter-word or inter-phrase intervals without spiking
    pub fn elapse_time(&mut self, dt_ms: f32) {
        self.synapses.decay_silent_period(dt_ms);
    }

    /// Probe memory: Generic non-specific cue reveals which assemblies were recently active
    pub fn probe_readout(&mut self, probe_indices: &[usize]) -> Vec<f32> {
        self.synapses.transmit_and_update(probe_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activity_silent_working_memory_retention() {
        let n = 100;
        let mut weights = vec![0.0f32; n * n];
        // Diagonal recurrent self-excitatory connections
        for i in 0..n {
            weights[i * n + i] = 1.0;
        }

        let mut memory = ActivitySilentMemory::new(n, weights);

        // Word SDR arrives at t = 0 ms (indices 10, 11, 12 active as gamma burst of 2 spikes)
        let token_sdr = Sdr {
            total_bits: n,
            active_indices: vec![10, 11, 12],
        };
        memory.encode(&token_sdr);
        memory.encode(&token_sdr); // gamma burst

        // Verify neurons 10, 11, 12 are now facilitated (calcium elevated)
        assert!(memory.synapses.is_facilitated(10));
        assert!(memory.synapses.is_facilitated(11));
        assert!(memory.synapses.is_facilitated(12));
        assert!(!memory.synapses.is_facilitated(0)); // untouched neuron

        // Silent period of 400 ms (zero spikes fired!)
        memory.elapse_time(400.0);

        // Neurons should STILL be facilitated after 400 ms of silence (tau_F = 1500 ms)
        assert!(
            memory.synapses.is_facilitated(10),
            "Memory trace must persist in activity-silent synaptic calcium"
        );

        // Readout with generic non-specific probe across all neurons
        let probe = (0..n).collect::<Vec<_>>();
        let readout_current = memory.probe_readout(&probe);

        let baseline_response = readout_current[0];
        let memory_response = readout_current[10];

        println!(
            "Readout: Baseline = {:.4}, Facilitated Memory = {:.4} (ratio: {:.2}x)",
            baseline_response, memory_response, memory_response / baseline_response
        );

        assert!(
            memory_response > baseline_response * 1.25,
            "Activity-silent memory trace must produce significantly amplified readout"
        );
    }
}
