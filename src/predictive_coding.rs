//! Pillar 4: Hierarchical Predictive Coding without Backpropagation
//!
//! # Scientific Foundation
//! - Primary Paper: Rao, R. P., & Ballard, D. H. (1999).
//!   "Predictive coding in the visual cortex: a functional interpretation of some extra-classical
//!   receptive-field effects." Nature Neuroscience, 2(1), 79-87.
//! - Theoretical Review: Millidge, B., Seth, A., & Buckley, C. L. (2021).
//!   "Predictive coding: a theoretical and experimental review." arXiv:2107.12979.
//! - Temporal Hierarchy: Ding, N., Melloni, L., Zhang, H., Tian, X., & Poeppel, D. (2016).
//!   "Cortical tracking of hierarchical linguistic structures in connected speech."
//!   Nature Neuroscience, 19(1), 158-164.
//!
//! # Biological Principles
//! 1. Brain does **not** perform end-to-end backpropagation through time.
//! 2. Cortex is arranged in a temporal and abstraction hierarchy:
//!    - Level 0: Fast token/sensory stream (Gamma: ~4-8 Hz).
//!    - Level 1: Syntactic phrase assemblies (Theta: ~2-3 Hz).
//!    - Level 2: Narrative context & discourse goals (Delta: ~0.5-1 Hz).
//! 3. **Dual Streams**:
//!    - **Top-Down Prior**: Deep pyramidal neurons (Layer 5/6) project generative predictions $\mu_l$.
//!    - **Bottom-Up Error**: Superficial pyramidal neurons (Layer 2/3) compute and project only prediction errors:
//!      $$\epsilon_l = r_l - \mu_l$$
//! 4. **Strictly Local Plasticity**:
//!    Weight updates require only information physically present at the synapse:
//!    $$\Delta W_l = \alpha \cdot \epsilon_l \cdot (r_{l+1})^T$$

/// A single Hierarchical Predictive Coding layer.
///
/// Implements equations from Rao & Ballard (1999) and Millidge et al. (2021).
#[derive(Debug, Clone)]
pub struct PredictiveLayer {
    /// Layer index in hierarchy (0 = lowest sensory level)
    pub level: usize,
    /// Dimensionality of representation in this layer
    pub dim: usize,
    /// Current internal state / representation ($r_l$)
    pub state: Vec<f32>,
    /// Prediction error computed at this level ($\epsilon_l = r_l - \mu_l$)
    pub error: Vec<f32>,
}

impl PredictiveLayer {
    pub fn new(level: usize, dim: usize) -> Self {
        Self {
            level,
            dim,
            state: vec![0.0f32; dim],
            error: vec![0.0f32; dim],
        }
    }

    /// Reset states for new sequence
    pub fn reset(&mut self) {
        for s in self.state.iter_mut() {
            *s = 0.0;
        }
        for e in self.error.iter_mut() {
            *e = 0.0;
        }
    }
}

/// Synaptic projection between two adjacent hierarchical predictive coding layers.
/// Generates top-down predictions ($L_{l+1} \to L_l$) and passes bottom-up errors ($L_l \to L_{l+1}$).
#[derive(Debug, Clone)]
pub struct PredictiveProjection {
    pub lower_dim: usize,
    pub upper_dim: usize,
    /// Synaptic generative weights: $W \in \mathbb{R}^{\text{lower\_dim} \times \text{upper\_dim}}$
    pub weights: Vec<f32>,
    /// Learning rate for local Hebbian error minimization ($\alpha$)
    pub learning_rate: f32,
}

impl PredictiveProjection {
    /// Initialize generative synaptic weights
    pub fn new(lower_dim: usize, upper_dim: usize, learning_rate: f32, initial_scale: f32) -> Self {
        let total = lower_dim * upper_dim;
        let weights = vec![initial_scale; total];
        Self {
            lower_dim,
            upper_dim,
            weights,
            learning_rate,
        }
    }

    /// Compute top-down prior prediction:
    /// $\mu_l = W \cdot r_{l+1}$
    ///
    /// Reference: Rao & Ballard (1999), Equation 1.
    pub fn generate_top_down_prediction(&self, upper_state: &[f32]) -> Vec<f32> {
        debug_assert_eq!(upper_state.len(), self.upper_dim);
        let mut prediction = vec![0.0f32; self.lower_dim];

        for i in 0..self.lower_dim {
            let row_offset = i * self.upper_dim;
            let mut sum = 0.0f32;
            for j in 0..self.upper_dim {
                sum += self.weights[row_offset + j] * upper_state[j];
            }
            prediction[i] = sum;
        }

        prediction
    }

    /// Update synapses using strictly local prediction error learning rule:
    /// $\Delta W_{ij} = \alpha \cdot \epsilon_{l, i} \cdot r_{l+1, j}$
    ///
    /// Notice: No backpropagation across layers. Only local pre- and postsynaptic values.
    /// Reference: Millidge et al. (2021), Section "Learning in Predictive Coding", Equation 8.
    pub fn update_weights_locally(&mut self, lower_error: &[f32], upper_state: &[f32]) {
        debug_assert_eq!(lower_error.len(), self.lower_dim);
        debug_assert_eq!(upper_state.len(), self.upper_dim);

        for i in 0..self.lower_dim {
            let row_offset = i * self.upper_dim;
            let err_i = lower_error[i];
            if err_i.abs() > 1e-6 {
                for j in 0..self.upper_dim {
                    let delta = self.learning_rate * err_i * upper_state[j];
                    self.weights[row_offset + j] += delta;
                }
            }
        }
    }
}

/// A 2-Tier Hierarchical Predictive Coding Network (Sensory Level $\leftrightarrow$ Context Level)
#[derive(Debug)]
pub struct HierarchicalPredictor {
    pub sensory_layer: PredictiveLayer,
    pub context_layer: PredictiveLayer,
    pub projection: PredictiveProjection,
}

impl HierarchicalPredictor {
    /// Initialize hierarchy
    pub fn new(sensory_dim: usize, context_dim: usize, learning_rate: f32) -> Self {
        Self {
            sensory_layer: PredictiveLayer::new(0, sensory_dim),
            context_layer: PredictiveLayer::new(1, context_dim),
            projection: PredictiveProjection::new(sensory_dim, context_dim, learning_rate, 0.1),
        }
    }

    /// Step the predictive hierarchy:
    /// 1. Top-down: Context predicts sensory state ($\mu_0 = W \cdot r_1$)
    /// 2. Error computation: $\epsilon_0 = r_{\text{sensory}} - \mu_0$
    /// 3. Bottom-up: Error drives context update ($\Delta r_1 \propto W^T \epsilon_0$)
    /// 4. Synaptic plasticity: $\Delta W = \alpha \cdot \epsilon_0 \cdot (r_1)^T$
    ///
    /// Reference: Rao & Ballard (1999), Equations 1-4.
    pub fn step(&mut self, sensory_input: &[f32]) -> f32 {
        assert_eq!(sensory_input.len(), self.sensory_layer.dim);

        // Step 1: Top-down prediction from context
        let prediction = self
            .projection
            .generate_top_down_prediction(&self.context_layer.state);

        // Step 2: Compute local sensory prediction error
        let mut total_squared_error = 0.0f32;
        for i in 0..self.sensory_layer.dim {
            let err = sensory_input[i] - prediction[i];
            self.sensory_layer.error[i] = err;
            total_squared_error += err * err;
        }

        // Step 3: Bottom-up error update to context representation (inference step)
        for j in 0..self.context_layer.dim {
            let mut feedback_drive = 0.0f32;
            for i in 0..self.sensory_layer.dim {
                let w = self.projection.weights[i * self.context_layer.dim + j];
                feedback_drive += w * self.sensory_layer.error[i];
            }
            // Context integrates prediction error (leaky integration)
            self.context_layer.state[j] = (self.context_layer.state[j] * 0.9 + feedback_drive * 0.1).clamp(-2.0, 2.0);
        }

        // Step 4: Synaptic update via purely local plasticity (learning step)
        self.projection
            .update_weights_locally(&self.sensory_layer.error, &self.context_layer.state);

        total_squared_error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictive_coding_error_minimization() {
        // Sensory dimension: 8, Context dimension: 4
        let mut predictor = HierarchicalPredictor::new(8, 4, 0.05);

        // Repeating sensory stimulus pattern
        let stimulus = vec![1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0];

        // First step: model has no prior prediction -> High error (surprise)
        let initial_error = predictor.step(&stimulus);

        // Present stimulus repeatedly across multiple cognitive ticks
        let mut final_error = initial_error;
        for _ in 0..50 {
            final_error = predictor.step(&stimulus);
        }

        println!(
            "Predictive Coding: Initial Error = {:.4}, Final Error after 50 ticks = {:.4}",
            initial_error, final_error
        );

        // Rao & Ballard theorem: Prediction error must monotonically decrease as generative model adapts
        assert!(
            final_error < initial_error * 0.25,
            "Local predictive coding must minimize prediction error without backpropagation"
        );
    }
}
