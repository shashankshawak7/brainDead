//! # BrainDead: Biologically Grounded Language Architecture
//!
//! ## MANDATORY COVENANT
//! **Read `FRANKIE_LAWS.md` before adding or modifying any code.**
//!
//! ### Core Constraints
//! 1. **Code is a liability:** Every line must be justified by peer-reviewed literature.
//! 2. **Extreme Resource Scarcity:** Drink every drop of CPU and RAM like the last drop in the world.
//!    - Zero heap allocations in cognitive loops.
//!    - 256-byte stack-allocated bit-packed SDRs (`BitSdr`).
//!    - Quantized 8-bit synapses (`u8`).
//! 3. **No research = No code:** Every function, struct, and constant must cite equations, authors, and page numbers.
//! 4. **Toy tests are not proof:** Real empirical linguistic validation required.
//! 5. **Do not reinvent the wheel:** Maximize reuse of existing primitives and stdlib.

pub mod sdr;
pub mod assembly;
pub mod synaptic_memory;
pub mod predictive_coding;
pub mod packed_sdr;
pub mod engine;
pub mod hierarchy;
pub mod discourse_hierarchy;
pub mod hippocampus;
pub mod generator;
pub mod binding;
pub mod reasoning;
pub mod syntax;
