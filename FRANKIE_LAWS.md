# FRANKIE OPERATIONAL LAWS & SCIENTIFIC COVENANT

**MANDATORY READING BEFORE TOUCHING ANY FILE IN BRAINDEAD**

---

## THE CORE CONSTRAINTS

### COMMANDMENT 0: THE ANTI-CHEST-THUMPING MANDATE (ZERO HYPE)
* **Never brag. Never hype. Never claim parity with frontier models on statistical sequence tricks.**
* A model predicting "the cat sat on the mat" is a reflex loop, NOT an artificial general intelligence.
* Any claim of capability must be backed by adversarial stress-testing, logic evaluation, and empirical measurement.
* Maintain cynical, ruthless senior architect perspective at all times: code is liability, claims are liability, arrogance is failure.

### LAW 1: CODE IS A LIABILITY, NOT AN ASSET
* Every line of code written is a future bug, a maintenance cost, and an energy drain.
* Default answer to any new function, struct, or abstraction: **DO NOT BUILD IT**.
* Before writing a line of code, ask:
  1. Can standard library or existing hardware already do this?
  2. Can this be deleted without breaking the core mathematical invariant?
  3. Is this supported by peer-reviewed neuroscience or empirical proof?

### LAW 2: EXTREME RESOURCE SCARCITY (THE DESERT COVENANT)
* **RAM, CPU cycles, cache lines, and disk space are scarce like water in the Sahara.**
* **Drink every drop like the last drop on Earth:**
  1. **Zero heap allocation in cognitive loops:** No `malloc`, no `Vec` reallocation, no string formatting in the hot path. Everything must be stack-allocated or use pre-allocated scratchpads.
  2. **Cache locality is king:** Data structures must fit inside CPU L1 (32 KB) or L2 (512 KB) cache. Spilling to RAM costs 200–300 clock cycles per access.
  3. **Low-precision by default:** No arbitrary 32-bit floats (`f32`) when 8-bit quantized integers (`u8`) or binary bitsets (`u64`) satisfy the biophysics.
  4. **Single-cycle hardware primitives:** Use bitwise operations, hardware `POPCNT`, and trailing zeros instead of branching loops and pointer chasing.

### LAW 3: NO RESEARCH = NO CODE
* **No speculative code allowed.**
* Every module **MUST** cite its foundational scientific literature in the module-level docstring:
  - Authors, Title, Journal/Conference, Year, DOI/ArXiv ID.
* Every struct, algorithm, and function **MUST** cite specific equations, biological mechanisms, and page numbers in inline comments.
* If a mechanism cannot be cited with mathematical or biological proof from literature, **it does not enter this codebase**.

### LAW 4: TOY TESTS ARE NOT PROOF
* Passing a unit test on a 10-element vector is **NOT proof of language capacity or intelligence**.
* Never declare victory or make frontier claims based on synthetic micro-benchmarks.
* Proof requires:
  1. Empirical measurement on real natural language text data.
  2. Information-theoretic quantification (perplexity, bits-per-character, mutual information, error reduction curves).
  3. Robustness against noise, out-of-vocabulary inputs, and scale degradation.

### LAW 5: DO NOT REINVENT THE WHEEL (MAXIMAL REUSE)
* Always reuse existing proven components before inventing new ones.
* Prioritize Rust standard library primitives over custom abstractions.
* Never duplicate data structures across modules. Single source of truth.

---

## THE 10 FATAL ISSUES UNCOVERED IN DNALLM (NEVER REPEAT)

These 10 catastrophic anti-patterns wasted 4 months of engineering. They are permanently banned in BrainDead:

1. **The MurmurHash Catastrophe (Non-Semantic SDRs):**
   - *Mistake:* Using cryptographic or random hashing to generate token SDRs. Words with identical meanings ("doctor", "physician") had 0% overlap, permanently destroying generalization before learning began.
   - *Law:* All SDRs must be generated via locality-sensitive semantic projections (Dasgupta Fly Algorithm, Science 2017) or competitive Hebbian clustering.
2. **The N-Gram Parasite (Hiding the Broken Brain):**
   - *Mistake:* Bolting on an n-gram Markov model when the neural net failed to predict words. This created the illusion of performance while masking complete cortical failure.
   - *Law:* Zero n-gram tables, zero Markov chains. The spiking cortical assemblies must do 100% of the cognitive work.
3. **Exponential Decay Amnesia:**
   - *Mistake:* Storing working memory in a continuous float array with `state *= 0.95`. After 20 tokens, context was 99% erased.
   - *Law:* Working memory must live in silent synaptic facilitation (Mongillo et al., Science 2008) and oscillatory phase attractors.
4. **Cargo Cult Anatomy (Fake Modularity):**
   - *Mistake:* Creating 17 named brain regions ("Broca", "Wernicke", "Striatum") that were identical copy-pasted wrappers with no algorithmic specialization.
   - *Law:* Neocortex is a uniform 6-layer microcircuit (Mountcastle canonical column). Specialization comes from sensory inputs and hierarchical timescale depth, not empty struct names.
5. **Architectural Bloat & Lock-Striped Routing:**
   - *Mistake:* Writing 1,579 lines of channel arbitration and mutex locks to simulate thalamic routing.
   - *Law:* Cortical communication is local dendritic drive and k-WTA lateral inhibition. Zero multi-threaded lock contention in the cognitive hot path.
6. **Catastrophic Forgetting via Unbounded Hebbian LTP:**
   - *Mistake:* Incrementing synaptic weights on every co-occurrence without synaptic homeostatic scaling or anti-Hebbian LTD, causing weights to saturate at maximum and erase past memories.
   - *Law:* Synaptic updates must be error-driven ($\Delta W \propto \epsilon \cdot r^T$) with homeostatic normalization.
7. **Monolithic Timescales (No Hierarchy):**
   - *Mistake:* Processing tokens, sentences, and discourse at the exact same clock tick rate, making long-range context tracking impossible.
   - *Law:* Cortex must operate across hierarchical oscillatory timescales: Gamma (tokens, ~4-8 Hz), Theta (phrases, ~2-3 Hz), Delta (discourse, ~0.5-1 Hz) per Ding et al. (Nature Neuroscience 2016).
8. **Premature Telemetry & GUI Distractions:**
   - *Mistake:* Building WebSockets, real-time visualizers, and REST endpoints while the core intelligence model could not even predict noun-verb agreement.
   - *Law:* Zero networking, zero GUIs, zero dashboards until the core predictive engine demonstrates frontier perplexity on large text.
9. **Heap Allocation Inside Cognitive Inner Loops:**
   - *Mistake:* Allocating `Vec`, cloning arrays, and formatting strings on every token, causing memory bus saturation and cache thrashing.
   - *Law:* Zero heap allocations during `process_word`. Fixed stack `BitSdr` (256 bytes) and pre-allocated scratchpads only.
10. **Dense Floating-Point Extravagance:**
    - *Mistake:* Using `f32` for every synaptic weight, blowing past CPU L1/L2 caches and bottlenecking on RAM bandwidth.
    - *Law:* 8-bit quantized integer synapses (`u8`) or binary bitsets, keeping working memory cache-resident.
