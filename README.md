# BrainDead: Biologically Grounded Language Architecture

Every line of code in this repository is mapped to peer-reviewed neuroscience literature.
Code is liability. If a mechanism cannot be cited with mathematical or biological proof, it does not enter this codebase.

---

## The Biological & Mathematical Pillars

### 1. Semantic Sparsification & Locality-Sensitive Hashing
* **Citations:**
  - Dasgupta, S., Stevens, C. F., & Navlakha, S. (2017). *A neural algorithm for a fundamental computing problem.* Science, 358(6364), 793-796.
* **Mechanism:** Random sparse expansion ($m \gg d$) followed by lateral inhibition (k-Winners-Take-All). Produces binary Sparse Distributed Representations (SDR) with mathematically proven similarity preservation. Consistently generates 2048-bit stack-resident vectors with exact 40-bit sparsity ($k=40$, $\approx 2\%$ active).

### 2. Hierarchical 2-Level Dentate Gyrus LSH Index
* **Citations:**
  - Dasgupta, S., Stevens, C. F., & Navlakha, S. (2017). *Science*, 358(6364), 793-796.
  - Treves, A., & Rolls, E. T. (1994). *A computational analysis of the role of the hippocampus in memory.* Hippocampus, 4(3), 374-391.
* **Architecture:**
  - **Level 1 (Coarse Regional Basin):** 256 Entorhinal-Dentate macro-clusters partitioned via SplitMix64 avalanche hashing.
  - **Level 2 (Fine Orthogonal Mini-column):** 16 CA3 micro-buckets per macro-cluster (4,096 total fine buckets).
  - **Capacity & Speed:** Indexes up to 32,768 candidate references across 4,096+ episodic propositions with an ultra-compact **73.7 KB memory footprint** (fitting entirely within CPU L2 cache). Evaluates queries in **134 nanoseconds** with 100% recall.

### 3. Vector Symbolic Architecture & Canonical Permutation Binding
* **Citations:**
  - Kanerva, P. (2009). *Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors.* Cognitive Computation, 1(2), 139-159.
  - Plate, T. A. (2003). *Holographic Reduced Representations: Distributed representations for cognitive structures.* CSLI Publications.
  - Kleyko, D., et al. (2020). *Vector Symbolic Architectures as a Computing Framework for Nanoscale Neuromorphic Computing.* Proceedings of the IEEE, 110(9), 1538-1571.
* **Mechanism:** Variable-role binding via canonical circular bit rotation:
  $$\rho^k(W) = W.\text{rotate\_left}(k \times 63)$$
  Permutation stride $S=63$ is coprime to 2048 ($\gcd(63, 2048) = 1$), ensuring maximal cycle length without bit aliasing. Preserves 100% exact 40-bit sparsity and exact mathematical self-inverse unbinding: $\rho^{-k}(\rho^k(W)) = W$.

### 4. Autonomous Recursive Constituent Tree Chunking
* **Citations:**
  - Ding, N., Melloni, L., Zhang, H., Tian, X., & Poeppel, D. (2016). *Cortical tracking of hierarchical linguistic structures in normal and jabberwocky speech.* Nature Neuroscience, 19(1), 158-164.
  - Harris, Z. S. (1954). *Distributional structure.* Word, 10(2-3), 146-162.
* **Mechanism:** Streamed text is segmented at unsupervised statistical scaffold boundaries. Contiguous content tokens within phrases autonomously bind into hierarchical binary constituent trees without manual AST construction or keyword dictionaries:
  $$NP = \text{merge}(\text{Modifier}, \text{Noun}) = \text{bundle}(\rho^1(\text{Modifier}), \rho^2(\text{Noun}))$$
  $$VP = \text{merge}(\text{Verb}, NP_{\text{obj}})$$
  $$S = \text{merge}(NP_{\text{subj}}, VP)$$
  Constituent extraction recovers terminal heads and modifiers with **40/40 bits precision**.

### 5. Tier 3 Discourse Temporal Receptive Windows & Unsupervised Anaphora
* **Citations:**
  - Lerner, Y., Honey, C. J., Silbert, L. J., & Hasson, U. (2011). *Topographic mapping of a hierarchy of temporal receptive windows using a narrated story.* Journal of Neuroscience, 31(8), 2906-2915.
  - Hasson, U., Chen, J., & Honey, C. J. (2015). *Hierarchical process memory: memory persists in the service of perception and action.* Trends in Cognitive Sciences, 19(6), 304-313.
* **Mechanism:** Tracks primary discourse agents across multi-sentence paragraphs using nested temporal windows (Gamma clause slots $\rightarrow$ Theta phrase chunks $\rightarrow$ Delta discourse window). Resolves anaphoric and coordinate elliptical subject omissions purely through attractor resonance, with zero hardcoded pronoun lists.

### 6. Complementary Learning Systems, Turrigiano Homeostasis & Distractor Eviction
* **Citations:**
  - McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). *Why there are complementary learning systems in the hippocampus and neocortex.* Psychological Review, 102(3), 419-457.
  - Turrigiano, G. G. (2008). *The self-tuning neuron: synaptic scaling of excitatory synapses.* Cell, 135(3), 422-435.
  - Diekelmann, S., & Born, J. (2010). *The memory function of sleep.* Nature Reviews Neuroscience, 11(2), 114-126.
* **Mechanism:**
  - Episodic propositions track synaptic consolidation salience via zero-cost interior mutability (`Cell<u16>`).
  - Queries and memory resonance apply LTP-like salience boosts (`ACCESS_SALIENCE_BOOST = 32`).
  - At capacity saturation (4,096 stored facts), homeostatic auto-eviction reclaims slots by purging the lowest-salience background distractor from the 2-level LSH index while strictly protecting active discourse referents (`overlap >= 35`).
  - Offline sleep replay cycles reinforce salient propositions and apply multiplicative downscaling ($S_i \leftarrow \lfloor S_i \times \frac{3}{4} \rfloor$).

### 7. Zero-Allocation Surface Sentence Realization
* **Citations:**
  - Levelt, W. J. (1989). *Speaking: From Intention to Articulation.* MIT Press.
  - Roelofs, A. (1992). *A spreading-activation theory of lemma retrieval in speaking.* Cognition, 42(1-3), 107-142.
* **Mechanism:** Decodes bundled episodic propositions and recursive constituent trees back into grammatical natural language clauses (`render_proposition`, `answer_query_surface`) with **0 heap allocations**, writing directly into caller-provided stack byte buffers.


---

## Empirical Benchmark Validation (`unified_empirical_benchmark.rs`)

Tested against the complete King James Bible (`data/bible.txt` - 1.84 MB, 342,618 words) on single CPU thread:

| Metric | Measured Value | Standard / Literature |
| :--- | :--- | :--- |
| **Corpus Streaming Throughput** | **818,906 tokens/sec** | Single-thread Rust stdlib |
| **Scaffold Discovery** | Top 15: `the`, `and`, `of`, `to`, `in`, `that`, `he`... | 100% Unsupervised (Harris 1954) |
| **Discourse Anaphora Precision** | **6/6 queries (40/40 bits)** | 9-sentence cross-topic narrative |
| **Chomskyan Tree Unbinding** | **3/3 queries (40/40 bits)** | Under 50 adversarial distractor noise |
| **LSH 2-Level Lookup Latency** | **134.22 nanoseconds/query** | Sub-linear $O(1)$ associative resonance |
| **Continuous Query Throughput** | **35,615 queries/sec** | 28.08 $\mu\text{s}$ end-to-end QA |
| **Hot-Loop Heap Allocations** | **0 bytes** | Zero `malloc` in cognitive loops |

---

## Commandment 0: Anti-Hype Mandate
BrainDead does not claim parity with 70B parameter web-scraped transformer trivia stores. It contains ~0.05% of the static world facts of frontier LLMs. Its achievement is architectural: **sub-30-microsecond deterministic compositional reasoning with zero backpropagation, zero GPUs, and zero external dependencies.**
