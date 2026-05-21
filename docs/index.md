# Intelligence Within Fixed Capacity

## Continual Learning Via Mechanistic Interpretability And Usage-Driven Memory Management

[Repository](https://github.com/nelson960/Preserving-Circuits) · [Living Research Log](./living-paper/)

Most deployed language models are still updated through discrete training or fine-tuning cycles rather than through safe, continuous, internal learning. They are pretrained on a large snapshot of data, adapted through instruction tuning, and later aligned through preference optimization. Real deployments are not static. Domains change, user needs change, production failures appear, safety requirements evolve, and new knowledge must be incorporated without destroying old capabilities.

Continual learning is the research area that tries to solve this problem. The central failure mode is catastrophic forgetting: after a model learns new information or a new task, performance on earlier knowledge or behavior can degrade sharply.

This proposal argues that catastrophic forgetting should be studied as a fixed-capacity memory-management problem grounded in mechanistic interpretability.

```text
new data -> optimizer update -> representation movement -> circuit drift -> behavior change
```

The goal is to understand what changes inside a model when it forgets, then test whether learning can be controlled by measuring and constraining the movement of latent representations and circuits. The governing principle is usage-driven capacity ownership: capacity should be protected in proportion to how much a representation or circuit is actively used, while disused capacity should be allowed to fade, compress, and become available for new learning.

## Abstract

Current approaches to continual learning often grow effective capacity, replay old data, isolate parameters, or apply heuristic weight protection. These methods help, but they do not fully describe how a fixed-capacity system should decide what to protect, what to update, what to route, and what to let fade. This proposal develops a mechanistic direction for fixed-capacity continual learning: track feature geometry, circuit roles, readout alignment, routing structure, and usage history; route new information into reusable computation where possible; align related internal representations before rewriting shared weights; release low-usage capacity; and compress repeated co-activations into abstract structure. The research aim is not exact retention of every task, but controlled retention, abstraction, and reusable computation inside a fixed representational budget.

## Living Research Log

This page is the public research proposal. The detailed living research log records the path of hypotheses, failed interventions, partial results, and the current mechanistic direction:

- [A Living Failure Map Toward Mechanistic Continual Learning](./living-paper/)

The living log is intentionally written as a research notebook rather than a finished claim. It tracks how the project moved from usage-driven neuron protection, to route and family-level interventions, to closed latent composition, to direct SAE-based feature drift and causal-use measurements in a small pretrained model.

## 1. Research Thesis

The core thesis is:

```text
Forgetting is not only a performance drop.
Forgetting is internal damage to representations, routes, writes, readouts, or gates.
```

When a model learns a new task, the optimizer updates shared parameters. Those updates can move latent concept representations, alter attention routes, change MLP transformations, break readout alignment, or suppress old circuits. A single old-task accuracy number hides these distinct mechanisms.

The proposed research will build a mechanistic account of forgetting by tracking both:

- activation-level representations: where concepts live and how they move;
- weight-level circuits: which parameter paths route, transform, write, and read those concepts.

The long-term intervention is a route-align-consolidate learning system: new information should first find or learn a route into existing reusable computation, then align with related internal representations, and only then be considered for slower shared-weight consolidation.

The governing principle is:

```text
capacity is owned by whoever uses it
```

Usage is measured by interaction history: whether a feature fires, whether it propagates downstream, and whether it causally contributes to behavior. High-usage circuits should be protected from overwrite or reused through aligned routes. Low-usage circuits should be available for adaptation. Persistent disuse should allow fading and capacity reclamation. This is not a separate memory trick added after training; it is the criterion that helps decide how plasticity is routed, aligned, and consolidated.

## 2. What Continual Learning Work Has Established

### 2.1 Continual Learning Has Multiple LLM Stages

LLM continual learning is not one procedure. Surveys describe several stages where forgetting can occur:

- continual pre-training, where the model absorbs new raw or domain-specific text;
- continual instruction or fine-tuning, where the model adapts to new downstream behaviors;
- continual alignment, where preference or reward feedback changes the model's response policy.

Wu et al. categorize LLM continual learning into continual pretraining, instruction tuning, and alignment, and emphasize that LLMs need updates because full retraining is too expensive and knowledge changes over time. Shi et al. describe vertical continual learning, from general to specific capabilities, and horizontal continual learning, across time and domains. See [Wu et al., 2024](https://arxiv.org/abs/2402.01364) and [Shi et al., 2024](https://arxiv.org/abs/2404.16789).

This matters because forgetting can look different at each stage:

- pre-training updates may disturb broad world or domain knowledge;
- fine-tuning may damage task behavior or instruction-following;
- alignment may steer, suppress, or amplify existing capabilities rather than deeply teach new ones.

### 2.2 Classic CL Methods Protect Weights, Masks, Or Subspaces

Learning without Forgetting uses new-task data while distilling old-task outputs, preserving earlier capabilities without requiring access to the old training data. See [Li and Hoiem, 2016](https://arxiv.org/abs/1606.09282).

Elastic Weight Consolidation protects weights important to previous tasks by slowing learning on those parameters. Kirkpatrick et al. frame this as selective protection of important weights during sequential task learning. See [Kirkpatrick et al., 2017](https://arxiv.org/abs/1612.00796).

Hard Attention to the Task learns task masks to preserve earlier task information while allowing current-task learning. See [Serra et al., 2018](https://proceedings.mlr.press/v80/serra18a.html).

Gradient Episodic Memory stores examples from previous tasks and constrains new gradients so losses on remembered tasks do not increase. See [Lopez-Paz and Ranzato, 2017](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning).

Average Gradient Episodic Memory makes GEM more efficient by replacing many per-task constraints with a single averaged memory-gradient constraint. See [Chaudhry et al., 2019](https://openreview.net/forum?id=Hkf2_sC5FX).

Gradient Projection Memory stores activation-derived subspaces from old tasks using SVD, then projects later gradients away from directions important to previous tasks. See [Saha et al., 2021](https://arxiv.org/abs/2103.09762).

Orthogonal Gradient Descent projects new-task gradients onto a subspace that preserves outputs on previous examples. See [Farajtabar et al., 2020](https://proceedings.mlr.press/v108/farajtabar20a.html).

PackNet uses pruning and parameter allocation to reserve parts of a network for different tasks, avoiding forgetting through parameter isolation. See [Mallya and Lazebnik, 2018](https://arxiv.org/abs/1711.05769).

These methods establish the right direction: forgetting is linked to parameter sharing and subspace interference. But they do not fully explain what old circuit or representation was damaged by a given update.

### 2.3 Newer Subspace Methods Are Moving Toward The Same Problem

Recent LLM continual-learning methods increasingly treat forgetting as a subspace problem.

O-LoRA learns tasks in different orthogonal low-rank vector subspaces to reduce interference. See [Wang et al., 2023](https://arxiv.org/abs/2310.14152).

Sculpting Subspaces uses adaptive SVD during full fine-tuning to identify task-specific low-rank parameter subspaces and constrain new updates to be orthogonal to prior critical directions. See [Nayak et al., 2025](https://arxiv.org/abs/2504.07097).

Naive LoRA summation studies when independently trained LoRA modules can be combined by addition, showing that approximate orthogonality helps and interference appears as domains overlap. See [Cao et al., 2025](https://arxiv.org/abs/2508.11985).

HDSD for vision-language continual learning explicitly decomposes parameter space into general and task-specific subspaces, using SVD-based structure to reduce subspace interference and parameter drift. See [Qin et al., 2026](https://arxiv.org/abs/2605.07512).

This line of work strongly supports the premise that old and new learning interact through geometry. The remaining question is more mechanistic:

```text
which old latent representation or circuit is moved by each update?
```

### 2.4 Mechanistic Work Shows Forgetting Has Internal Forms

Recent mechanistic work starts to explain forgetting below the benchmark level.

Masip et al. describe forgetting as transformations of feature encodings. Features can fade, rotate, overlap with other features, or become misaligned with downstream readouts. They argue that performance-only or last-layer analysis misses these mechanisms. See [Masip et al., 2026](https://arxiv.org/abs/2601.22012).

Laitinen Imanov analyzes catastrophic forgetting during LLM continual fine-tuning and identifies gradient interference in attention weights, representational drift in intermediate layers, and loss landscape flattening as mechanisms of forgetting. The paper reports attention-head disruption and strong correlation between forgetting severity and gradient/task-similarity measures. See [Laitinen Imanov, 2026](https://arxiv.org/abs/2601.18699).

This supports the direction of this proposal: forgetting should be decomposed into internal causes, not only reported as an external accuracy drop.

### 2.5 Architecture And Memory Work Point Toward Routing And Consolidation

Several newer approaches imply that continual learning needs better routing and memory structure.

Nested Learning reframes models as nested optimization systems and interprets optimizers such as SGD with momentum and Adam as memory-like modules that compress gradient information. See [Behrouz et al., 2025](https://arxiv.org/abs/2512.24695).

MeSH identifies bottlenecks in recursive transformers: undifferentiated computation and overload in a single hidden state. It externalizes state into a memory buffer and uses routers to diversify computation. See [Yu et al., 2025](https://arxiv.org/abs/2510.07739).

Prototype-based models show another direction: using explicit representational slots to make decision behavior more interpretable. ProtoTEx, for example, uses prototype tensors as latent clusters for explanation. See [Das et al., 2022](https://arxiv.org/abs/2204.05426).

Production-oriented continual improvement systems also show that adaptation is not just training. Pioneer Agent frames SLM improvement as a closed loop of failure diagnosis, data curation, retraining, and regression avoidance. See [Atreja et al., 2026](https://arxiv.org/abs/2604.09791).

Recent online memory work also points to fixed-capacity update rules. δ-mem augments a frozen attention backbone with a compact associative memory state updated by a delta rule, writing residual information into a fixed-size state instead of growing the model. See [Lei et al., 2026](https://arxiv.org/abs/2605.12357).

The common theme is:

```text
continual learning needs routing, memory, diagnostics, and rollback,
not just another fine-tuning run
```

## 3. The Gap This Project Targets

Existing work gives strong partial answers:

- EWC protects parameters.
- Learning without Forgetting preserves old outputs through distillation.
- HAT protects task-specific masks.
- GEM and A-GEM constrain gradients using episodic memory.
- GPM protects old activation subspaces.
- OGD constrains updates to preserve old outputs.
- PackNet allocates parameters across tasks.
- O-LoRA and OSFT protect orthogonal parameter subspaces.
- LoRA and adapters isolate some updates.
- MoE and memory systems route computation.
- agentic systems manage data, failures, and regressions.

But these approaches still leave one central question under-specified:

```text
What latent representation and circuit does a candidate update move?
```

They also leave a governing question under-specified:

```text
Once we know what an update will move, how do we decide whether that circuit deserves protection?
```

This proposal answers with usage-driven capacity ownership. Old circuits should not be protected merely because they are old or because they belong to a named task. They should be protected in proportion to their current interaction history. A circuit that has not been used for a long time should contribute little protection cost. A circuit that fires often, propagates strongly, or causally supports behavior should be expensive to disturb.

This also addresses the null-space problem in subspace-projection methods. If every old activation subspace is permanently protected, the available null space shrinks with each task until there is nowhere left to write. Fixed-capacity learning cannot rely only on indefinite protection. It also needs compression: repeated or related memories must merge into abstractions, and low-usage details must fade so capacity can be reclaimed.

The proposed research targets this gap directly. It aims to connect:

```text
parameter update
-> hidden-state movement
-> concept subspace drift
-> circuit damage or preservation
-> behavioral forgetting or retention
```

This is the difference between saying:

```text
Task A accuracy dropped by 12 percent
```

and saying:

```text
Task A dropped because the old concept subspace remained decodable,
but the readout rotated away from it after the Task B update.
```

That second diagnosis is mechanistic. It suggests a different fix than if the representation were erased or if the attention route had drifted.

## 4. Mechanistic Taxonomy Of Forgetting

The first contribution of the research is a taxonomy of forgetting mechanisms.

### 4.1 Representation Erased

The old concept representation collapses or fades. Its activation norm or feature strength drops enough that the concept is no longer recoverable.

Diagnostic signs:

- lower linear probe accuracy;
- lower feature/subspace norm;
- lower allocated capacity;
- old concept no longer recoverable from intermediate layers.

Possible fix:

- protect the old concept subspace;
- replay old hidden states;
- penalize representation drift.

### 4.2 Readout Broken

The old information is still present, but downstream layers no longer read it correctly.

Diagnostic signs:

- probe accuracy remains high;
- task accuracy drops;
- readout or unembedding alignment changes;
- patching old activations rescues behavior.

Possible fix:

- preserve readout alignment;
- adapt readout without moving the representation;
- add a readout-stability penalty.

### 4.3 Route Drifted

The model still has the relevant information somewhere, but attention or routing no longer moves it to the right destination.

Diagnostic signs:

- QK routing matrices change;
- attention patterns shift away from old causal tokens or features;
- OV writes still carry useful information but are no longer triggered correctly.

Possible fix:

- preserve old QK geometry;
- constrain route-changing blocks;
- separate routing updates from value/write updates.

### 4.4 Write Path Damaged

The model still routes to the right place, but the information written into the residual stream changes.

Diagnostic signs:

- attention locations stay similar;
- OV or MLP write vectors drift;
- residual-stream value code changes;
- behavior changes despite similar routing.

Possible fix:

- preserve OV/write subspace;
- distill residual value codes;
- project new gradients away from old write directions.

### 4.5 Collision Or Overwrite

The new task uses parameter directions or latent subspaces that overlap destructively with old knowledge.

Diagnostic signs:

- high gradient overlap between old and new concepts;
- high old-concept drift after a new update;
- old and new subspaces have small principal angles;
- performance tradeoff appears immediately after one-step updates.

Possible fix:

- orthogonal projection;
- write gating;
- adapter or low-rank route;
- delayed consolidation.

### 4.6 Gated Off

The old circuit survives, but normalization, masking, routing, or gating suppresses it.

Diagnostic signs:

- old circuit can be rescued by patching;
- weights are similar but activations are reduced;
- layer norm or gate statistics shift;
- old capability appears under some prompts but not others.

Possible fix:

- preserve activation scale;
- constrain gating changes;
- add route/gate diagnostics.

### 4.7 Reuse

The new task uses an old circuit productively.

Diagnostic signs:

- old-task performance stays stable or improves;
- new-task gradients align with old-circuit directions;
- old concept subspace participates in the new behavior without destructive drift.

Possible fix:

- encourage compositional routing;
- consolidate shared subspaces;
- avoid unnecessary isolation when reuse is beneficial.

## 5. Circuit Survival Ledger

The second contribution is a circuit survival ledger: a structured record of old computation while new training occurs.

For an old circuit `A`, track:

```text
C_QK_A(t)       attention routing geometry
C_write_A(t)    value/write geometry
value_code_A(t) residual-stream information written by the circuit
readout_A(t)    output or unembedding alignment
causal_A(t)     patching, knockout, or intervention effect
u_A(t)          usage score or running interaction history
merge_A(t)      whether this circuit compressed into an abstract parent
accuracy_A(t)   external task behavior
```

The ledger asks:

```text
Did the old circuit die?
Did it drift?
Did it move to another route?
Did it survive but stop being used?
Did the new task reuse it?
```

This ledger makes the proposal concrete. It converts forgetting from a scalar metric into a time-indexed mechanistic object.

Usage makes the ledger actionable. A drifted circuit with near-zero usage may be acceptable capacity reclamation. A drifted circuit with high usage is a real forgetting risk. A merge event is not automatically failure either: it may indicate that specific memories were compressed into a more abstract parent circuit.

### Role Preservation Versus Address Preservation

A key distinction is role preservation versus address preservation. In trained networks, the same computational role may move between heads, MLPs, layers, or subspaces across seeds or training stages. Therefore, this project will not treat forgetting as "head L2H1 changed." It will ask whether the old causal role still exists somewhere, whether it still contributes to the old behavior, and whether the model has migrated, reused, or abandoned that role.

## 6. Latent Concept Geometry

The activation-level side of the research maps where concepts live across layers and checkpoints.

For a concept `c`, layer `l`, and checkpoint `t`, collect an activation matrix:

```text
H_c,l,t in R^(n x d)
```

where `n` is the number of examples and `d` is hidden dimension.

The first analysis computes:

- SVD/PCA to estimate dominant concept subspaces;
- CKA to compare representations across checkpoints;
- linear probes to measure concept decodability;
- principal angles to compare concept subspaces;
- activation drift norms after updates.

CKA is important because it compares representation similarity more robustly than raw vector alignment and can identify correspondences between representations across trained networks. See [Kornblith et al., 2019](https://arxiv.org/abs/1905.00414).

The output is a concept map:

```text
concept -> layer -> checkpoint -> subspace -> drift -> behavior
```

This allows the research to ask:

- where does a concept become decodable?
- does it fade during new training?
- does it rotate into another concept?
- does it remain present after behavior fails?
- which layers are stable and which are plastic?

## 7. Usage-Driven Capacity Ownership

The proposal needs a task-label-free criterion for deciding which representations deserve protection. The criterion is interaction history.

For a feature or circuit `i`, define an interaction score:

```text
interaction(i, t) = activation strength * downstream influence
```

In a simple feature basis this can be approximated as:

```text
interaction(i, t) = |f_i(x_t)| * sum_j |w_ij| |f_j(x_t)|
```

The usage score is a running average:

```text
u_i(t) = lambda * u_i(t-1) + (1 - lambda) * interaction(i, t)
```

High usage means the representation is active and behaviorally connected, so updates that disturb it should pay a high protection cost. Low usage means the representation is weakly involved or idle, so its capacity can be reused more freely. Persistent low usage should allow controlled fading, not accidental forgetting.

At larger scale, usage should be tracked at the feature-family level, not only per feature. A family is a cluster of features that are geometrically related, co-activate, and feed similar downstream circuits. Family-level usage gives a continuous protection landscape without task labels:

```text
high usage family -> protect
medium usage family -> read carefully
low usage but related family -> possible write target
no matching family -> novel allocation or fast-state write
```

### Decodability Versus Causality

A linear probe can show that information is present in an activation, but it does not prove that the model uses that information. Therefore, every representation-level result should be paired with causal tests where possible: activation patching, subspace patching, ablation, knockout, or readout intervention. The project will distinguish three cases: information absent, information decodable but unused, and information causally used.

## 8. Update-To-Representation Bridge

The mathematical center of the proposal is:

```text
Delta h_l(x) ~= J_h_l,theta(x) Delta theta
```

In words:

```text
a parameter update matters because of the hidden-state movement it causes
```

This connects parameter space to representation space.

For a candidate update on parameter block `b`, the research estimates:

```text
new_gain_b = movement of the new concept in the desired direction
old_drift_b = movement of protected old concept subspaces
circuit_damage_b = movement of old routing/write/readout measurements
```

Full hidden-state Jacobians are too expensive for realistic models, so the practical plan uses:

- Jacobian-vector products to measure update-induced activation movement;
- vector-Jacobian products to identify parameter directions responsible for a latent direction;
- gradient overlap between old and new concepts;
- NTK-style overlap as a function-space proxy;
- blockwise approximations over heads, MLPs, adapters, layers, and readouts.

The Neural Tangent Kernel literature motivates viewing training dynamics in function space rather than only parameter space. See [Jacot et al., 2018](https://arxiv.org/abs/1806.07572).

## 9. Proposed Approach: Route, Align, Consolidate

The proposed intervention is not immediate rewriting of shared computation. The approach is:

```text
route -> align -> consolidate
```

Normal optimization asks:

```text
which parameter update reduces the new loss?
```

This research asks:

```text
can the new information first be routed into existing reusable computation,
can that route align with related latent geometry,
and only then should shared weights be changed?
```

### 9.1 Route

New information should first be connected to existing computation where possible. A new route may be an attention path, an adapter, a task router, a subspace direction, or another parameterized path into a shared circuit. The route is allowed to be plastic before the shared computation is rewritten.

The question is:

```text
is the model missing the computation,
or is it missing a route into computation it already has?
```

This matters because many failures may come from entangled routing rather than absent knowledge. If a model already has a useful circuit, the first objective should be to reach and reuse it, not overwrite it.

### 9.2 Align

A learned route is not enough. It should use the shared computation in a compatible internal format. The alignment step asks whether the new route produces latent states that match the geometry of related old routes, circuits, or feature families.

Alignment can be measured with:

- paired activation similarity;
- CKA between route-induced representations;
- class- or concept-conditioned subspace similarity;
- causal tests showing that the aligned representation is actually used.

The goal is not superficial similarity. The goal is role-compatible reuse:

```text
new route -> same reusable computation -> preserved old role and useful new behavior
```

### 9.3 Consolidate

Only after routing and alignment are established should shared weights be considered for slower consolidation. Consolidation is the stage where new usage may be absorbed into shared computation, but only if mechanistic invariants suggest existing roles are preserved.

Candidate invariants include:

- old route and readout behavior remain causally intact;
- aligned representations stay close enough to their related family geometry;
- high-usage circuits do not suffer large latent drift;
- new learning does not only succeed by suppressing old computation;
- the update improves reusable structure rather than memorizing a narrow case.

Latent-Tangent Write Attention is one possible consolidation mechanism. It treats learning as attention over write locations:

```text
which parameter block should receive this learning signal,
given what it will do to latent concepts and causal roles?
```

For a candidate shared update on parameter block `b`, a consolidation score can be written as:

```text
score_b =
  reusable_gain_b
  - sum_i u_i * old_concept_drift_b(i)
  - sum_i u_i * circuit_damage_b(i)
  - interference_b
  - update_cost_b
```

Then the update can be gated, projected, delayed, or rejected:

```text
Delta theta_b = -eta * M_b * Pi_b * g_b
```

where:

- `g_b` is the candidate gradient for block `b`;
- `M_b` is a soft consolidation gate;
- `Pi_b` is a projection that removes damaging directions when available;
- `eta` is the learning rate.

The proposal does not assume that a gate is always necessary. The gate is a candidate mechanism for the consolidation stage, not the whole theory. The deeper research question is:

```text
when is aligned routing sufficient,
and when does shared consolidation need gating, projection, compression, or delay?
```

The compression clause is essential for fixed capacity. When the available low-interference subspace becomes too small, the system should search for co-activating, similar, low-usage circuits and merge them into an abstract parent representation. New learning can then write into freed capacity rather than colliding indefinitely with protected subspaces.

## 10. Fast State, Routing, And Slow Consolidation

The full system should distinguish update frequency, not permanent memory tiers. New information can first enter a fixed-capacity fast state, similar in spirit to delta-rule online memory. A residual write stores what is new, while a forget gate trades old and new content within the same fixed-size state.

The next step is routing. If the new information is related to existing computation, the model should learn or identify a route into that computation before modifying the shared circuit itself. This keeps plasticity local while the system tests whether reuse is possible.

Slow shared weights should change only after evidence accumulates:

- the new information is repeated or useful enough to matter;
- a route into existing computation has been found or a new family is justified;
- the route is representationally compatible with related old computation;
- high-usage roles are preserved under causal and geometric checks;
- compression or abstraction can release capacity when needed.

This is not "temporary memory versus permanent memory" as separate kinds of knowledge. It is one usage landscape operating across timescales. Fast state absorbs novelty. Routes connect novelty to reusable computation. Slow weights consolidate patterns only when they are repeated, useful, aligned, and safe to write into the current feature-family geometry.

## 11. Research Plan

The research program starts with controlled models where activations, weights, and checkpoints can be inspected directly, then moves toward larger models and more realistic continual-learning settings. The goal is to understand how information moves from embeddings through latent geometry, routing, shared computation, readouts, and optimizer updates.

To make that research possible, I am building a Neural Representation Atlas for inspecting neurons, learned features, weight operators, bilinear attention interactions, and causal circuits in trained models. My goal is to contribute to AI safety by building tools and research workflows that make model internals more observable, testable, and eventually more controllable.

The planned direction is:

- map how concepts and computational roles are represented across layers;
- distinguish route failure from representation failure and readout failure;
- study whether new information can be routed into existing reusable computation;
- measure when related routes share latent geometry and when they diverge;
- develop consolidation rules for deciding when shared weights should change;
- study capacity boundaries where routing is not enough and abstraction or compression is required.

The public aim is not to claim a finished continual-learning algorithm. The aim is to build a mechanistic path toward models that can reuse, align, and consolidate knowledge without blindly overwriting the circuits that already support useful behavior.

## 12. Risks And Mitigations

The main risks are that concept subspaces may rotate across checkpoints, concepts may be superposed rather than cleanly separable, full Jacobian measurements may be too expensive, and consolidation rules may over-protect old knowledge at the cost of new learning. The proposal handles these risks by using representation-similarity methods such as CKA, treating concepts as subspaces rather than single axes, relying on JVP/VJP and blockwise approximations instead of full Jacobians, and evaluating stability and plasticity together rather than optimizing preservation alone.

## 13. Expected Contributions

This research aims to produce seven contributions.

### 13.1 A Mechanistic Forgetting Taxonomy

A clear vocabulary for distinguishing representation erasure, readout failure, route drift, overwrite collision, gating suppression, and reuse.

### 13.2 A Circuit Survival Ledger

A time-indexed method for tracking whether old circuits survive during new learning.

### 13.3 A Representation Drift Measurement

A test of whether update-induced movement of old concept subspaces predicts forgetting better than parameter distance or accuracy alone.

### 13.4 Update Attribution

A method for assigning old-circuit damage to QK, OV, MLP, normalization, adapter, or readout groups.

### 13.5 A Route-Align-Consolidate Learning Prototype

A first implementation of a learning system that routes new information into reusable computation, aligns related latent representations, and tests when shared weights should consolidate through gating, projection, delay, or compression.

### 13.6 A Usage-Driven Write Criterion

A task-label-free, interaction-history-based criterion for deciding which circuits deserve protection during any update. This replaces the assumption that all old-task circuits are equally worth preserving.

### 13.7 A Mechanistic Evaluation Benchmark

An evaluation suite that measures representation drift, capacity degradation, fading, readout misalignment, circuit survival, and usage-score evolution alongside accuracy. Standard benchmarks cannot distinguish irrecoverable capacity loss from recoverable readout misalignment.

## 14. What Would Falsify This Direction?

This research direction would be weakened if representation drift does not predict forgetting better than simpler measures such as parameter distance, gradient norm, or old-task loss; if usage history does not predict which circuits deserve protection; if probe-measured concept drift is mostly non-causal; if old behavior fails even when latent geometry and circuit ledgers remain stable; if compression destroys useful specifics without producing better abstraction; or if consolidation rules reduce forgetting only by preventing new learning. These outcomes would suggest that the proposed latent-tangent and usage-driven measurements are incomplete or that the intervention is over-constraining plasticity.

## 15. Research Milestones

First milestone:

```text
build reliable measurements of representation drift, route drift,
readout misalignment, circuit usage, and causal role preservation
```

Second milestone:

```text
use those measurements to distinguish representation failure,
route failure, readout failure, overwrite, gating suppression, and reuse
```

Third milestone:

```text
develop route-first and alignment-first learning procedures
that connect new information to existing reusable computation
```

Long-term milestone:

```text
move toward continual learning as controlled routing, alignment,
capacity management, and consolidation,
not uncontrolled global fine-tuning
```

## 16. Public Abstract

Continual learning usually measures forgetting after it happens. This research tries to explain forgetting while it happens. The central hypothesis is that catastrophic forgetting is caused by new parameter updates moving old latent representations, altering old circuit routes, damaging value writes, breaking readout alignment, or suppressing surviving circuits. The project will track concepts across layers and checkpoints, connect parameter updates to hidden-state movement with Jacobian-vector analysis, and attribute forgetting to specific circuit components. The long-term goal is a route-align-consolidate learning framework: new information should first be routed into reusable computation, aligned with related latent geometry, and only then considered for shared-weight consolidation under usage-driven capacity constraints.

## References

- [Continual Learning for Large Language Models: A Survey](https://arxiv.org/abs/2402.01364)
- [Continual Learning of Large Language Models: A Comprehensive Survey](https://arxiv.org/abs/2404.16789)
- [Learning without Forgetting](https://arxiv.org/abs/1606.09282)
- [Overcoming Catastrophic Forgetting in Neural Networks](https://arxiv.org/abs/1612.00796)
- [Overcoming Catastrophic Forgetting with Hard Attention to the Task](https://proceedings.mlr.press/v80/serra18a.html)
- [Gradient Episodic Memory for Continual Learning](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning)
- [Efficient Lifelong Learning with A-GEM](https://openreview.net/forum?id=Hkf2_sC5FX)
- [Gradient Projection Memory for Continual Learning](https://arxiv.org/abs/2103.09762)
- [Orthogonal Gradient Descent for Continual Learning](https://proceedings.mlr.press/v108/farajtabar20a.html)
- [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://arxiv.org/abs/1711.05769)
- [Orthogonal Subspace Learning for Language Model Continual Learning](https://arxiv.org/abs/2310.14152)
- [Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning](https://arxiv.org/abs/2504.07097)
- [Efficient Modular Learning through Naive LoRA Summation](https://arxiv.org/abs/2508.11985)
- [Hierarchical Dual-Subspace Decoupling for Continual Learning in Vision-Language Models](https://arxiv.org/abs/2605.07512)
- [$δ$-mem: Efficient Online Memory for Large Language Models](https://arxiv.org/abs/2605.12357)
- [Putting a Face to Forgetting: Continual Learning meets Mechanistic Interpretability](https://arxiv.org/abs/2601.22012)
- [Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning](https://arxiv.org/abs/2601.18699)
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/)
- [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414)
- [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572)
- [Nested Learning: The Illusion of Deep Learning Architectures](https://arxiv.org/abs/2512.24695)
- [MeSH: Memory-as-State-Highways for Recursive Transformers](https://arxiv.org/abs/2510.07739)
- [ProtoTEx: Explaining Model Decisions with Prototype Tensors](https://arxiv.org/abs/2204.05426)
- [Pioneer Agent: Continual Improvement of Small Language Models in Production](https://arxiv.org/abs/2604.09791)
- [Memory Bounds for Continual Learning](https://arxiv.org/abs/2204.10830)
- [The Organization of Behavior](https://www.britannica.com/topic/The-Organization-of-Behavior)
