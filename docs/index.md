# Preserving Circuits In Continual Learning

## A Mechanistic Research Proposal

Most deployed language models are still updated through discrete training or fine-tuning cycles rather than through safe, continuous, internal learning. They are pretrained on a large snapshot of data, adapted through instruction tuning, and later aligned through preference optimization. Real deployments are not static. Domains change, user needs change, production failures appear, safety requirements evolve, and new knowledge must be incorporated without destroying old capabilities.

Continual learning is the research area that tries to solve this problem. The central failure mode is catastrophic forgetting: after a model learns new information or a new task, performance on earlier knowledge or behavior can degrade sharply.

This proposal argues that catastrophic forgetting should be studied as a mechanistic write problem.

```text
new data -> optimizer update -> representation movement -> circuit drift -> behavior change
```

The goal is to understand what changes inside a model when it forgets, then test whether learning can be controlled by measuring and constraining the movement of latent representations and circuits.

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

The long-term intervention is a write-controlled optimizer: an update rule that does not only ask which gradient reduces the new loss, but also asks which old representations and circuits the update will disturb.

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

### Decodability Versus Causality

A linear probe can show that information is present in an activation, but it does not prove that the model uses that information. Therefore, every representation-level result should be paired with causal tests where possible: activation patching, subspace patching, ablation, knockout, or readout intervention. The project will distinguish three cases: information absent, information decodable but unused, and information causally used.

## 7. Update-To-Representation Bridge

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

## 8. Proposed Intervention: Latent-Tangent Write Attention

The proposed intervention is an optimizer-side write gate.

Normal optimization asks:

```text
which parameter update reduces the new loss?
```

This research asks:

```text
which parameter update reduces the new loss
while preserving old representations and circuits?
```

For each parameter block `b`, define a write score:

```text
score_b =
  new_concept_gain_b
  - old_concept_drift_b
  - circuit_damage_b
  - interference_b
  - update_cost_b
```

Then use the score to gate or project the update:

```text
Delta theta_b = -eta * M_b * Pi_b * g_b
```

where:

- `g_b` is the usual gradient for block `b`;
- `M_b` is a soft write gate;
- `Pi_b` is a projection that removes damaging directions when available;
- `eta` is the learning rate.

This is called Latent-Tangent Write Attention because it treats learning as attention over write locations:

```text
which parameter block should receive this learning signal,
given what it will do to latent concepts?
```

The proposal does not assume this will immediately beat every continual-learning method. The first research target is more basic and more important:

```text
Can update-induced representation drift predict forgetting?
```

If yes, write-controlled learning becomes a justified next step.

## 9. Experimental Plan

The experimental program will start with controlled models where activations, weights, and checkpoints can be inspected directly. The first goal is to produce a clear forgetting case, then measure whether old concept representations drift, whether old circuits or readouts change, and which parameter groups caused the movement after candidate updates. Once this diagnostic loop is working, the research will test simple write-controlled interventions such as drift penalties, blockwise gates, and projections against ordinary fine-tuning and standard continual-learning baselines. To make that research possible, I am building a Neural Representation Atlas for inspecting neurons, learned features, weight operators, bilinear attention interactions, and causal circuits in trained models. My goal is to contribute to AI safety by building tools and experiments that make model internals more observable, testable, and eventually more controllable.

## 10. Risks And Mitigations

The main risks are that concept subspaces may rotate across checkpoints, concepts may be superposed rather than cleanly separable, full Jacobian measurements may be too expensive, and write gates may over-protect old knowledge at the cost of new learning. The proposal handles these risks by using representation-similarity methods such as CKA, treating concepts as subspaces rather than single axes, relying on JVP/VJP and blockwise approximations instead of full Jacobians, and evaluating stability and plasticity together rather than optimizing preservation alone.

## 11. Expected Contributions

This research aims to produce five contributions.

### 11.1 A Mechanistic Forgetting Taxonomy

A clear vocabulary for distinguishing representation erasure, readout failure, route drift, overwrite collision, gating suppression, and reuse.

### 11.2 A Circuit Survival Ledger

A time-indexed method for tracking whether old circuits survive during new learning.

### 11.3 A Representation Drift Measurement

A test of whether update-induced movement of old concept subspaces predicts forgetting better than parameter distance or accuracy alone.

### 11.4 Update Attribution

A method for assigning old-circuit damage to QK, OV, MLP, normalization, adapter, or readout groups.

### 11.5 A Prototype Write-Controlled Optimizer

A first implementation of Latent-Tangent Write Attention or a simpler blockwise write gate that reduces old-concept drift while preserving new learning.

## 12. What Would Falsify This Direction?

This research direction would be weakened if representation drift does not predict forgetting better than simpler measures such as parameter distance, gradient norm, or old-task loss; if probe-measured concept drift is mostly non-causal; if old behavior fails even when latent geometry and circuit ledgers remain stable; or if write-gated updates reduce forgetting only by preventing new learning. These outcomes would suggest that the proposed latent-tangent measurements are incomplete or that the intervention is over-constraining plasticity.

## 13. What Would Count As Success

Minimum useful result:

```text
representation drift predicts forgetting better than parameter distance alone
```

Strong diagnostic result:

```text
the method can distinguish erased representations,
broken readouts,
route drift,
overwrite,
gating suppression,
and reuse
```

Strong intervention result:

```text
write-controlled updates reduce old-concept drift
while keeping new-task learning competitive with normal fine-tuning
```

Long-term result:

```text
continual learning becomes a controlled write process,
not uncontrolled global fine-tuning
```

## 14. Public Abstract

Continual learning usually measures forgetting after it happens. This research tries to explain forgetting while it happens. The central hypothesis is that catastrophic forgetting is caused by new parameter updates moving old latent representations, altering old circuit routes, damaging value writes, breaking readout alignment, or suppressing surviving circuits. The project will track concepts across layers and checkpoints, connect parameter updates to hidden-state movement with Jacobian-vector analysis, and attribute forgetting to specific circuit components. The long-term goal is Latent-Tangent Write Attention: an optimizer-side mechanism that decides where to write new knowledge by estimating which representations and circuits a candidate update will move.

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
- [Putting a Face to Forgetting: Continual Learning meets Mechanistic Interpretability](https://arxiv.org/abs/2601.22012)
- [Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning](https://arxiv.org/abs/2601.18699)
- [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414)
- [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572)
- [Nested Learning: The Illusion of Deep Learning Architectures](https://arxiv.org/abs/2512.24695)
- [MeSH: Memory-as-State-Highways for Recursive Transformers](https://arxiv.org/abs/2510.07739)
- [ProtoTEx: Explaining Model Decisions with Prototype Tensors](https://arxiv.org/abs/2204.05426)
- [Pioneer Agent: Continual Improvement of Small Language Models in Production](https://arxiv.org/abs/2604.09791)
