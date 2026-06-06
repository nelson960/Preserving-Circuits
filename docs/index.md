---
layout: default
title: Preserving Circuits
---

## Continual Learning As A Geometry And Write Problem

A working note on forgetting, routing, rewiring, representational rebasing, and
capacity inside neural networks.

> **Status.** This is not a final document. The framing, mechanisms, and
> terminology may change as experiments fail, new evidence appears, or better
> continual-learning research becomes available.

Read this page as five steps:

1. The problem: new learning changes old model geometry.
2. What existing continual-learning work has already established.
3. A mechanistic view of how forgetting breaks internally.
4. A read/reuse/rebase/write mechanism for changing the model without blind overwrite.
5. What can still break this direction.

Most deployed language models are still updated through discrete training or
fine-tuning cycles. Real use is not static: domains change, users reveal new
needs, failures appear, safety requirements evolve, and new knowledge must be
added without damaging old capabilities.

The usual continual-learning question is:

```text
How do we learn new data without forgetting old data?
```

This page makes that question more specific:

```text
When new data arrives, what internal geometry should move,
and what behavior should remain invariant?
```

Forgetting is treated as failed controlled change. A new update can overwrite a
route, rotate a representation, break a readout, change a gate, or consume
capacity that old behavior still depends on. But the opposite failure also
matters: if the old geometry is protected too rigidly, the model may be unable
to perform the coordinated representational rebasing needed to learn new data.
The useful question is not only whether the model got the next batch right. It
is what geometry the update moved, what behavior survived, what computation was
reused, what was damaged, and what capacity was consumed.

The governing principle is:

```text
capacity is owned by usage
```

Capacity should not be protected merely because it is old. It should be
protected when it is still active, stable, and behaviorally useful. Capacity
that is dormant, redundant, or obsolete should be compressed, rewired, decayed,
or reused.

## Abstract

This document treats continual learning as controlled representational change
inside a fixed-capacity neural network. New data should not simply push every
shared weight in the direction of the current loss. But the solution is also
not to freeze old activations forever. Small controlled experiments now suggest
that successful extra-data learning can require coordinated representational
rebasing: old behavior remains usable while old internal coordinates move.

The model should first reason over its own internal state: what route is
active, what existing computation can be reused, what old geometry is still
load-bearing, what capacity is free, what behavior must be preserved, and where
a new write or rebasing operation would cause interference.

The aim is a neural network that can change itself from inside the learning
loop. A new pattern may be routed into an existing path, bridged to a related
path, written into a newly opened direction, or blocked from damaging protected
geometry. Dormant structure should gradually lose ownership so that capacity
can be reused. Repeated useful structure should consolidate into slower shared
weights. In this view, stopping catastrophic forgetting is not only a matter of
preserving old outputs or protecting old coordinates. It is a matter of
learning how to place, isolate, merge, rewire, compress, decay, and sometimes
coherently rebase internal representations while preserving behavior.

A small same-spec transformer experiment that motivated this revision is
written separately as [Continual Learning Is A Geometry Problem]({{ site.baseurl }}/continual-learning-geometry/).
It compares three training paths: a model trained on 100 words, the same model
continually updated on the next 100 words, and the same architecture trained
from scratch on all 200 words. The key observation is that the successful
200-word model moved the old 100-word residual states substantially while
keeping behavior good. That weakens a purely "protect exact old activations"
view and strengthens the need for behavior-preserving rebasing.

## 1. The Core Claim

The thesis is:

```text
continual learning is behavior-preserving representational change
under fixed capacity
```

When a model learns new data, the update is not just a movement in parameter
space. It is a change to the model's internal geometry. Sometimes that change
is a local write into selected structure. Sometimes it is a route adjustment.
Sometimes it may need to be a coordinated rebasing of the representation. The
change is useful when it preserves old behavior while making new behavior
possible. It causes forgetting when it damages geometry that old behavior still
depends on, or when uncontrolled movement breaks readouts, routes, or feature
relations.

The thesis has four parts:

- **Forgetting is internal damage.** Accuracy drops are symptoms. The internal
  causes can be representation drift, route drift, write collision, readout
  misalignment, gate suppression, topology collapse, or capacity exhaustion.
- **Protection alone is insufficient.** If every old direction is protected
  forever, fixed-capacity learning runs out of room. A real learner must also
  merge, compress, rewire, decay, reuse, and allow coherent rebasing when old
  behavior can survive the coordinate change.
- **Usage determines ownership.** Stable, active, behaviorally useful geometry
  should be expensive to disturb. Dormant or redundant geometry should become
  cheap to change.
- **The update policy should be read from geometry.** New data should first be
  read through the current residual stream and route state, then changed by a
  mechanism that decides whether to reuse, split, bridge, create, protect,
  rebase, compress, or decay.

The page therefore follows two levels at the same time:

- activation-level geometry: where concepts and roles live, how they move, and
  whether they remain decodable;
- weight-level and topology-level geometry: which parameter paths route,
  transform, write, gate, and read those concepts.

The long-term mechanism is a route-reuse-rebase-write-consolidate learner. New
information should first search for reusable computation, then align or bridge
related internal structure, then decide whether the correct move is a local
write, a route change, or a broader basis change, and finally consolidate
repeated useful structure into slower shared weights.

This is not meant to be an external memory system wrapped around a frozen
model. The intended endpoint is a model-native architecture in which the
residual stream, MLP writes, topology masks, routing gates, fast traces,
compression gates, and forget gates all participate in the continual-learning
dynamics.

### A Simple Walkthrough

Imagine the model receives a new fact, relation, or skill. Ordinary training
turns that into a gradient and moves shared weights. The mechanistic view asks
the model to do more internal work before committing the update.

```text
new input arrives
-> read the current residual stream and active routes
-> ask whether an existing computation can solve part of it
-> reuse that computation if the role is compatible
-> bridge the route if it is close but not aligned
-> split or isolate the route if it conflicts with old behavior
-> create a new path if nothing reusable exists
-> rebase related geometry if a local write is too constrained
-> write only into the selected structure
-> consolidate if the pattern repeats and remains useful
-> decay or compress old structure when usage disappears
```

The important point is that the model should not treat every new error as a
license to overwrite the same shared directions. It should first decide whether
the new pattern is a reuse problem, a routing problem, a new-capacity problem,
an interference problem, or a representation-rebasing problem.

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

This supports the direction here: forgetting should be decomposed into internal causes, not only reported as an external accuracy drop.

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

### 2.6 What This Work Adds So Far

The previous section is about the outside field. This page adds a narrower
working claim: the missing object is the internal geometry changed by the
update. It is not enough to know that old and new tasks interfere. We need to
know what representation, route, readout, or capacity relationship a candidate
update will move; whether that structure is still useful; whether old behavior
can survive the movement; and whether the model can route around, reuse, or
rebase the structure instead of blindly overwriting it.

The current evidence, kept in the [living log]({{ site.baseurl }}/living-paper/)
and in the small [geometry-rebasing case study]({{ site.baseurl }}/continual-learning-geometry/),
supports a few limited points:

- forgetting can be decomposed into route drift, write drift, readout drift,
  gating failure, and overwrite collision;
- protecting individual weights or neurons is too local to solve the problem;
- preserving exact old activations can also be too rigid, because successful
  same-capacity learning of more data can move old residual states substantially
  while preserving behavior;
- geometric route reasoning can decide when to reuse, split, or protect a path
  in controlled settings;
- learned route construction can grow a gate for a novel key while staying
  silent on protected old keys in synthetic route space;
- model-native fast writing and online projection can be made mechanically
  active, but they do not yet solve full heldout retention.

So the evidence does not yet prove a finished continual learner. It narrows the
engineering target: build a model that can read its own state, choose whether a
local write is enough, reuse existing computation when possible, rewire when
necessary, allow coordinated rebasing when capacity pressure requires it, and
decay unused structure without destroying useful old behavior.

## 3. The Gap: We Still Do Not Know What Geometry The Update Changed

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
What latent representation, circuit, route, readout, or basis
does a candidate update move?
```

They also leave a governing question under-specified:

```text
Once we know what an update will move, how do we decide whether that circuit deserves protection?
```

The working answer here is usage-driven capacity ownership. Old circuits should not be protected merely because they are old or because they belong to a named task. They should be protected in proportion to their current interaction history. A circuit that has not been used for a long time should contribute little protection cost. A circuit that fires often, propagates strongly, or causally supports behavior should be expensive to disturb.

This also addresses the null-space problem in subspace-projection methods. If every old activation subspace is permanently protected, the available null space shrinks with each task until there is nowhere left to write. Fixed-capacity learning cannot rely only on indefinite protection. It also needs compression and rebasing: repeated or related memories must merge into abstractions, old coordinates may need to move together, and low-usage details must fade so capacity can be reclaimed.

The useful measurement chain is:

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

That second diagnosis is mechanistic. It suggests a different fix than if the
representation were erased or if the attention route had drifted. It also leaves
room for the case where a large representation movement is not failure: if the
old behavior and functional relations survive, the movement may be useful
rebasing rather than destructive forgetting.

## 4. How Forgetting Breaks Internally

A useful first move is to stop treating forgetting as one thing. In a neural
network, old behavior can fail in several different ways, and each one suggests
a different repair.

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

A circuit survival ledger is a structured record of old computation while new training occurs.

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

This ledger makes the idea concrete. It converts forgetting from a scalar metric into a time-indexed mechanistic object.

Usage makes the ledger actionable. A drifted circuit with near-zero usage may be acceptable capacity reclamation. A drifted circuit with high usage is a real forgetting risk. A merge event is not automatically failure either: it may indicate that specific memories were compressed into a more abstract parent circuit.

### Role Preservation Versus Address Preservation

A key distinction is role preservation versus address preservation. In trained networks, the same computational role may move between heads, MLPs, layers, or subspaces across seeds or training stages. Therefore, this page does not treat forgetting as "head L2H1 changed." The question is whether the old causal role still exists somewhere, whether it still contributes to the old behavior, and whether the model has migrated, reused, or abandoned that role.

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

The system needs a task-label-free criterion for deciding which representations deserve protection. The criterion is interaction history.

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

A linear probe can show that information is present in an activation, but it does not prove that the model uses that information. Therefore, every representation-level result should be paired with causal tests where possible: activation patching, subspace patching, ablation, knockout, or readout intervention. The important distinction is between three cases: information absent, information decodable but unused, and information causally used.

## 8. Update-To-Representation Bridge

The mathematical center of the idea is:

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

## 9. How I Am Aiming To Solve It: Read, Route, Reuse, Rebase, Write, Consolidate

The intervention is not immediate rewriting of shared computation. The model
should first inspect what is already happening inside itself, then decide what
kind of change is needed. Sometimes the right change is a local write. Sometimes
it is a route adjustment. Sometimes the old and new data require a coordinated
representational rebasing where old internal coordinates move while old
behavior remains usable.

The rough loop is:

```text
read internal state
-> route or reuse existing computation
-> rebase related geometry if local writing is too constrained
-> write into selected structure
-> protect what is still load-bearing
-> consolidate repeated useful structure
-> decay or compress unused structure
```

Normal optimization asks:

```text
which parameter update reduces the new loss?
```

The write question is:

```text
what computation is already active,
what part of it can be reused,
where would a new write collide,
does the current representation need to rebase,
and what should change inside the network?
```

This is the mechanical difference. The model is not only minimizing loss. It is
also deciding the form of the update:

```text
new input + residual stream + current routes + usage history
-> reuse / bridge / split / create / rebase / protect / decay
-> targeted state change
```

The target is therefore not "hold every old residual vector fixed." The target
is to preserve useful behavior, useful relations, and load-bearing routes while
allowing internal coordinates to move when that movement is coherent and
behavior-preserving.

### 9.1 Read The Internal State

Before writing, the model needs a local picture of its own computation. The
useful signals are not labels like "old task" and "new task." They are internal
signals:

- current residual-stream state;
- which routes and MLP paths are active;
- which old circuits are being touched by the new gradient;
- whether the active path is familiar, novel, conflicting, or unused;
- whether the new signal matches an existing feature family;
- whether there is free capacity nearby;
- whether a dormant route can be reused or decayed.

In compact form:

```text
observer_t =
  f(
    h_t,
    routes_t,
    gradient_t,
    usage_t,
    protected_geometry_t,
    free_capacity_t
  )
```

The observer does not need symbolic labels. It only needs enough geometry to
answer: does this input belong to an existing computation, does it need a new
route, will writing here damage something still useful, or is the current
representation itself too constrained and in need of rebasing?

### 9.2 Route And Reuse Computation

The first choice should be reuse. If the model already has a useful computation,
the new input should learn a route into it rather than rewriting the computation
itself. A new route may be an attention path, an MLP path, a sparse topology
edge, a slot route, a low-rank operator, or another parameterized path into a
shared circuit.

The route question is:

```text
is the model missing the computation,
or is it missing a route into computation it already has?
```

Reuse is not just similarity. It must be role-compatible. The reused path should
preserve what the old circuit does while letting the new input benefit from it.
That can be checked with:

- paired activation similarity;
- CKA between route-induced representations;
- class- or concept-conditioned subspace similarity;
- causal tests showing that the aligned representation is actually used.

The desired pattern is:

```text
new input
-> new or adjusted route
-> same reusable computation
-> preserved old role
-> useful new behavior
```

If the route is close but not compatible, the model should bridge it. If the
route collides with an old role, it should split. If there is no related
computation, it should create a new path. If multiple related routes are being
pulled into a new arrangement while behavior can remain stable, the model may
need to rebase the shared representation rather than protect all old coordinates
as fixed addresses.

### 9.3 Rebase The Representation When Needed

A controlled rebasing operation is different from a local write. A local write
tries to place new information into selected structure while leaving most of the
old geometry nearly fixed. A rebasing operation allows a family of related
states to move together so that the model can create a better shared coordinate
system.

The case study on [continual-learning geometry]({{ site.baseurl }}/continual-learning-geometry/)
shows why this matters. The same tiny transformer trained from scratch on 200
words learned both the old and new spans well, but its old 100-word residual
states moved substantially relative to the 100-word-only model:

```text
final-layer effective rank: 74.32 -> 87.87
relative matched drift: 0.7880
CKA: 0.4906
```

By contrast, a safe continual update barely moved the old geometry:

```text
relative matched drift: 0.0779
CKA: 0.9930
new span remained mostly unlearned
```

An aggressive continual update moved more, but still not enough or coherently
enough:

```text
relative matched drift: 0.1348
old margin weakened
new span remained poorly learned
```

This suggests that useful continual learning may need a controlled operation
between exact preservation and blind fine-tuning:

```text
old and new states move together
old behavior remains decodable and causally usable
readouts/routes adapt with the moved representation
capacity rank or basis expands where needed
```

The open problem is detecting when drift is destructive and when drift is
useful rebasing.

### 9.4 Write Into Selected Structure

Once routing has identified where the signal belongs, the write should be local
and selective. The update can target several kinds of internal state:

```text
Delta state =
  Delta weights
  + Delta topology
  + Delta route gates
  + Delta fast trace
  + Delta basis/operator state
  + Delta protection/free-capacity map
```

A candidate write should be scored by both benefit and damage:

```text
write_score =
  new_gain
  + reuse_gain
  + free_capacity
  - protected_drift
  - route_conflict
  - readout_damage
  - capacity_cost
```

Then the update can be accepted, projected, delayed, redirected, or blocked:

```text
Delta theta_b = -eta * M_b * Pi_b * g_b
```

where:

- `g_b` is the candidate gradient for block `b`;
- `M_b` is a soft write gate;
- `Pi_b` removes damaging directions when a protected geometry is at risk;
- `eta` is the learning rate.

This is where rewiring matters. A fixed dense weight matrix tries to store too
many things in the same directions. A topology state can make a different kind
of change: grow a useful edge, isolate a conflicting edge, prune a dormant edge,
or redirect a route into a reusable operator. The weight value says how strongly
a path fires; the topology says whether that path should exist as a learning
route at all.

### 9.5 Consolidate, Compress, And Decay

Only after routing and alignment are established should shared weights be considered for slower consolidation. Consolidation is the stage where new usage may be absorbed into shared computation, but only if mechanistic invariants suggest existing roles are preserved.

Candidate invariants include:

- old route and readout behavior remain causally intact;
- aligned representations stay close enough to their related family geometry;
- old behavior survives if the family geometry moves coherently;
- high-usage circuits do not suffer large latent drift;
- new learning does not only succeed by suppressing old computation;
- the update improves reusable structure rather than memorizing a narrow case.

Compression and decay are part of the same mechanism, not cleanup after the
fact. A fixed-capacity model cannot protect every old direction forever.
Dormant structure should gradually lose ownership:

```text
usage falls
-> protection weakens
-> similar low-usage paths can merge
-> dead paths can prune or become free capacity
```

Decay should be gradual. A route should not disappear because it was quiet for
one batch. It should decay when it has low usage, low causal contribution, weak
recent activation, and a stronger replacement or abstraction exists.

A simple form is:

```text
keep_pressure =
  usage
  + causal_contribution
  + recent_activation
  + replacement_absence

decay_pressure =
  disuse
  + redundancy
  + capacity_pressure
  - keep_pressure
```

The desired result is not infinite storage. It is controlled replacement:

```text
specific memory -> reusable abstraction when repeated
dormant path -> freed capacity when unused
conflicting path -> isolated or rerouted
useful old path -> protected from destructive writes
```

This does not assume that a gate is always necessary. The gate is a candidate mechanism for the consolidation stage, not the whole theory. The deeper question is:

```text
when is aligned routing sufficient,
and when does shared consolidation need gating, projection, compression, or delay?
```

## 10. Fast State, Routing, And Slow Consolidation

The full system should distinguish update frequency, not permanent memory tiers. New information can first enter a fixed-capacity fast state, similar in spirit to delta-rule online memory. A residual write stores what is new, while a forget gate trades old and new content within the same fixed-size state.

The next step is routing. If the new information is related to existing computation, the model should learn or identify a route into that computation before modifying the shared circuit itself. This keeps plasticity local while the system tests whether reuse is possible.

Slow shared weights should change only after evidence accumulates:

- the new information is repeated or useful enough to matter;
- a route into existing computation has been found or a new family is justified;
- the route is representationally compatible with related old computation;
- high-usage roles are preserved under causal and geometric checks;
- compression or abstraction can release capacity when needed.

This is not "temporary memory versus permanent memory" as separate kinds of knowledge. It is one usage landscape operating across timescales. Fast state absorbs novelty. Routes connect novelty to reusable computation. Slow weights consolidate patterns only when they are repeated, useful, aligned, and safe to absorb into the current feature-family geometry. When the current geometry is too constrained, the harder question is whether related representations can rebase together without breaking old behavior.

## 11. How I Am Working On It

The work starts with controlled models where activations, weights, and checkpoints can be inspected directly, then moves toward larger models and more realistic continual-learning settings. The goal is to understand how information moves from embeddings through latent geometry, routing, shared computation, readouts, and optimizer updates.

To make that possible, I am building a Neural Representation Atlas for inspecting neurons, learned features, weight operators, bilinear attention interactions, and causal circuits in trained models. The point is to make model internals more observable, testable, and eventually more controllable.

The planned direction is:

- map how concepts and computational roles are represented across layers;
- distinguish route failure from representation failure and readout failure;
- study whether new information can be routed into existing reusable computation;
- measure when related routes share latent geometry and when they diverge;
- measure when old representations can move coherently without behavioral loss;
- develop consolidation rules for deciding when shared weights should change;
- study capacity boundaries where routing is not enough and rebasing,
  abstraction, or compression is required.

The aim is not to claim a finished continual-learning algorithm. The aim is to build a mechanistic path toward models that can reuse, align, rebase, and consolidate knowledge without blindly overwriting the circuits that already support useful behavior.

## 12. Where This Can Break

The main risks are that concept subspaces may rotate across checkpoints, concepts may be superposed rather than cleanly separable, full Jacobian measurements may be too expensive, and consolidation rules may over-protect old knowledge at the cost of new learning. The new rebasing evidence adds another risk: exact activation preservation may look safe while silently blocking the kind of coordinated movement needed for more data. The practical response is to use representation-similarity methods such as CKA, treat concepts as subspaces rather than single axes, rely on JVP/VJP and blockwise approximations instead of full Jacobians, and evaluate stability, plasticity, and behavior-preserving drift together rather than optimizing preservation alone.

## 13. What Would Change This Direction?

This direction would be weakened if representation drift does not predict forgetting better than simpler measures such as parameter distance, gradient norm, or old-task loss; if usage history does not predict which circuits deserve protection; if probe-measured concept drift is mostly non-causal; if old behavior fails even when latent geometry and circuit ledgers remain stable; if large coordinated drift cannot be distinguished from destructive drift; if compression destroys useful specifics without producing better abstraction; or if consolidation rules reduce forgetting only by preventing new learning. These outcomes would suggest that the latent-tangent, rebasing, and usage-driven measurements are incomplete or that the intervention is over-constraining plasticity.

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
