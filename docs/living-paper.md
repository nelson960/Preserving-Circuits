---
layout: default
title: A Living Failure Map Toward Mechanistic Continual Learning
permalink: /living-paper/
---

# A Living Failure Map Toward Mechanistic Continual Learning

From usage-driven neuron protection to SAE-based feature drift and causal representation tracking.

This is not a final paper. It is a living research log written to preserve the path of hypotheses, failed interventions, partial successes, equations, and current evidence. The goal is to understand why continual learning fails mechanistically before trying to propose another broad solution.

## Current Thesis

Continual learning is not solved by protecting weights, adding modules, or enforcing latent closure alone. Each intervention exposes a deeper layer of the problem:

- neurons are not the right unit of computation;
- features are distributed across circuits;
- cleanly decodable information is not necessarily causally used;
- preserving coordinates does not necessarily preserve mappings;
- replay-free monolithic weight evolution collapses under sequential pressure;
- orthogonal protection can lock the learner by consuming nearly all useful plasticity.

The current working target is therefore narrower and more mechanistic:

> Preserve causally used feature geometry and readout alignment under sequential training, while measuring when drift becomes capacity collision and behavioral forgetting.

## 1. What Counts As Continual Learning Here?

Standard continual-learning evaluation often asks a narrow question:

```text
after learning Task B, does Task A accuracy stay high?
```

That is necessary, but not sufficient. A model can keep old accuracy high through replay, parameter isolation, task-specific adapters, hard masks, or unbounded module growth. Those methods may avoid forgetting in a benchmark, but they do not prove that the model is learning sequentially inside a reusable fixed-capacity system.

This project therefore uses a stricter standard:

> A continual learner should learn sequentially arriving data, preserve old behavior, reuse compatible internal computations, reject false reuse, compose skills learned at different times, control capacity growth, and expose a time-series mechanistic record of representation drift, causal-use drift, and eventual forgetting.

### 1.1 Sequential Learning

The learner must receive data over time:

```text
D_1 -> update -> D_2 -> update -> D_3 -> update
```

Joint multitask training is useful as a control, but it is not the target. The important question is what happens when each new distribution arrives after the previous one has already shaped the weights and representations.

### 1.2 Retention

Old behavior must not collapse after new learning:

```text
performance(D_old after update) remains high
```

But retention alone is not enough. A system can retain by freezing everything, allocating a new module for every task, or replaying the entire old dataset. Those are useful baselines, not the full goal.

### 1.3 Reusable Computation

When a new task is structurally related to an old task, the learner should reuse old computation instead of building an unrelated mechanism.

Example:

```text
learn ADD01
later learn ADD12
```

A stronger learner should discover that both tasks need addition-like computation. It should route into, align with, extend, or consolidate around the existing addition structure where possible.

This is why the early experiments focused on:

- needed neurons;
- old-important neurons;
- route alignment;
- family-level admission;
- causal reuse tests.

### 1.4 False Reuse Rejection

The learner should also know when not to reuse.

Example:

```text
ADD family should not automatically absorb MAX, COPY, or unrelated tasks
```

If alignment makes geometry look similar but hurts target behavior or increases shared-gradient pressure, that is false reuse.

So the benchmark should ask:

```text
Can the system decide when a new task belongs to an existing computation family,
and when it should branch, allocate, or remain separate?
```

This is why route-family admission must use behavior and pressure, not only representational similarity.

### 1.5 Compositionality

Skills learned at different times should compose.

Example:

```text
learn ADD
later learn MAX
then solve max(add(a,b), c)
```

This is stronger than retention. It tests whether the model has learned reusable operations rather than isolated task memories.

### 1.6 Latent Closure

A reusable computation must output a representation that can be reused as input to another computation.

For an operator:

```text
F(E(a), E(b)) ≈ E(op(a,b))
```

Without closure, a model may produce an output that decodes correctly once but cannot be fed into another operation. This creates external compositionality without internal compositionality.

Closed latent algebra was introduced to test exactly this failure mode.

### 1.7 Capacity Discipline

A continual learner should not simply allocate a full new subsystem forever.

We therefore track:

```text
new parameters per task
operator count
router count
memory or replay size
training steps
shared module drift
```

Modules are not forbidden. But parameter growth must be measured and justified. If the system grows linearly with tasks, it is closer to expansion-based continual learning than fixed-capacity learning.

### 1.8 Time-Series Internal Drift

Forgetting is not usually a single instant. It can unfold as a trajectory:

```text
step0 -> step10 -> step25 -> step50 -> step100
```

At each checkpoint, we want to measure:

- behavior;
- raw hidden-state drift;
- SAE feature drift;
- causal feature use;
- readout alignment;
- capacity overlap;
- patch recovery.

The current working hypothesis is:

```text
representation drift
-> feature fading
-> causal reweighting
-> readout misalignment
-> capacity collision
-> behavioral collapse
```

The Pythia+SAE work was introduced to start measuring this trajectory directly.

### 1.9 Mechanistic Verification

The benchmark should not only report that a model works. It should expose why it works or fails.

Required measurements include:

```text
activation drift
feature rotation
feature fading
feature capacity C_i
causal ablation
patch recovery
readout alignment gamma_i
behavioral forgetting
```

This gives a stricter standard than average task accuracy. It asks whether a method preserves the internal structures that actually support behavior.

### 1.10 Benchmark Levels

The benchmark we are formulating has levels:

```text
Level 1: Retention
Does old behavior survive?

Level 2: Sequential learning
Was the task learned after previous tasks, not jointly?

Level 3: Reuse
Does the new task use old computation when appropriate?

Level 4: False reuse rejection
Does the model avoid forcing unrelated tasks into wrong families?

Level 5: Composition
Can skills learned at different times compose?

Level 6: Latent closure
Can outputs become inputs for later computations?

Level 7: Capacity discipline
Does learning avoid unbounded parameter or replay growth?

Level 8: Mechanistic drift tracking
Do we observe features, circuits, readouts, and causal roles over time?

Level 9: Collapse diagnosis
When forgetting happens, can we say whether it came from fading,
rotation, capacity collision, readout misalignment, routing failure,
or causal reweighting?
```

This standard explains why many earlier experiments that looked successful were still incomplete. Closed latent algebra solved composition, but not replay-free monolithic weight overwrite. Modular gating prevented false reuse, but still grew capacity for new primitives. Fixed-SAE feature tracking exposed drift, but still needs stronger training pressure to reach full collapse.

## 2. Research Goal

The original goal was simple:

> Build a model that can learn sequentially without catastrophic forgetting, while keeping its internal representations observable, testable, and eventually controllable.

This meant studying learning at two levels:

1. **Activation level**: what representations are active for a concept or task?
2. **Weight level**: which parameter updates move, erase, reroute, or misread those representations?

The first intuition was that forgetting might be preventable if the optimizer could avoid updating weights that carry old knowledge. That led to the first family of experiments: identify which neurons or circuits are necessary, and update only what is safe.

## 3. Early Transformer Route Diagnostic

Before the MLP usage-score work, we tested a small transformer-style copy-position task.

The model learned:

```text
Task A: final query should copy position 0
Task B: final query should copy position 1
```

The intended diagnostic was to ask whether the old attention route survived after learning the new route.

We tracked:

- final query attention pattern;
- query-key circuit drift, `C_QK`;
- output-value circuit drift, `C_OV`;
- unembedding/readout drift, `W_U`;
- route input drift from token and position embeddings.

The first interpretation was tempting:

```text
W_Q and W_K changed, so the route changed.
```

But ablations showed the route could move even when `W_Q` and `W_K` were frozen. The model changed route inputs instead:

```text
freeze W_Q, W_K
route still moves by changing W_E and W_P
```

When `W_E`, `W_P`, `W_Q`, and `W_K` were all frozen, Task A was mostly preserved but Task B could not learn.

### Lesson

The causal role was not tied to a single address like "head 0 layer 1". The same route could be destroyed by changing upstream inputs rather than the route weights themselves.

This introduced a principle that stays important:

> We should track role preservation, not address preservation.

An old function may migrate between heads, layers, neurons, subspaces, or input encodings. The question is not only "did this module change?" The question is:

```text
Does the old causal role still exist somewhere,
and does it still contribute to the old behavior?
```

## 4. Attempt One: Only Update What Is Necessary

The first optimizer-level idea was:

```text
new data arrives
find the neurons needed for the new task
avoid neurons important to old tasks
update only the safe subset
```

This required defining "important" and "necessary" without guessing.

For a hidden neuron `i`, we tested several candidate scores:

```text
A_i = mean absolute activation of neuron i
D_i = downstream weight norm of neuron i
E_i = old-task loss increase when neuron i is ablated
```

Then combinations:

```text
A_i
D_i
E_i
A_i E_i
A_i D_i E_i
```

The initial usage-style score was based on the idea that a neuron matters if it both fires and affects downstream computation:

```text
usage_i ≈ activation_i × downstream_influence_i × causal_effect_i
```

This was meant to approximate the phrase:

```text
capacity is owned by whoever uses it
```

### Experiment

We trained a small numerical MLP on base operations:

```text
COPY0, COPY1, ADD01, MAX
```

Then introduced:

```text
ADD12
```

We measured which neurons predicted damage after the new-task update.

The most useful diagnostic was ablation effect:

```text
E_i = max(0, L_ablate_i - L_base)
```

Across multiple seeds, `E_i` and downstream norm `D_i` predicted per-neuron loss attribution and drift better than raw activation alone.

### What Failed

Protection based on these scores only modestly helped. It did not solve forgetting.

The key empirical problem was that the new task needed many of the same neurons the old tasks depended on. Protecting those neurons preserved old behavior but blocked new learning.

This exposed a structural conflict:

```text
old-important neurons ∩ new-needed neurons is large
```

### Lesson

The problem was not merely identifying important neurons.

The deeper problem was:

> new learning often needs the same computational substrate as old learning.

So "only update the safe neurons" is insufficient when the safe set does not contain enough useful plasticity.

## 5. Attempt Two: Surgical Update

The next idea was to make the update rule explicit.

For each neuron:

```text
protect_i = E_old_i > threshold
needed_i  = E_new_i or G_new_i > threshold
```

Then:

```text
update_i = needed_i and not protect_i
```

The gradient mask was:

```text
if protect_i:
    zero gradients into and out of neuron i

if not needed_i:
    zero gradients into and out of neuron i
```

This created four cases:

```text
protected=True,  needed=False  -> skip
protected=False, needed=True   -> update
protected=False, needed=False  -> skip
protected=True,  needed=True   -> conflict
```

### What Failed

The conflict case dominated the hard part.

Surgical masking improved old retention slightly, but the new task could not fully learn because too much of the new gradient was blocked.

### Lesson

Hard masking creates a stability-plasticity deadlock:

```text
protect old computation -> block new learning
allow new learning      -> damage old computation
```

This showed that binary protection is too rigid.

## 6. Attempt Three: Soft Blending

Instead of blocking conflict neurons, we allowed them to update slowly.

The update scale was:

```text
blend_i = E_new_i / (E_new_i + λ E_old_i)
```

Interpretation:

- high new need, low old importance -> update strongly;
- high old importance, low new need -> nearly freeze;
- high old importance, high new need -> damped update, not zero.

### What Worked

Soft blending performed better than hard masking. It traded learning speed for old-task preservation.

### What Failed

It still did not solve the old/new tradeoff. The update was still assigned independently per neuron, but the computations were distributed.

### Lesson

Graded plasticity helps, but the unit of computation is larger than a neuron.

This pushed the research toward circuits and feature families.

## 7. Attempt Four: Meaning Transformation

The next idea was not to protect the old neuron from the new task, but to let the neuron's meaning generalize.

For example:

```text
old: neuron participates in "add digit 0 and digit 1"
new: ADD12 also needs this neuron
goal: transform neuron toward "add adjacent digits"
```

The hypothesis was:

> conflict is not only danger; conflict may identify where abstraction should happen.

The attempted local mechanism:

1. find conflict neurons;
2. collect old activating inputs;
3. collect new activating inputs;
4. find shared subspace;
5. move the neuron's incoming weights toward the shared direction.

In simplified form:

```text
X_shared = concat(X_old_active, X_new_active)
U, S, V = SVD(X_shared)
w_i <- (1 - α) w_i + α Project_shared(w_i)
```

### What Failed

The local transform preserved old behavior, but did not make the new behavior decodable.

The neuron moved safely, but the function did not become ADD12.

### Lesson

A single neuron participating in addition is not itself the addition function.

The computation is distributed. Updating one neuron in isolation is too local.

This pushed the research to family-level circuits.

## 8. Attempt Five: Family-Level Circuits

We tried to identify small functional groups of neurons rather than individual neurons.

The diagnostic used pairwise ablation synergy:

```text
E_i  = loss increase from ablating neuron i
E_j  = loss increase from ablating neuron j
E_ij = loss increase from ablating both i and j

synergy_ij = E_ij - E_i - E_j
```

If:

```text
synergy_ij >> 0
```

then neurons `i` and `j` participate in a shared functional unit.

The family-level idea:

```text
identify ADD01 synergy family
apply coordinated blending to the family
```

The family-level blend used shared family importance:

```text
family_E = max(E_i for i in family)

blend_i = G_new_i / (G_new_i + λ family_E)
```

### What Worked

The ADD01 synergy family was real enough to preserve old behavior under strong protection.

### What Failed

ADD12 learning needed neurons outside the discovered ADD01 family. The family did not contain the whole reusable operation.

The missing pieces included:

- routing/input-position neurons;
- readout neurons;
- distributed support structure outside the obvious ADD01 synergy family.

### Lesson

The old-task circuit was not the full reusable operation. A family found from old-task synergy alone can be too narrow.

This pushed the research to a different diagnosis:

> the model may have entangled operation and routing.

## 9. Entanglement Diagnosis

ADD01 and ADD12 share the same operation:

```text
addition
```

but differ in operand positions:

```text
ADD01 = add positions 0 and 1
ADD12 = add positions 1 and 2
```

The MLP appeared to fuse:

```text
how to add
which positions to add
```

into the same hidden computation.

This meant learning ADD12 required modifying the same weights that encoded ADD01. The model had not learned a position-independent addition operation.

### Position-Flag Test

We added explicit position/routing information to the input to see whether missing input information caused the conflict.

### What Failed

The blocked-gradient fraction did not meaningfully change.

### Lesson

The entanglement was not just missing input metadata. It was inside the learned hidden computation.

This pushed the research toward architectural factorization.

## 10. Attempt Six: Factorization

The factorization hypothesis:

> separate "where to read from" from "what operation to perform."

The model form:

```text
h_t(x) = shared_op(R_t(x))
f_t(x) = W_out h_t(x)
```

where:

```text
R_t       = task-specific route
shared_op = reusable computation module
W_out     = shared readout
```

For ADD12, the new task first learned a route into frozen shared computation:

```text
min_R CE(f_12(x), y)
```

with:

```text
R_01, shared_op, W_out frozen
```

### Alignment

We then forced the new route to reuse analogous old internal geometry:

```text
min_R CE(f_12(x), y) + λ ||h_12(x) - h_01(Px)||^2
```

where `P` maps ADD12 inputs to analogous ADD01 cases.

### What Worked

Route learning and representation alignment avoided forgetting in controlled toy ADD tasks.

The aligned route reused the shared computation more cleanly than an unconstrained route.

### What Failed Or Stayed Open

This was architecture-assisted. It did not solve arbitrary monolithic continual learning.

Also, later ablations showed the shadow commit gate was not necessary in the easy regime. After alignment, mild shared updates were already safe.

### Lesson

Factorization can prevent some forgetting by design, but it changes the architecture. It does not fully solve weight overwrite in an already-trained monolithic model.

It also suggested a stronger mechanism:

> representational alignment may make the new-task gradient more compatible with old computation.

## 11. Route -> Align -> Consolidate

The next formulation was:

```text
1. Route: learn a new route into existing computation
2. Align: force internal representation reuse
3. Consolidate: update shared weights only when safe
```

The consolidation gate treated gradient descent as proposing writes, not automatically committing them:

```text
θ' = θ + Δθ

accept Δθ only if:
  old behavior is preserved
  new behavior is preserved
  alignment is preserved
  shared-gradient pressure does not increase too much
```

### What Happened

The gate was not the mechanism in the easy setting. Naive shared updates after alignment were already safe.

Stress tests showed that the remaining failures were mostly route/alignment capacity failures, not cases where the commit gate saved the model.

### Lesson

The honest claim changed:

```text
Route + alignment can place the model in a safe shared-update basin.
```

Not:

```text
The commit gate is necessary for safe consolidation.
```

This pushed us to evaluate when alignment should be admitted and when it should be rejected.

## 12. Route-Family Admission And False Reuse

The next problem:

> not every new task should be forced into an existing computation family.

For related ADD routes, alignment can help. For non-analog tasks like MAX12 or COPY2, forced alignment can make geometry look similar while hurting behavior or increasing shared-gradient pressure.

We defined route-family admission as a decision problem:

```text
given route R and family F,
should R be admitted into F?
```

A route should not be admitted just because CKA or class-center similarity improves.

Admission must consider:

- target behavior;
- old family behavior;
- shared-gradient pressure;
- causal dependence on the family code;
- false alignment gap.

The false alignment pattern:

```text
geometry similarity improves
but behavior does not improve
and shared-gradient pressure increases
```

### Lesson

This separated two different kinds of reuse:

```text
output-code reuse      = can use the same output representation
computation reuse      = uses the same causal operation
```

Output-code reuse is not enough to claim a shared computation family.

## 13. Composition Failure And Latent Closure

The next benchmark tested whether the shared operation was actually reusable.

The model could solve one-step routes like:

```text
ADD01
ADD12
```

but deeper composition exposed a type mismatch:

```text
SUM012 = ADD(ADD(d0, d1), d2)
```

The model had external compositionality:

```text
hidden -> readout -> correct class
```

but not latent closure:

```text
hidden output of ADD was not a valid input code for another ADD call
```

The requirement became:

> a reusable operation must output the same representation type that it accepts.

For ADD:

```text
F(E(a), E(b)) ≈ E((a+b) mod n)
```

Training objective:

```text
L =
  CE(D(F(E(a), E(b))), (a+b) mod n)
  + λ ||F(E(a), E(b)) - E((a+b) mod n)||^2
```

### What Worked

Closed latent ADD solved multi-step composition.

Closed latent algebra extended this to multiple operators:

```text
ADD, MAX, COPY, MIN, SUB
```

and mixed compositions:

```text
max(add(a,b), c)
add(max(a,b), c)
sub(add(a,b), c)
```

The closure loss was necessary. Without it, intermediate representations drifted out of the valid code space and multi-step composition degraded.

### What This Solved

Closed latent algebra solved:

- representation type mismatch;
- recursive composition;
- zero-shot operator chaining;
- modular skill reuse.

### What It Did Not Solve

It did not solve full fixed-capacity, replay-free continual learning.

It works partly by maintaining a stable code space and adding or reusing operators. That is valuable, but it is not the same as arbitrary monolithic weight evolution without forgetting.

### Lesson

Reusable computation requires type-stable latent representations.

But type stability alone does not prevent destructive updates to the mapping.

## 14. Modular Continual Learning And Admission Gating

We then tested a library of closed latent operators.

Policies:

```text
always_new_operator
always_try_reuse
admission_gated_reuse
```

The gate searched existing operator programs:

```text
if program accuracy >= threshold:
    reuse program
else:
    train new operator
```

### What Worked

Gated reuse prevented false reuse.

For example:

```text
DOUBLE_SHIFT = SHIFT(SHIFT(x))
```

could be solved without adding a new operator.

### What Failed Or Stayed Limited

New primitive skills still required new modules.

So this addressed parameter growth only when tasks were composable from previous primitives. It did not solve fixed-capacity learning in the general case.

### Lesson

Admission gating is useful engineering, but continual learning still needs consolidation, compression, or controlled weight evolution.

## 15. Monolithic Weight Evolution

To address the concern that modular methods avoid the hard problem, we tested a monolithic operator.

Setup:

```text
one MLP
all weights updated sequentially
task token included
latent closure enforced
```

Important design fix:

```text
input = h_a || h_b || W_T[task]
```

not:

```text
h_a + h_b + W_T[task]
```

because summing operands makes non-commutative tasks like `SUB(x,y)` impossible.

### With Exact Replay

The monolithic model worked.

It retained all tasks and composed them zero-shot.

### Without Replay

The model collapsed.

First-task retention fell near chance.

Closure reduced manifold drift but did not preserve the learned mapping.

This distinction was crucial:

```text
closure preserves coordinates
not necessarily the function implemented by weights
```

### Lesson

Exact replay can stabilize monolithic weight evolution, but replay-free continual learning remains unsolved.

This pushed us to gradient projection.

## 16. Gradient Projection Memory And Gradient Locking

We implemented GPM-style orthogonal gradient projection.

The idea:

```text
protect old activation subspace
project new gradients into orthogonal complement
```

### Adam Failure

With Adam, projection did not preserve true orthogonality.

Adam rescales coordinates elementwise:

```text
g_i / (sqrt(v_i) + ε)
```

This changes the direction after projection, leaking updates into protected subspaces.

### SGD Failure

With SGD, orthogonality was mathematically cleaner, but the model hit gradient locking.

The input was:

```text
h_a || h_b || task_emb
```

The operands `h_a` and `h_b` are shared across all tasks and occupy most of the active input subspace. After Task 1, protecting that subspace leaves too little room for Task 2.

### Lesson

Orthogonal protection can consume the available learning space.

The learner becomes safe but unable to adapt:

```text
no forgetting because no meaningful learning
```

This is one form of the null-space problem.

## 17. Pivot To Mechanistic Observation In A Pretrained Model

At this point the research direction changed.

Instead of inventing another optimizer immediately, we decided to observe what actually happens inside a small pretrained language model during conflicting fine-tuning.

Setup:

```text
model: EleutherAI/pythia-70m
site: residual stream after block 4 / hidden_states[5]
SAE: fixed reference SAE with 2048 features
concept: animal
stressor: vehicle-as-animal conflict fine-tuning
checkpoints: step0, step10, step25, step50, step100
```

The goal was to track:

- raw residual geometry;
- SAE feature geometry;
- decodable semantic features;
- causal feature use;
- drift before collapse.

## 18. Full Representation Drift

In the high-dimensional residual and SAE spaces, the animal representation moved but did not collapse.

Animal centroid shift:

```text
Residual stream, 512D:
  shift = 3.1991
  shift / original cluster spread = 0.5779
  cosine(step0, step100) = 0.9746

SAE feature space, 2048D:
  shift = 2.5870
  shift / original cluster spread = 0.5209
  cosine(step0, step100) = 0.9885
```

Animal-vehicle separation stayed stable:

```text
Residual: 5.8262 -> 5.8415
SAE:      5.0675 -> 5.0708
```

### Lesson

This was not full catastrophic erasure.

The concept cluster moved by about half of its radius, but animal/vehicle separation remained intact.

This looked like early drift rather than collapse.

## 19. Decodable Feature Drift

The cleanest decodable animal SAE feature was:

```text
feature 254
AUROC = 1.0 at baseline
```

After fine-tuning:

```text
raw animal direction rotation: ~10.08 degrees
feature selectivity: 0.4114 -> 0.3050
AUROC: 1.0000 -> 0.9954
fading ratio: 1.0000 -> 0.7302
```

### Lesson

A clean semantic feature can remain decodable while fading substantially.

So forgetting is not binary:

```text
feature exists / feature gone
```

There is a middle stage:

```text
feature still readable but weaker, rotated, or less selectively used
```

## 20. Decodability Versus Causality

The next question:

> Does the clean decodable animal feature actually cause animal behavior?

Direct SAE ablation showed:

```text
feature 254 has almost no causal effect
```

So the cleanest probe feature was not the main feature used by the model for the tested animal next-token behavior.

We then ranked features by first-order causal attribution:

```text
z_j * <∇_h log p(y), d_j>
```

where:

```text
z_j = SAE feature activation
d_j = SAE decoder direction
```

Top causal feature:

```text
feature 853
```

Causal effect:

```text
feature 853 ablation delta:
  step0   = -0.0400
  step100 = -0.0148

causal top-5 ablation delta:
  step0   = -0.0721
  step100 = -0.0300
```

### Lesson

Decodability is not causality.

A feature can be semantically clean under a probe and still not be the feature the model uses for behavior.

The causally used feature set became much less behaviorally influential after fine-tuning.

This suggests:

```text
early forgetting may involve causal reweighting before visible concept collapse
```

## 21. Current Working Model Of Forgetting

The current hypothesis is that catastrophic forgetting has stages:

```text
1. representation drift
2. feature fading
3. causal reweighting
4. readout misalignment
5. capacity collision
6. behavioral collapse
```

Our Pythia run appears to show stages 1-3:

- full representation shifted;
- semantic animal feature faded;
- causal animal-supporting features lost behavioral influence;
- concept separation did not collapse yet.

This means stronger pressure is needed to observe later stages.

## 22. Paper-Style Metrics We Need Next

The forgetting paper motivates a more geometric measurement.

### Rotation

```text
θ_i = arccos(
  (φ_i_before · φ_i_after)
  / (||φ_i_before|| ||φ_i_after||)
)
```

We already have a version of this for raw concept directions.

### Fading

```text
fading_i = ||φ_i_after|| / ||φ_i_before||
```

We already measure this through fixed-SAE feature activation changes.

### Capacity

The next missing metric:

```text
C_i = (φ_i^T φ_i)^2 / Σ_j (φ_i^T φ_j)^2
```

This asks how exclusively a feature owns its representational direction.

We do not yet measure this properly, because that requires tracking feature vectors before and after fine-tuning, not just fixed-SAE activations.

Likely tools:

- train SAE before and after, then match features;
- train a crosscoder between checkpoints;
- compute overlap changes among matched features.

### Readout Alignment

Another missing metric:

```text
γ_i = w_readout^T φ_i
```

This measures whether downstream computation still reads the feature correctly.

Our causal ablation is currently a behavioral proxy for this, not the exact readout-alignment metric.

## 23. What Failed And What It Taught Us

| Stage | Hypothesis | Method | What Failed | Lesson |
| :--- | :--- | :--- | :--- | :--- |
| Transformer route tracking | protect old route weights | freeze `W_Q`, `W_K` | route moved through inputs | causal role is not fixed address |
| Usage score | important neurons can be protected | `A`, `D`, `E`, `AE`, `ADE` | new task needed old-important neurons | importance detection is not enough |
| Surgical mask | update only needed and safe neurons | hard gradient mask | blocked too much new learning | conflict set carries useful computation |
| Soft blending | update conflict neurons slowly | continuous gradient scale | tradeoff remained | neuron-level update is too local |
| Meaning transform | conflict neurons should generalize | shared-subspace neuron transform | old safe, new not learned | function is distributed |
| Family blending | update small circuit families | synergy-based family | family incomplete | old synergy family is not full operation |
| Position factorization | add position/routing signal | position flags | conflict unchanged | entanglement is in learned computation |
| Factorized routes | separate route and operation | `h_t = shared_op(R_t(x))` | architecture-assisted | helps but does not solve arbitrary models |
| Consolidation gate | accept only safe shared updates | shadow commit gate | not needed in easy setting | alignment, not gate, was main mechanism |
| Family admission | avoid false reuse | behavior/pressure gate | still modular | useful, but not fixed-capacity |
| Closed latent algebra | enforce reusable type space | closure loss | solves composition, not overwrite | closure preserves type, not mapping |
| Monolithic replay | one network can learn sequentially | exact replay | depends on replay | replay stabilizes but stores old data |
| No-replay monolithic | closure prevents forgetting | sequential training | old task collapsed | closure alone insufficient |
| GPM | orthogonal updates protect old tasks | projected gradients | gradient locking | safe subspace can be too small |
| SAE semantic feature | clean feature explains behavior | feature 254 | decodable but non-causal | probes are not causal evidence |
| Causal SAE ranking | find actually used features | gradient × SAE decoder | effect weakened after fine-tune | causal-use drift is measurable |

## 24. Next Experiments

The next experiments should not add another toy architecture immediately. They should push the current mechanistic observation until it reaches collapse.

### Stronger Training Pressure

Increase:

- fine-tuning steps;
- learning rate;
- number of unfrozen layers;
- conflict dataset size;
- number of sequential contradictory stages.

Example sequence:

```text
baseline animal
-> vehicle-as-animal
-> animal-as-object
-> animal-as-vehicle
-> recovery or replay attempt
```

Measure at every checkpoint:

- old behavior loss;
- raw hidden rotation;
- feature fading;
- causal ablation effect;
- patch recovery;
- concept separation;
- feature capacity if available.

### Better Feature Tracking

The fixed SAE tells us how baseline features activate after fine-tuning. But it does not reveal whether features migrate into new SAE directions.

Next options:

1. train SAE on baseline and final checkpoints, then match features;
2. train crosscoder across checkpoints;
3. track feature families rather than individual features.

### Capacity Measurement

Compute:

```text
C_i = (φ_i^T φ_i)^2 / Σ_j (φ_i^T φ_j)^2
```

before and after sequential training.

This should show whether a feature is merely less active or actually losing representational capacity through overlap/collision.

### Readout And Circuit Measurement

Measure whether downstream layers still use the same features:

- ablation at multiple layers;
- activation patching from baseline to fine-tuned model;
- patch recovery from fine-tuned to baseline;
- readout alignment proxies;
- direct path attribution.

## 25. Current Research Direction

The current direction is no longer:

```text
find a clever mask and prevent forgetting
```

It is:

```text
observe the stages of forgetting mechanistically,
separate decodable drift from causal-use drift,
measure when drift becomes capacity collision,
then design interventions at the correct level.
```

The likely intervention will not protect all old activations. It will need to preserve or reconstruct:

- causally used feature geometry;
- readout alignment;
- family-level capacity;
- reusable operation structure;
- enough plasticity for genuinely new learning.

## 26. Ending State

We have not solved continual learning.

We have mapped several false starts and identified a sharper target:

> continual learning should be studied as preservation of causally used feature geometry and readout alignment under sequential weight updates.

The current Pythia+SAE results show the beginning of the process:

```text
representation drift
semantic feature fading
causal-use reweighting
without full concept collapse yet
```

The next milestone is to push this setup until behavioral forgetting appears, then measure whether capacity degradation and readout misalignment explain the collapse.
