---
layout: default
title: A Living Failure Map Toward Mechanistic Continual Learning
permalink: /living-paper/
---

# A Living Failure Map Toward Mechanistic Continual Learning

This page is the compressed living log. It keeps only results that changed the
research direction, plus the core math used in each important experiment. Raw
run output belongs in `model/analysis/*.json`; this page records what we have
actually learned.

## Section Chooser

- [Current Thesis](#current-thesis)
- [Abbreviation Key](#abbreviation-key)
- [Living Log Rule](#living-log-rule)
- [Evidence Ledger](#evidence-ledger)
  - [1. Route Drift In A Minimal Transformer](#1-route-drift-in-a-minimal-transformer)
  - [2. Neuron Importance Predicts Conflict But Does Not Solve It](#2-neuron-importance-predicts-conflict-but-does-not-solve-it)
  - [3. Capacity Reset And Readout-Only Reuse Failed](#3-capacity-reset-and-readout-only-reuse-failed)
  - [4. Controlled Route Reasoning: Reuse Versus Split](#4-controlled-route-reasoning-reuse-versus-split)
  - [5. Real Embedding Relation Reasoning](#5-real-embedding-relation-reasoning)
  - [6. Dynamic Multi-Factor GCO Reasoner](#6-dynamic-multi-factor-gco-reasoner)
  - [7. Living-Map GFO On Real-Book Text](#7-living-map-gfo-on-real-book-text)
  - [8. Native Trace Writing In The Transformer](#8-native-trace-writing-in-the-transformer)
  - [9. Online GCO Projection On Transformer MLPs](#9-online-gco-projection-on-transformer-mlps)
  - [10. Native Scratch GCO: Outcome-Controlled Writes](#10-native-scratch-gco-outcome-controlled-writes)
  - [11. Write-Only Continual Learning Failed The Geometry Test](#11-write-only-continual-learning-failed-the-geometry-test)
  - [12. Reasoner Design Pivot: Nested Learners Are Too Risky For Now](#12-reasoner-design-pivot-nested-learners-are-too-risky-for-now)
- [Current Target After The Geometry Pivot](#current-target-after-the-geometry-pivot)
- [What Is Still Unresolved](#what-is-still-unresolved)

## Current Thesis

Continual learning fails because new learning changes old internal geometry.
Sometimes the change is destructive: routes drift, writes collide, readouts
misalign, or useful capacity is overwritten. But the opposite failure also
matters: protecting exact old activations too rigidly can prevent the model from
performing the coordinated representational rebasing needed to learn more data.

The current thesis is therefore no longer "continual learning is only a write
problem." It is:

```text
continual learning is behavior-preserving representational change
under fixed capacity
```

The current direction is therefore:

```text
experience geometry
+ model geometry
+ behavior geometry
+ time geometry
-> targeted structural transformation or coordinated rebasing
```

The evolved target is not an AdamW wrapper and not merely a protection method.
It is a model-native learner that can decide whether to reuse, write, protect,
split, bridge, rewire, consolidate, decay, or rebase related representations
while preserving old behavior.

## Abbreviation Key

This page mixes old prototype names with the current target architecture. Use
these meanings when reading the evidence ledger:

- **CL** means continual learning: learning from a stream without erasing
  useful behavior learned earlier.
- **GCO** means Geometric Continual Optimizer. This is the current target: a
  model-native learner that reasons about what to reuse, what to write, what
  to protect, what to rewire, what to let decay, and when related geometry must
  rebase instead of remaining fixed.
- **GFO** means Geometric Forgetting Optimizer. In this log it refers to the
  earlier activation-anchor and living-map prototype lineage that tested
  geometric protection before the architecture moved toward full GCO.
- **Living map** means the earlier external concept/anchor controller used to
  test semantic retention. It produced useful evidence, but it is not the final
  requirement because the final learner should be model-native.
- **AdamW** is the ordinary baseline optimizer. It updates parameters from the
  current gradient plus moment estimates without explicit geometric ownership.
- **Replay** means training again on stored old examples or old targets. It is a
  baseline for retention, not the desired final mechanism.
- **MLP** means the transformer feed-forward block. Current GCO projection work
  focuses there because MLP writes are more local than attention writes.
- **LM loss** means next-token language-model loss.
- **QA objective** means question-answer probe loss or accuracy used to test
  whether a learned fact or relation survives.
- **WWR** means wrong-write rate: how often the reasoner writes into the wrong
  route or memory region.
- **DAS** means dynamic adaptation score: a compact score for action quality
  across the simulated reasoner stream.
- **SSR** means state-space reasoner: the attempted mini-controller that maps
  evidence streams into write/protect/rewire/forget gates. It is parked as a
  primary path because it creates a nested learner inside the learner.
- **SVD** means singular value decomposition, used for capacity and subspace
  diagnostics.
- **CKA** means centered kernel alignment. It measures representational
  similarity across checkpoints or models.
- **Rebasing** means coherent movement of a family of representations while
  behavior remains usable. It is different from freezing old activations and
  different from uncontrolled fine-tuning drift.
- **Hypernetwork-like reasoner** means a controller that generates or modulates
  parameter edits. This is parked for now because it adds policy drift and
  compute before the invariants are clear.
- **EMA** means exponential moving average, used for smooth usage or frequency
  traces.
- **SGD** means stochastic gradient descent, the simplest gradient optimizer
  baseline.
- **CE** means cross-entropy loss.
- **GRU** means gated recurrent unit, a recurrent neural-network baseline used
  in the reasoner tests.

Common symbols:

```text
W_t  = weights
A_t  = active topology / wiring mask
M_t  = activation pathway matrix
H_t  = pressure history
P_t  = pressure gate
S_t  = recurrent geometric reasoning state
C_t  = protected/free capacity map
B_t  = behavior invariants / margins / readout constraints
T_t  = rebasing or basis-change operator
```

## Living Log Rule

Each durable experiment entry should answer:

```text
Claim:       what belief did this test support or reject?
Math:        the exact geometric object or update rule tested
Evidence:    the smallest numeric result that matters
Decision:    what changed in the research direction
Status:      proved / supported / failed / unresolved
Risk:        what can still invalidate the conclusion
```

Do not keep near-duplicate runs, progress-bar checks, JSON-write checks, or
seed-by-seed logs unless they reverse a prior conclusion.

## Evidence Ledger

### 1. Route Drift In A Minimal Transformer

**Claim.** Forgetting can be decomposed into route drift, write drift, and
readout drift instead of reported only as accuracy loss.

**Math.** In the copy-position transformer, the final query route to source
position `j` is:

```text
C_QK = W_Q W_K^T

R_j = x_query^T C_QK x_j

attention_j = softmax(R)_j
```

The write path is:

```text
C_OV = W_V W_O

residual_update = sum_j attention_j C_OV x_j
```

The readout is:

```text
logits = W_U h_final
```

The task stream was:

```text
Task A: [d0, d1, d2, QUERY] -> d0
Task B: [d0, d1, d2, QUERY] -> d1
```

**Evidence.**

```text
After Task A:
  Task A accuracy = 1.000
  query attention mostly position 0

After Task B:
  Task A accuracy = 0.200
  Task B accuracy = 1.000
  query attention mostly position 1
  C_QK drift = 2.067006
  C_OV drift = 0.647868
  W_U drift  = 0.308040
```

Freezing only `W_Q` and `W_K` did not preserve Task A. The route still moved
because embeddings and positions changed. Freezing `W_E`, `W_P`, `W_Q`, and
`W_K` preserved most of Task A but blocked Task B.

**Decision.** Preserve role geometry, not only operator weights. The protected
object is not `W_Q W_K^T` alone. It is the full bilinear route:

```text
x_query^T W_Q W_K^T x_source
```

**Status.** Proved in the minimal attention setting.

**Risk.** The task is intentionally simple; the result identifies a mechanism,
not a scalable optimizer.

### 2. Neuron Importance Predicts Conflict But Does Not Solve It

**Claim.** Neuron-level importance can identify likely collision regions, but
neuron protection alone does not solve continual learning.

**Math.** For hidden neuron `i`, tested diagnostics:

```text
A_i = mean |h_i|
D_i = ||W_out[i, :]||
E_i = max(0, L_ablate_i - L_base)

usage_i ~= A_i D_i E_i
```

Hard surgical masking used:

```text
protect_i = E_old_i > tau_old
needed_i  = G_new_i > tau_new

update_i = needed_i and not protect_i
```

Soft blending used:

```text
blend_i = E_new_i / (E_new_i + lambda E_old_i)
```

The toy stream was:

```text
old tasks: COPY0, COPY1, ADD01, MAX
new task:  ADD12
```

**Evidence.**

Importance scores were useful diagnostics:

```text
E vs loss_attribution Spearman rho = 0.7503 +/- 0.0540
E vs total_drift      Spearman rho = 0.6528 +/- 0.0706
```

But protection was not enough:

```text
best protection old_acc = 0.656, new_acc = 0.993
best surgical   old_acc = 0.697, new_acc = 0.813
```

The conflict mass was large:

```text
AEold_Gnew conflict neurons = 11.4 +/- 1.2
AEold_Gnew blocked_g        = 0.412 +/- 0.062
```

About 41% of the new-task gradient wanted old-used neurons.

**Decision.** Importance is a sensor, not the mechanism. The model needs route
allocation and transformation writes, not just protected neurons.

**Status.** Supported across toy MLP allocation tests.

**Risk.** The unit of analysis is still too coarse; the real unit is likely a
distributed route/operator.

### 3. Capacity Reset And Readout-Only Reuse Failed

**Claim.** Low-old-importance capacity is not empty, and new learning may need
feature-level movement rather than readout-only updates.

**Math.** Capacity reset selected the lowest-old-importance hidden pool:

```text
S_low = bottom_k(E_old or AE_old)
```

Then reset only:

```text
W1[:, S_low]
b1[S_low]
W2[S_low, :]
```

Readout-only reuse froze incoming features and updated only:

```text
Delta W2[S, :]
```

**Evidence.**

Random reset harmed new learning:

```text
AE_low_old_reset:
  reset_old_acc = 0.958
  old_acc       = 0.693
  new_acc       = 0.520

AE_low_old without reset:
  old_acc = 0.700
  new_acc = 0.782
```

Readout-only learning did not recover ADD12:

```text
AE_readout_all:
  old_acc = 0.634
  new_acc = 0.258

AE_safe_readout:
  old_acc = 0.727
  new_acc = 0.224
```

Hybrid feature+readout learned better but forgot more:

```text
AE_hybrid:
  incoming neurons = 32
  readout rows     = 42.5
  old_acc          = 0.662
  new_acc          = 0.852
```

**Decision.** Do not treat weakly used neurons as blank capacity. New tasks can
require feature movement, and feature movement is where old/new conflict lives.

**Status.** Failed interventions, useful constraints.

**Risk.** A better geometric reset or operator-level recycle might still work;
random reset is what failed.

### 4. Controlled Route Reasoning: Reuse Versus Split

**Claim.** A route-level geometric reasoner can distinguish compatible reuse
from conflicting split when the route geometry is controlled.

**Math.** Each route `r` maintained evidence:

```text
xi_{r,t} =
[
  activation_match,
  loss_gain,
  novelty,
  capacity,
  validation_damage,
  gradient_conflict,
  frequency
]
```

Recurrent route state:

```text
s_{r,t} = tanh(A_s s_{r,t-1} + B_s xi_{r,t} + b_s)
```

Role belief:

```text
b_{r,t} = softmax(W_b s_{r,t})

z_r in {
  unused,
  noisy,
  forming,
  useful,
  protected,
  conflicting,
  obsolete
}
```

Actions:

```text
reuse route
protect route
split route
write route
rewire topology
```

**Evidence.**

Controlled sequence:

```text
compatible task:
  T_B ~= T_A + noise
  GCO reused the same route
  Task A improved while Task B was learned

conflicting task:
  T_B = -T_A
  Adam/SGD learned B but forgot A
  GCO split/protected and kept A stable
```

**Decision.** The route-state belief mechanism is a viable core for GCO.

**Status.** Proved in synthetic route-space.

**Risk.** The oracle and route structure were controlled; real semantics require
noisier sensors.

### 5. Real Embedding Relation Reasoning

**Claim.** Real text embeddings contain relation geometry useful for route
decisions, but contradiction/conflict is not solved by cosine similarity.

**Math.** For embedding pair `(e_a, e_b)`, the reasoner used:

```text
phi(a,b) =
[
  e_a,
  e_b,
  |e_a - e_b|,
  e_a * e_b,
  cos(e_a, e_b),
  ||e_a - e_b||,
  angle(e_a, e_b)
]
```

Actions:

```text
compatible -> reuse route
conflict   -> split route
bridge     -> bridge route
novel      -> create route
```

**Evidence.**

Dataset:

```text
160 handcrafted text pairs
40 compatible
40 conflict
40 bridge
40 novel
embedding model = all-MiniLM-L6-v2
```

Results:

```text
random baseline        18.75%
cosine threshold       46.88%
MLP classifier         68.75%
GCO recurrent reasoner 68.75%

GCO per-class:
  compatible 60.00%
  conflict   33.33%
  bridge    100.00%
  novel     100.00%
```

**Decision.** Use embeddings as one sensor, not the decision rule. Conflict
needs behavior, activation, and damage sensors.

**Status.** Early real-embedding signal, not proof of superiority over
classifiers.

**Risk.** Conflict accuracy is still weak.

### 6. Dynamic Multi-Factor GCO Reasoner

**Claim.** Dynamic, multi-factor geometric reasoning beats static/current-state
baselines for structural route actions.

**Math.** The evolved evidence vector for route/operator `r` at layer `l` is:

```text
xi_{l,r,t} =
[
  activation_match,
  semantic_margin_contribution,
  usefulness,
  conflict_with_old_routes,
  predicted_damage,
  frequency,
  novelty,
  free_capacity,
  route_depth,
  input_length,
  time_or_recency,
  l / L
]
```

The recurrent reasoner:

```text
s_{l,r,t} =
tanh(A_s s_{l,r,t-1} + B_s xi_{l,r,t} + b_s)

b_{l,r,t} = softmax(W_b s_{l,r,t})
```

Belief-derived gates:

```text
g_write   = P(forming or useful)
g_protect = P(protected)
g_rewire  = P(conflicting or new)
g_decay   = P(noisy or obsolete)
```

Action space:

```text
IGNORE
WRITE_REUSE
WRITE_NEW
PROTECT
SPLIT_ROUTE
BRIDGE_ROUTE
REWRITE_WEAK
DECAY_ROUTE
```

**Evidence.**

Evaluation phase was `T in [600, 800]` after online simulation training.

```text
configuration          action_acc  WWR     split_recall  route_purity  DAS
GCO-full               68.00%      4.80%   37.04%        46.15%        99.45
GCO-no-time            67.00%      2.40%   37.04%        46.15%        99.65
GCO-no-frequency       67.50%      4.00%   44.44%        46.15%        99.35
GCO-no-activation      66.00%      3.20%   40.74%        46.15%        98.25
GCO-no-weight-imp.     70.50%      5.60%   40.74%        46.15%       101.55
GCO-no-jac-damage      65.00%      4.00%   33.33%        46.15%        96.85
GCO-no-topology        69.00%      4.80%   40.74%        46.15%       100.45
GCO-no-recurrence      67.50%      4.00%   44.44%        46.15%        99.35
MLP-current-only       51.00%      0.00%   29.63%        46.15%        84.85
GRU-sequence-baseline  36.00%      0.00%   25.93%        46.15%        69.85
cosine-rule-baseline   56.50%     34.40%    0.00%        46.15%        73.15
```

**Decision.** Replace scalar pressure in the transformer with route evidence,
recurrent route state, and belief/action heads.

**Status.** Supported. The full reasoner beats static/current baselines and
massively reduces wrong-write rate versus cosine.

**Risk.** Some ablations match or beat full GCO, and route purity is unchanged.
This means some sensors are noisy or misweighted.

### 7. Living-Map GFO On Real-Book Text

**Claim.** GFO-style geometric preservation improves retention/composition over
AdamW and replay in real text, but the external living map is not the final
model-native architecture.

**Math.** The living-map system protected semantic anchors. For prompt `q` and
answer `y+`, first-token semantic margin:

```text
margin(q) = log p(y+ | q) - log p(y- | q)
```

Anchor drift:

```text
drift_i = ||h_l(q_i; theta_t) - h_l(q_i; theta_anchor)||^2
```

Semantic margin loss:

```text
L_margin = max(0, m_required - margin(q))
```

Full-answer preservation added teacher-forced sequence loss:

```text
L_answer = CE(answer_tokens | prompt)
```

**Evidence.**

Representative five-seed cluster-2 result:

```text
heldout_retention_token_accuracy_final
  adamw          0.0444
  replay_living  0.1778
  gfo_living     0.4111

heldout_composition_token_accuracy_final
  adamw          0.1667
  replay_living  0.1667
  gfo_living     0.4000

semantic_answer_exact_match_rate_final
  adamw          0.2354
  replay_living  0.5848
  gfo_living     1.0000
```

GFO also repeatedly produced lower destructive drift and zero semantic-margin
violation in many runs.

**Decision.** Keep living-map results as an external-system upper signal, but
move mechanisms into the transformer.

**Status.** Supported as a benchmark signal.

**Risk.** Uses external anchors/controllers. It does not satisfy the final
requirement: input data only, model-native CL dynamics.

### 8. Native Trace Writing In The Transformer

**Claim.** Native trace slots can write and read memory inside the transformer,
but routing currently collapses.

**Math.** Native trace adapter reads both residual and input/source stream:

```text
read_features = [x, source, x - source, x * source]

reason_update =
sigmoid(source_mix)
* sigmoid(reason_gate)
* softplus(reason_gain)
* tanh(reasoner(read_features))

reasoned = LayerNorm(x + reason_update)
```

Sparse slot routing:

```text
score_s = normalize(reasoned)^T normalize(key_s + fast_key_delta_s)

top_gates = softmax(top_k(score + write_gate + homeostasis))
```

Fast write state:

```text
slot_state_s      <- compressed_state
fast_key_delta_s  <- normalize(slot_summary_s) - key_s
fast_memory_s     <- compressed_state
fast_value_s      <- compressed_value
```

Fast read:

```text
update =
low_rank_param_update
+ softplus(fast_read_gain) * (fast_memory_s U_s + fast_value_s)
```

**Evidence.**

Native runs showed:

```text
fast_value_norm > 0
fast_update_energy > 0
write_rate > 0
error_pressure > 0
```

Final native GCO/pre run still showed collapse:

```text
native_usage_imbalance = 1.0000
slot_max_share = 0.7460
slot_usage_ema_max = 0.4861
slot_usage_ema_min = 0.0006
```

**Decision.** Fast write/read is alive. The bottleneck is memory distribution:
route monopoly, underused slots, and weak heldout retention.

**Status.** Mechanism works mechanically; architecture unresolved.

**Risk.** Current slot homeostasis is not enough. Needs belief/action route
reasoner and true structural operators.

### 9. Online GCO Projection On Transformer MLPs

**Claim.** The online GCO optimizer can observe MLP activation pathways,
accumulate pressure, and remove a protected geometric component from updates.

**Math.** For an MLP weight matrix `W_l`, activation pathway:

```text
M_t = (1 / B) |a_t|^T |x_t|
M_t = M_t / (percentile(M_t, 99) + eps)
M_t = clip(M_t, 0, 1)
```

Pressure history:

```text
H_t = beta H_{t-1} + (1 - beta) (|G_t| * M_t)
```

Layer-wise pressure:

```text
mu_l    = mu_base * (L - l) / L
gamma_l = gamma_base * (l + 1) / L

P_t = sigmoid(gamma_l (H_t - mu_l))
P_effective = warmup(t) P_t
```

Row-wise projection:

```text
S_j = <G_j, W_j> / (<W_j, W_j> + eps)
Proj_j = S_j W_j

G_tilde = G - P * Proj
```

When `P -> 1`:

```text
<G_tilde_j, W_j> = 0
```

**Evidence.**

Final diagnostic run:

```text
projected MLP matrices         = 8
gco_pressure_mean              = 0.4405
gco_projection_delta_ratio     = 0.0839
gco_safe_update_ratio          = 0.9863
seen_retention_forgetting_mean = 0.0000
heldout_retention_accuracy     = 0.0556
```

**Decision.** Online GCO projection is a working v0 baseline. It is not yet the
evolved structural GCO.

**Status.** Implemented and verified mechanically.

**Risk.** Projection alone did not solve heldout retention or slot collapse.

### 10. Native Scratch GCO: Outcome-Controlled Writes

**Claim.** A scratch transformer can run without `torch.optim`, using backward
gradients only as local error signals, while GCO performs targeted native
writes. The current write mechanism is mechanically healthy, but formation does
not yet persist strongly enough to create broad hardening.

**Math.** The direct write target uses the current hidden input and output
error:

```text
K_l = h_l
E_l = grad_{h_{l+1}} L
```

For a linear GCO matrix:

```text
Delta W_l^* =
E_l K_l^T
(
  K_l K_l^T
  + lambda_p P_protect
  + lambda_n I
)^{-1}
```

The forward uses `W * A`, so the direct write is converted back through the
active topology:

```text
Delta W_native = Delta W_effective / max(A_t, A_min)
```

with `A_min = 0.05` in the current ablation. This prevents future topology
pruning from creating unstable division by near-zero routes.

Measured same-batch utility is:

```text
U =
(L_before - L_after) / |L_before|
- edit_cost
- capacity_cost
- rewire_cost
- forget_cost
```

The write gate is trained by comparative advantage:

```text
adv = U - EMA(U)

target_write =
0.5 + 0.5 tanh(
  warmup(t)
  clamp(route_advantage / route_credit_scale, -c, c)
)

if U > 0:
  target_write = max(target_write, 0.5)
```

Formation is trained by absolute positive utility, not by advantage:

```text
route_formation_raw =
max(0, U)
* global_eligibility_share
* selected_count

formation_credit =
tanh(
  warmup(t)
  clamp(route_formation_raw / route_formation_scale, 0, c_f)
)

F_ij <- F_ij + eta_F formation_credit_ij (1 - F_ij)
```

This distinction matters:

```text
advantage          -> should the write gate open more than baseline?
positive utility   -> did this route help enough to become more formed?
negative utility   -> failed write / possible rewire signal
```

**Evidence.**

First, a positive-utility write floor fixed the target but exposed a write-gate
collapse:

```text
gco-native-positive-write-floor-fit80-seed0.json

U=+1.28e-05
adv=-4.89e-06
target=0.500/0.500/0.500
write_gate_raw_mean=0.3879
```

Then the write gate was separated from the internal heuristic teacher:

```text
gco-native-outcome-controlled-write-fit80-seed0.json

write_gate_raw_mean=0.5034
outcome_gate_error=0.0039
active_hardening=0.00085
```

Then positive utility was allowed to build formation even when advantage was
negative:

```text
gco-native-positive-utility-formation-fit80-seed0.json

U=+1.65e-05
adv=-6.77e-07
target=0.500/0.500/0.500
form=0.291/0.964
active_hardening=0.00881
active_crystalline=0.00000
```

Finally, the outcome-only ablation disabled the internal reasoner and structural
controllers:

```text
--reasoner-lr 0
--internal-gate-lr-scale 0
--internal-write-gate-lr-scale 0
--grow-lr 0
--prune-lr 0
--forget-lr 0
```

Result:

```text
gco-native-outcome-only-formation-fit80-seed0.json

write_gate_mean=0.5030
active_hardening=0.00884
active_crystalline=0.00000
form=0.293/0.964
rewire=0
forget=0
```

This is almost identical to the positive-utility formation run, so the internal
heuristic reasoner is not the current bottleneck.

**Decision.** The native scratch loop is no longer self-sabotaging. The write
gate, measured outcome credit, and positive-utility formation are active. The
next problem is not "can it write?" but "can formation accumulate at
circuit/route scale?"

**Status.** Supported mechanically on a one-chunk real-book scratch run.

**Risk.** The model still fits weakly, hardening remains below 1% active mass,
and crystalline/protection has not emerged. Formation is probably too
weight-local.

### 11. Write-Only Continual Learning Failed The Geometry Test

**Claim.** Continual learning is not solved by safe writing or protected local
rewiring alone. A successful same-capacity model trained from scratch on more
data uses coordinated representational rebasing: old behavior remains good
while old residual states move substantially.

**Math.** Three same-spec tiny transformers were compared:

```text
model spec:
  vocab_size = 2000
  d_model    = 128
  layers     = 2
  heads      = 4
  d_ff       = 256
  seq_len    = 32

training paths:
  base100: random -> first 100 words
  CL:      base100 -> second 100 words
  base200: random -> first 200 words
```

Behavior metric:

```text
margin = logit(correct token) - max_{y != correct} logit(y)
```

Geometry metrics:

```text
effective_rank(H) =
exp(
  -sum_i p_i log p_i
)

p_i = sigma_i / sum_j sigma_j
```

Matched drift used orthogonal Procrustes alignment:

```text
R* = argmin_R ||H_B R - H_A||_F
subject to R^T R = I

drift = mean_i ||h_B,i R* - h_A,i||
relative_drift = drift / mean_i ||h_A,i - mean(H_A)||
```

Representational similarity used linear CKA:

```text
CKA(X, Y) =
||X^T Y||_F^2
/
(
  ||X^T X||_F ||Y^T Y||_F
)
```

**Evidence.**

Base fits:

```text
base100:
  loss 7.7006 -> 0.0145
  acc  = 0.9913
  mean margin on old span = +11.4909

base200:
  loss 7.6761 -> 0.0247
  acc  = 0.9867
  mean margin on old span = +11.8631
  mean margin on new span = +11.5536
```

Continual updates from `base100`:

```text
safe update:
  old loss 0.014533 -> 0.014550
  old acc  0.9913   -> 0.9913
  new loss 13.6280  -> 12.4667

aggressive update:
  old loss 0.014533 -> 0.031328
  old margin +11.4909 -> +9.0665
  new loss 13.6280 -> 10.0840
```

The safe update preserved old behavior but did not learn the new span. The
aggressive update learned slightly more new text but weakened old behavior and
still did not approach the successful `base200` solution.

Geometry on the same original 100 words:

```text
base100 -> base200, final layer:
  rank 74.32 -> 87.87
  relative matched drift = 0.7880
  CKA = 0.4906
  aligned cosine = 0.6783

base100 -> safe CL, final layer:
  rank 74.32 -> 74.18
  relative matched drift = 0.0779
  CKA = 0.9930

base100 -> aggressive CL, final layer:
  rank 74.32 -> 74.87
  relative matched drift = 0.1348
  CKA = 0.9760
```

Full page with images:
[Continual Learning Is A Geometry Problem]({{ site.baseurl }}/continual-learning-geometry/).

**Decision.** The research direction changes from:

```text
find the safe write location and protect old activations
```

to:

```text
preserve old behavior and useful relations
while allowing coordinated representational rebasing
```

Exact activation protection is now treated as a diagnostic baseline, not the
final objective. Useful drift and destructive drift must be separated.

**Status.** Strongest current pivot. Supported by same-spec controlled runs,
not yet by broad benchmarks.

**Risk.** One tiny model, one text source, near-memorization regime, and one
main seed. The conclusion is directional: it shows why write/protection alone
is incomplete, not that the final rebasing algorithm exists.

### 12. Reasoner Design Pivot: Nested Learners Are Too Risky For Now

**Claim.** A reasoner is still needed, but constantly training a second
network inside the learner is not the right default path yet. It drifts toward
nested learning or hypernetwork-style policy learning, which adds compute,
credit-assignment instability, and another continual-learning problem inside
the continual-learning system.

**Math.** The learned-reasoner family used evidence vectors and gates:

```text
xi_t =
[
  activation,
  error,
  pressure,
  novelty,
  frequency,
  capacity,
  collision,
  predicted_damage,
  recency
]

s_t = f_theta(s_{t-1}, xi_t)

[
  g_write,
  g_protect,
  g_rewire,
  g_forget,
  g_compress,
  priority
] = heads_theta(s_t)
```

The state-space reasoner / SSR path made this more explicit:

```text
s_{t+1} = A s_t + B xi_t
value_t = C s_t
gate_t = sigmoid(D s_t)

theta_reasoner <- theta_reasoner + eta * outcome_credit
```

This is powerful, but it creates a second learner whose own policy changes
during training.

**Evidence.**

The dynamic multi-factor reasoner beat static/current baselines in controlled
route simulation:

```text
GCO-full action accuracy = 68.00%
MLP-current-only         = 51.00%
GRU-sequence-baseline    = 36.00%
cosine-rule-baseline     = 56.50%

GCO-full wrong-write rate = 4.80%
cosine wrong-write rate   = 34.40%
```

The native SSR-style writer was mechanically active and improved some write
signals:

```text
ssr-write-protect-coupled-3chunk:
  update ~= 0.00202
  active hardening ~= 0.33-0.44
  SSRgateW ~= 0.52
  SSRgateP ~= 0.35-0.40
```

But the practical failure mode was architectural, not just numeric:

```text
online learned reasoner
-> changing write/protect policy
-> policy instability risk
-> extra forward/credit path
-> nested continual-learning problem
```

Hypernetwork-like variants have the same problem if they generate edits from a
small controller that itself must keep learning online. They may be useful
later, but they are not the stable foundation for the next phase.

**Decision.** Park constantly trained online reasoners, SSR controllers, and
hypernetwork-style edit generators as primary mechanisms. Use deterministic or
slowly parameterized algorithmic reasoning first: compute the evidence map,
apply explicit geometry/behavior tests, and only then decide write/protect/
rebase/decay actions. If a learned reasoner returns, it should be trained
offline or updated slowly against stable objectives, not allowed to rewrite its
own policy at every step.

**Status.** The reasoner concept is supported; the online nested-learner
implementation path is parked.

**Risk.** A fixed algorithmic reasoner may be too rigid. Eventually the system
may need a learned reasoner, but only after the invariants are clear.

## Current Target After The Geometry Pivot

The evolved state is:

```text
Omega_t = {
  W_t,  weights
  A_t,  active topology / wiring mask
  Q_t, R_t, basis rotations or neuron permutations
  O_t,  reusable thought operators
  C_t,  capacity and protected/free subspace map
  B_t,  behavior invariants / margins / readout constraints
  S_t   geometric reasoning state
}
```

The intended layer is:

```text
h_{l+1} =
sigma(
  [
    Q_l (A_l * W_l) R_l^T
    +
    sum_r pi_{l,r}(h_l, S_t) O_{l,r}
  ] h_l
)
```

Each thought operator is low rank:

```text
O_{l,r} = U_{l,r} C_{l,r} V_{l,r}^T
```

Direct geometric write target:

```text
Delta W_l K_l ~= E_l

K_l = h_l
E_l = -grad_{h_{l+1}} L
```

Simplified protected write:

```text
Delta W_l^* =
E_l K_l^T
(
  K_l K_l^T
  + lambda_p P_protect
  + lambda_n I
)^{-1}
```

Targeted edit set:

```text
e_ij =
|Delta W_ij^*|
*
[
  1
  + alpha_w Write_ij
  + alpha_f Free_ij
  + alpha_n Novelty_ij
  - alpha_p Protect_ij
  - alpha_d Damage_ij
  - alpha_c Conflict_ij
]

K_t = TopK(e_t, k_t)

Delta W_GCO = mask(K_t) * Delta W^*
```

The missing object after the pivot is a rebasing operator. A local write tries:

```text
Delta W K ~= E
```

A rebasing step instead asks for a joint movement:

```text
H_old' = T H_old
H_new' = T H_new

subject to:
  behavior_old(H_old') ~= behavior_old(H_old)
  loss_new decreases
  relation_geometry preserved where useful
  capacity/rank improves or interference falls
```

The next model object, if experiments resume, should be `GeometricMLP` or
`GeometricBlock` with:

```text
A_t topology mask
multi-scale formation memory F_ij, F_row, F_col, F_block
hardening hysteresis with enter/exit thresholds
low-rank O_r operators
route evidence xi
explicit behavior invariants B_t
algorithmic geometry reasoner before learned online reasoner
rebasing / basis-change operator T_t
operator create / protect / merge / decay
```

## What Is Still Unresolved

Keep these visible:

```text
slot collapse / route monopoly
heldout retention weakness
real semantic conflict detection
capacity recovery without destroying weak structure
direct geometric write solver
formation persistence at circuit/route scale
structural topology mask A_t
basis or neuron rearrangement Q_t, R_t
operator creation and consolidation
behavior-preserving rebasing operator T_t
distinguishing useful drift from destructive drift
offline anchor/Jacobian/behavior audit phase
architecture/direct-write control against AdamW one-chunk fitting
multi-seed and longer-text geometry checks
```

The strongest current conclusion is:

```text
geometric reasoning works in controlled route-space;
dynamic route reasoning beats static/current baselines;
native writing, outcome-controlled direct writes, and online projection work mechanically;
write/protection-only CL preserves old geometry or damages it without learning enough new data;
successful same-capacity learning can require coordinated representational rebasing;
full model-native continual learning needs structural routing,
multi-scale formation, operator creation, capacity-aware targeted writes,
and behavior-preserving geometry rebasing.
```
