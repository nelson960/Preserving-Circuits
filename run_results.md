# Controlled CL Run Results

This file is the evidence ledger for the controlled continual-learning work.
It records the meaningful results and what they imply. It is separate from
`CL_Architecture.md`, which should contain only architecture and math.

## Current Claim

The current evidence supports this limited claim:

```text
In a toy continual-learning setting, controlled updates can preserve selected
old behavior, guard uncertain behavior, update changed facts, suppress obsolete
answers, learn staged new facts, and keep representational geometry healthier
than naive sequential training.
```

The current evidence does not support these stronger claims:

```text
the method solves continual learning
the method scales to large models
the invariant-tangent update is always better than loss-controlled training
the role controller can fully decide preserve/drop roles without supervision
```

## Architecture Under Test

The controlled learner uses:

```text
base model
new staged data
preserve anchors
guard anchors
dynamic committed anchors
obsolete/drop checks
geometry measurements
bounded committed memory
```

The unique update mechanism under test is the invariant-tangent update:

```text
g_N = grad_theta L_new(theta)

A_t = protected behavior + geometry constraint rows

g_tangent =
  g_N - A_t^T (A_t A_t^T + rho I)^-1 A_t g_N

g_update =
  g_tangent + alpha_restore * g_restore
```

This differs from ordinary preservation losses because the protected behavior
and geometry do not only add loss terms. They reshape the update direction.

## Main Colab Comparison

Source files:

```text
/Users/nelson/Downloads/colab-rich-invariant.json
/Users/nelson/Downloads/colab-loss-baseline.json
```

Both runs used the same staged continual-learning setup:

```text
base_word_target        3000
conversation_word_target 2400
conversation_stages     4
d_model                 192
layers                  3
heads                   4
d_ff                    768
commit_memory_budget    12
```

### Behavior Accuracy

Exact-match results:

| method | preserve | guard | changed | new | composition | obsolete old answer |
|---|---:|---:|---:|---:|---:|---:|
| base | 1.000 | 1.000 | 0.000 | 0.000 | 0.333 | 1.000 |
| naive sequential | 0.000 | 0.000 | 0.000 | 0.300 | 0.167 | 0.000 |
| loss-controlled | 1.000 | 1.000 | 1.000 | 0.700 | 0.833 | 0.000 |
| invariant-tangent | 1.000 | 1.000 | 1.000 | 0.600 | 0.778 | 0.000 |
| joint from scratch | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

Interpretation:

```text
naive learns some new data but catastrophically forgets old preserve/guard behavior
loss-controlled preserves and learns better than invariant-tangent on new behavior
invariant-tangent still learns meaningful new behavior while preserving old behavior
joint is the upper reference, not continual learning
```

### Behavior Loss

| method | preserve loss | guard loss | changed loss | new loss | composition loss |
|---|---:|---:|---:|---:|---:|
| naive sequential | 7.328 | 7.723 | 6.449 | 3.217 | 4.630 |
| loss-controlled | 0.000230 | 0.000201 | 0.00145 | 0.432 | 0.228 |
| invariant-tangent | 0.000230 | 0.000202 | 0.00112 | 0.891 | 0.608 |

Interpretation:

```text
loss-controlled has stronger new fitting
invariant-tangent keeps old behavior but is more conservative on new data
```

## Residual Geometry

Mean residual geometry relative to the base model:

| comparison | drift relative | CKA | rank delta |
|---|---:|---:|---:|
| naive vs base | 0.724 | 0.603 | -5.749 |
| loss-controlled vs base | 0.698 | 0.829 | +1.116 |
| invariant-tangent vs base | 0.469 | 0.841 | +3.713 |
| joint vs base | 1.006 | 0.553 | +5.890 |

Interpretation:

```text
naive damages old geometry
loss-controlled preserves behavior and improves CKA, but still drifts heavily
invariant-tangent has much lower global residual drift and high CKA
joint learns everything but builds a different geometry from scratch
```

Important comparison:

```text
loss-controlled new exact      = 0.700
invariant-tangent new exact    = 0.600

loss-controlled residual drift = 0.698
invariant-tangent drift        = 0.469
```

The invariant-tangent update trades some new-learning strength for safer
representational motion.

## Role Geometry

Mean role-geometry metrics:

| comparison | centroid drift | centroid cosine | group CKA | separation drift |
|---|---:|---:|---:|---:|
| naive vs base | 3.806 | 0.491 | 0.629 | 0.270 |
| loss-controlled vs base | 1.957 | 0.812 | 0.774 | 0.310 |
| invariant-tangent vs base | 2.295 | 0.795 | 0.757 | 0.227 |
| joint vs base | 5.626 | -0.059 | 0.677 | 0.525 |

Interpretation:

```text
loss-controlled keeps role centroids slightly closer
invariant-tangent better preserves role separation geometry
joint reorganizes roles heavily
```

## Feature Geometry

Mean feature-geometry metrics:

| comparison | centroid drift | centroid cosine | group CKA | separation drift |
|---|---:|---:|---:|---:|
| naive vs base | 3.790 | 0.514 | 0.584 | 0.354 |
| loss-controlled vs base | 2.012 | 0.828 | 0.704 | 0.175 |
| invariant-tangent vs base | 2.219 | 0.807 | 0.676 | 0.148 |
| joint vs base | 5.581 | -0.058 | 0.583 | 0.398 |

Interpretation:

```text
loss-controlled keeps feature centroids slightly closer
invariant-tangent better preserves feature separation geometry
```

This matches the constraint design: the invariant-tangent run explicitly
included category centroid and separation rows.

## Projection Diagnostics

Invariant-tangent Colab run:

| stage | constraint rows | removed gradient fraction | safe gradient fraction | committed memory size |
|---:|---:|---:|---:|---:|
| 1 | 6.0 | 0.0329 | 0.9993 | 9 |
| 2 | 19.0 | 0.0296 | 0.9995 | 12 |
| 3 | 20.8 | 0.0458 | 0.9992 | 12 |
| 4 | 20.8 | 0.0391 | 0.9992 | 12 |

Interpretation:

```text
the constraint matrix is active
the update is not collapsing to zero
projection removes a small but measurable unsafe component
committed memory remains bounded at 12
```

The current toy problem does not create a huge gradient conflict. The projection
therefore changes the path subtly, not by deleting most of the gradient.

## Important Failure: Pure Projection

A tuned run removed the restorative term:

```text
projected_restore_strength = 0.0
lambda_geometry_anchor     = 0.02
```

Result:

```text
preserve exact = 0.000
guard exact    = 0.000
changed exact  = 0.000
```

Anchor drift became very large:

```text
stage 1 preserve KL ~= 7.79
stage 1 guard KL    ~= 8.62
stage 1 geometry    ~= 1.75
```

Interpretation:

```text
pure tangent projection is not sufficient
projection prevents local collision but does not restore accumulated drift
stable CL needs projection + bounded restorative control + verification
```

This failure is architecturally useful because it identifies the stable form:

```text
g_update =
  project_tangent(g_new, A_t)
  + alpha_restore * g_restore
```

## Earlier Bounded Controlled Proof

Source:

```text
model/analysis/mini-cl-bounded-proof-seed0.json
```

Key result:

```text
base:
  preserve 1.000, guard 1.000, changed 0.000, new 0.000, compose 0.333

naive:
  preserve 0.000, guard 0.000, changed 0.000, new 0.400, compose 0.222

controlled:
  preserve 1.000, guard 1.000, changed 1.000, new 0.767, compose 0.889

joint:
  preserve 1.000, guard 1.000, changed 1.000, new 1.000, compose 1.000
```

Memory budget:

```text
commit_memory_budget = 12
stage 1 total = 12
stage 2 total = 12
stage 3 total = 12
```

Interpretation:

```text
bounded dynamic committed memory worked
naive CL catastrophically forgot
controlled CL preserved old behavior and learned staged new behavior
```

## Storage / Capacity Frontier

The storage frontier sweep measured how much data small models can fit from
scratch before strict accuracy drops.

Important boundary for facts-style data:

```text
tiny facts 3000 words ~= 8447 tokens  -> strict fit
tiny facts 5000 words ~= 14078 tokens -> strict fail
```

Approximate lesson:

```text
capacity pressure is measurable
factual binding data hits limits earlier than book/mixed text
continual learning cannot be solved as write protection alone
capacity and representation geometry matter
```

## Current Interpretation

The strongest current evidence:

```text
1. Naive sequential learning catastrophically forgets.
2. Behavior/guard/committed constraints prevent catastrophic forgetting.
3. Bounded committed memory prevents unbounded replay growth.
4. Invariant-tangent projection changes the geometry tradeoff:
   it reduces residual drift and preserves separation geometry better,
   but currently learns new behavior less strongly than loss-controlled training.
5. Pure projection is unstable; projection must be combined with bounded restore.
```

## Current Open Problems

```text
new learning is still below joint training
projection is not yet clearly better than loss-controlled training on behavior
constraint rows are expensive
the role controller is still externally assigned in the toy setup
composition generalization remains weak unless the task is simple or trained
scaling needs lower-rank or block-local approximations
```

## What To Show In A Report

The report should not dump every run. It should show this story:

```text
1. Same model, same task, naive CL forgets.
2. Controlled CL avoids catastrophic forgetting.
3. Ordinary loss-controlled CL learns well but moves geometry more.
4. Invariant-tangent CL moves geometry less and preserves separation structure.
5. The mechanism is a constrained update operator, not just replay.
6. The current limitation is the stability-plasticity tradeoff.
```

