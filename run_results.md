# Controlled CL Run Results

This file is the evidence ledger for controlled continual-learning experiments.
It records meaningful results and what they imply. It is separate from
`CL_Architecture.md`, which should describe the architecture and math rather
than every run.

## Current Claim

The current evidence supports this limited claim:

```text
In a toy staged continual-learning setting, an invariant-tangent update with
a bounded restore correction can preserve selected old behavior, guard behavior,
learn changed facts, learn new facts, suppress obsolete answers, and keep
residual geometry healthier than naive sequential learning.
```

The current evidence does not yet support these stronger claims:

```text
the method solves continual learning
the method scales to large models
restore strength 0.05 is robust or optimal
the role controller can fully decide preserve/drop/guard roles by itself
the method works across seeds or hard open-ended streams
```

## Mechanism Under Test

The unique update mechanism under test is:

```text
g_N = grad_theta L_new(theta)

A_t = protected behavior + protected geometry constraint rows

g_tangent =
  g_N - A_t^T (A_t A_t^T + rho I)^-1 A_t g_N

g_restore =
  grad_theta (L_preserve + L_guard + L_geometry)

g_update =
  g_tangent + alpha_restore * g_restore
```

Projection alone failed. Projection plus restore succeeded.

## Main Evidence Pair

Source artifacts:

```text
model/analysis/colab-plasticity-audit/colab-plasticity-audit.json
model/analysis/colab-plasticity-audit2/colab-plasticity-audit2.json
```

Both runs used the same staged continual-learning protocol:

```text
base_word_target          5000
conversation_word_target  1800
conversation_stages       4
d_model                   192
layers                    3
heads                     4
d_ff                      768
controlled_update_mode    projected_invariant_tangent
projected_solver          gram
constraint_mode           category_centroid_separation
commit_memory_budget      48
```

The only decisive difference:

```text
projection-only run:
  projected_restore_strength = 0.0

successful restore run:
  projected_restore_strength = 0.05
```

## Behavior Accuracy

Exact-match success scores:

| method | preserve | guard | changed | new | composition | suppress obsolete |
|---|---:|---:|---:|---:|---:|---:|
| base | 1.000 | 1.000 | 0.000 | 0.000 | 0.333 | 0.000 |
| naive sequential | 0.000 | 0.000 | 0.000 | 0.300 | 0.167 | 1.000 |
| invariant-tangent, no restore | 0.000 | 0.000 | 0.000 | 0.300 | 0.167 | 1.000 |
| invariant-tangent + restore | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| joint from scratch | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Interpretation:

```text
naive learns a little new data but catastrophically forgets preserve/guard
projection-only also forgets preserve/guard
projection + restore preserves old behavior and learns all target categories
joint is an upper reference, not a continual-learning method
```

## Behavior Loss

| method | preserve loss | guard loss | changed loss | new loss | composition loss |
|---|---:|---:|---:|---:|---:|
| naive sequential | 5.9348 | 5.5818 | 5.9220 | 2.6712 | 3.7792 |
| invariant-tangent, no restore | 8.1709 | 8.4551 | 7.3757 | 4.3208 | 6.1792 |
| invariant-tangent + restore | 0.0001 | 0.0001 | 0.0026 | 0.0020 | 0.0014 |

Interpretation:

```text
pure tangent projection left enough plasticity, but failed stability
restore correction pulled the trajectory back to the protected manifold
```

## Residual Geometry

Mean residual geometry relative to the base model:

| comparison | drift relative | CKA | rank delta |
|---|---:|---:|---:|
| naive vs base | 0.5753 | 0.6715 | -0.0033 |
| invariant-tangent, no restore vs base | 0.5693 | 0.6280 | -1.5120 |
| invariant-tangent + restore vs base | 0.3934 | 0.8608 | +8.3101 |
| joint vs base | 0.8196 | 0.5498 | +8.5128 |

Interpretation:

```text
restore run has lower residual drift than naive and joint
restore run has much higher CKA with base than naive and joint
joint learns everything but creates a more different internal geometry
```

## Role And Feature Geometry

Successful restore run, mean role geometry:

| comparison | centroid drift | centroid cosine | group CKA | separation drift |
|---|---:|---:|---:|---:|
| naive vs base | 4.2723 | 0.5320 | 0.6838 | 0.1421 |
| invariant-tangent + restore vs base | 2.6439 | 0.7942 | 0.7662 | 0.1591 |
| joint vs base | 6.2848 | -0.0083 | 0.7106 | 0.2587 |

Successful restore run, mean feature geometry:

| comparison | centroid drift | centroid cosine | group CKA | separation drift |
|---|---:|---:|---:|---:|
| naive vs base | 4.3988 | 0.5431 | 0.6474 | 0.1474 |
| invariant-tangent + restore vs base | 2.6149 | 0.8188 | 0.6969 | 0.2237 |
| joint vs base | 6.2571 | 0.0055 | 0.6551 | 0.2155 |

Interpretation:

```text
restore run strongly improves centroid drift and centroid cosine versus naive
separation drift is mixed and should not be overclaimed
```

## Plasticity Audit

Projection-only run:

| stage | safe/raw | final/raw | removed | effective rank | rows | redundancy | committed memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9987 | 0.9987 | 0.0480 | 3.026 | 6.0 | 0.4957 | 9 |
| 2 | 0.9982 | 0.9982 | 0.0588 | 1.106 | 19.0 | 0.9413 | 18 |
| 3 | 0.9930 | 0.9930 | 0.1169 | 1.081 | 21.2 | 0.9480 | 27 |
| 4 | 0.9981 | 0.9981 | 0.0614 | 1.072 | 18.2 | 0.9404 | 36 |

Restore run:

| stage | safe/raw | final/raw | removed | effective rank | rows | redundancy | committed memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9995 | 1.4272 | 0.0273 | 5.282 | 6.0 | 0.1197 | 9 |
| 2 | 0.9988 | 1.4413 | 0.0466 | 1.107 | 19.0 | 0.9412 | 18 |
| 3 | 0.9972 | 1.0591 | 0.0739 | 1.046 | 21.2 | 0.9495 | 27 |
| 4 | 0.9990 | 1.2120 | 0.0436 | 1.078 | 18.2 | 0.9399 | 36 |

Interpretation:

```text
plasticity did not collapse
safe/raw stayed close to 1.0 across all stages
restore makes final/raw exceed 1.0 because it adds corrective force
constraint rows become highly redundant after stage 1
```

The next optimization target is therefore:

```text
compress redundant constraint rows into a low-rank protective basis
```

## Current Interpretation

The strongest evidence now:

```text
1. Naive sequential learning catastrophically forgets.
2. Projection-only tangent updates are not stable enough.
3. Projection + bounded restore succeeds on the staged toy CL task.
4. Successful run preserves old/guard behavior, learns changed/new/composed facts,
   and suppresses obsolete old answers.
5. Successful run keeps residual geometry closer to base than naive or joint.
6. Plasticity does not collapse; the bottleneck is stable protection, not lack
   of new-learning direction.
```

## Current Open Problems

```text
restore strength has not been swept
results are not yet multi-seed
role labels are still externally supplied by the protocol
constraint basis is redundant
checkpoint-free plot regeneration is limited
longer loops and larger models remain untested
```

## What To Show In The Report

The report should show:

```text
1. Projection-only failed.
2. Projection + restore fixed the failure.
3. Naive forgot.
4. Restore run learned all target categories.
5. Restore run preserved residual geometry better than naive/joint.
6. The claim is promising but limited: one strong run, not robustness proof.
```
