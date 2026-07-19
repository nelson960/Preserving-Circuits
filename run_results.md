# Continual Learning Evidence Ledger

Updated: 2026-06-20

## Purpose

This file records only experiments that directly support or constrain the
current architecture. It is intentionally not a chronological notebook. Failed
branches and superseded hypotheses remain in `docs/archive/`.

## Evidence Chain

| stage | experiment | result | role in current architecture |
|---:|---|---|---|
| 1 | Same-spec 100/200-word geometry | More-data training reorganized old geometry while preserving behavior | Established geometry as a CL problem |
| 2 | Behavior-preserving path | Learned new text while retaining old output behavior | Showed a protected path can exist |
| 3 | Invariant-Tangent plus restore | Projection plus restore succeeded where projection-only failed | Defined the weight-update operator |
| 4 | Controlled preserve/drop/guard | Selective suppression worked with neutral protection | Established controlled forgetting requirements |
| 5 | Dynamic committed anchors | Newly learned behavior survived later stages | Established recursive commit |
| 6 | Autonomous trace field | Merge, branch, novelty, noise rejection, and release emerged under fixed slots | Defined evidence organization |
| 7 | Recurrent trace summary | Five bounded summaries approximated full history | Established bounded memory feasibility |
| 8 | Functional dependency field | Output, geometry, and feature-family sensitivity produced a compressed protection basis | Connected traces to parameter directions |
| 9 | Integrated delayed autonomous CL | Sparse novelty matured, obsolete behavior was replaced, noise was ignored, geometry survived | First complete toy loop |
| 10 | Real-text dependency bridge | Preservation improved but correction learning failed | Current transformer boundary |

## Current Primary Run

Artifact:

```text
research/02_trace_dependency_plasticity/results/gco-tiny-autonomous-delayed-dependency-invariant-cl-seed0/
  autonomous_dependency_invariant_cl.json
  autonomous_behavior.png
  autonomous_trace_lifecycle.png
  autonomous_geometry.png
  autonomous_capacity_dependency.png
```

Protocol:

| property | value |
|---|---:|
| model parameters | 246 |
| trace slots | 5 |
| online singleton events | 62 |
| persistent trace scalars | 155 |
| hidden training roles | none |
| seed | 0 |

Final behavior:

| group | model MSE | trace error | dominant slot | dominant share |
|---|---:|---:|---:|---:|
| branch down | 0.00979 | 0.16160 | 2 | 1.0000 |
| branch root | 0.00218 | 0.20171 | 2 | 0.9067 |
| branch up | 0.00529 | 0.04078 | 4 | 1.0000 |
| merge A | 0.00172 | 0.03516 | 1 | 1.0000 |
| merge B | 0.00254 | 0.04310 | 1 | 1.0000 |
| noise | 0.37029 | 1.83054 | 3 | 0.3308 |
| novel replacement | 0.00330 | 0.06860 | 3 | 1.0000 |
| obsolete | 0.29352 | 1.23218 | 3 | 0.9988 |
| stable | 0.00227 | 0.05597 | 0 | 1.0000 |

Geometry and capacity:

| metric | value |
|---|---:|
| hidden CKA | 0.999385 |
| hidden relative drift | 0.049031 |
| pair-geometry drift | 0.013072 |
| final pending fraction | 0.675509 |
| dependency retained rank | 40–48 |
| mean raw gradient fraction removed | 0.898809 |
| mean final/raw predicted damage | 0.334465 |
| maximum final/raw predicted damage | 1.326809 |

Delayed novel maturation:

| novel occurrence | write | support | verified gain |
|---:|---:|---:|---:|
| 1 | 0.000019 | 0.000019 | 0.000006 |
| 2 | 0.007818 | 0.007818 | 0.025128 |
| 3 | 0.059440 | 0.059440 | 0.213182 |
| 7 | 0.246755 | 0.246755 | 0.418956 |
| 13 | 0.600316 | 0.600316 | 0.307384 |
| 16 | 0.547681 | 0.547681 | 0.290246 |

Noise boundary:

| metric | value |
|---|---:|
| maximum isolated-noise write | 0.000336 |
| maximum isolated-noise verified gain | 0.000440 |

Interpretation:

```text
delayed recurrence created increasing permission to learn
verified gain converted successful writes into learned mass
novel evidence repurposed the obsolete trace region
isolated noise remained effectively unwritten
stable and branched functions survived
trace storage remained fixed
```

Primary limitation exposed by this run:

```text
pending fraction reached 67.55%
```

The next run must determine whether pending pressure reaches equilibrium or
eventually dilutes verified memory.

## Foundational Invariant-Tangent Evidence

Historical detailed report:

```text
docs/archive/invariant_tangent_early_results.md
```

Primary artifacts:

```text
research/01_invariant_tangent/results/colab-loss-baseline/colab-loss-baseline.json
research/01_invariant_tangent/results/colab-plasticity-audit/colab-plasticity-audit.json
research/01_invariant_tangent/results/colab-plasticity-audit2/colab-plasticity-audit2.json
research/01_invariant_tangent/results/colab-rich-invariant/colab-rich-invariant.json
```

The important result was not that projection alone worked. It did not.
Projection plus bounded restore preserved old/guard behavior and learned the
controlled target categories in the successful toy run.

| mechanism | preserve | guard | changed | new | composition | obsolete retained |
|---|---:|---:|---:|---:|---:|---:|
| naive sequential | 0.000 | 0.000 | 0.000 | 0.300 | 0.167 | 0.000 |
| projection only | 0.000 | 0.000 | 0.000 | 0.300 | 0.167 | 0.000 |
| projection plus restore | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

This result established the update operator but relied on externally defined
behavior roles. The autonomous trace work later removed those roles from the
learning mechanism.

## Bounded Evidence Results

### Autonomous Trace Field

Artifact:

```text
research/02_trace_dependency_plasticity/results/gco-tiny-autonomous-trace-field-seed0/trace_field.json
```

Supported structure checks:

- duplicate sources merged;
- contextual branches remained distinct;
- the shared root remained compositional;
- recurring novelty replaced released obsolete structure;
- stable evidence survived;
- isolated noise received no exclusive slot.

### Recurrent Summary

Artifact:

```text
research/02_trace_dependency_plasticity/results/gco-tiny-recurrent-trace-field-seed0/recurrent_trace_field.json
```

| metric | full history | recurrent summary | current only |
|---|---:|---:|---:|
| final MSE | 1.0499 | 1.1052 | 7.4637 |
| retained-old error | 0.5040 | 0.5799 | 8.3803 |
| persistent scalars | 1736 | 365 | 0 |

Recurrent versus full attention CKA was `0.9984`, with `4.76x` storage
compression.

### Covariance Compression

Artifact:

```text
research/02_trace_dependency_plasticity/results/gco-tiny-trace-covariance-sweep-seed0/trace_covariance_sweep.json
```

Full covariance, diagonal, variance trace, and tested low-rank summaries had the
same result in this toy stream. The variance-trace representation used `50`
scalars versus `365` for full covariance. This does not prove covariance is
unnecessary in richer streams.

## Trace-To-Weight Integration

### Initial Trace/Invariant Bridge

Artifact:

```text
research/02_trace_dependency_plasticity/results/gco-tiny-trace-invariant-integration-seed0/trace_invariant_integration.json
```

The bridge showed that trace-weighted loss, invariant projection, and bounded
restore could preserve retained behavior while learning novelty. It did not yet
derive a compact dependency basis.

### Functional Dependency Field

Artifact:

```text
research/02_trace_dependency_plasticity/results/gco-tiny-functional-dependency-field-seed0/functional_dependency_field.json
```

The dependency field combined direct output rows, pair-geometry rows, and
hidden feature-family directions. In the toy run it matched behavior protection
while reducing geometry drift relative to the direct trace-invariant basis.

## Real-Text Boundary Run

Artifact:

```text
research/03_semantic_reasoning/results/gco-tiny-text-dependency-cl-answer-balanced-seed0/text_dependency_cl.json
```

| method | stable loss | new-book loss | novel loss | corrected loss | new over old | CKA |
|---|---:|---:|---:|---:|---:|---:|
| naive | 0.5865 | 4.8440 | 1.3344 | 0.6617 | 0.3750 | 0.7334 |
| trace loss mix | 0.1860 | 6.2399 | 1.3653 | 0.6747 | 0.3750 | 0.8329 |
| trace invariant | 0.1771 | 6.2998 | 1.3265 | 0.6769 | 0.3750 | 0.8394 |
| dependency field | 0.1785 | 6.2793 | 1.3487 | 0.6633 | 0.3750 | 0.8356 |

Interpretation:

```text
protection and geometry retention improved
new-book plasticity weakened
corrected answers did not defeat obsolete answers
```

This run is a boundary/failure result, not evidence that real-text CL is solved.

## Reproduction

Current integrated delayed run:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
python experiments/gco_math/gco_tiny_autonomous_dependency_invariant_cl.py \
  --device mps
```

Default output:

```text
research/02_trace_dependency_plasticity/results/gco-tiny-autonomous-delayed-dependency-invariant-cl-seed0
```

## Excluded From The Active Claim

The following categories were useful exploration but do not independently
support the current claim:

- heuristic role classifiers;
- consequence-action classifiers that collapsed to majority actions;
- probabilistic plasticity mechanisms that collapsed to always-write;
- recurrent controllers whose role accuracy did not produce learning;
- routing and adapter variants superseded by the trace/dependency formulation;
- large chronological command dumps without a stable interpretation.

Their historical context is retained in `docs/archive/`, but they should not be
presented as components of the current architecture.
