# Preserving Circuits

Research on continual learning through mechanistic circuit preservation.

## Research Goal

I am studying whether catastrophic forgetting can be reduced by controlling how optimizer updates move latent concept representations, instead of only optimizing loss in parameter space.

The core question:

> Can an optimizer learn where to write new knowledge while preserving old circuits?

## Main Idea

Track old concepts as latent subspaces across layers and checkpoints, then measure how candidate updates move those subspaces.

The guiding approximation is:

```text
representation change ~= hidden-state Jacobian * parameter update
```

If an update helps a new concept but drifts old concepts too much, it should be gated, projected, or rejected.

## How I Will Do It

1. Map concept geometry from activations using SVD/PCA, CKA, probes, and subspace overlap.
2. Track circuit survival through routing, write, value-code, and readout measurements.
3. Measure one-step representation drift from candidate optimizer updates.
4. Compare drift against actual forgetting.
5. Build a blockwise write-gated optimizer that protects old concept subspaces.
6. Test against Adam/SGD, EWC, HAT, GPM, frozen-backbone, and adapter baselines.

## Project Layout

- `docs/index.md`: living public research proposal.
- `docs/`: research notes, gap analysis, and proposal drafts.
- `paper/`: future paper artifacts.

## Success Criteria

The research is useful if:

- representation drift predicts forgetting better than parameter distance alone;
- write-gated updates reduce forgetting compared to normal fine-tuning;
- new concepts are learned without corrupting old concept subspaces;
- the method works without expanding model capacity.
