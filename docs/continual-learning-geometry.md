---
layout: default
title: Continual Learning Is A Geometry Problem
permalink: /continual-learning-geometry/
---

# Continual Learning Is A Geometry Problem

This is a small mechanistic study of continual learning in one fixed-size
transformer. The goal is not to claim a final theory. The goal is to make one
failure mode visible:

```text
learning new text is not just writing new facts into unused weights
```

When the same architecture learns more text from scratch, its internal
representation reorganizes. Rank rises, residual states move, and the old text
gets represented differently even when old behavior remains good. When the
already trained 100-word model is updated afterward, the update either protects
the old geometry and barely learns the new text, or moves strongly and weakens
the old solution.

That points to a harder thesis:

```text
continual learning is a representation reorganization problem
under fixed capacity, not only a protected-write problem
```

## Section Chooser

- [Question]({{ site.baseurl }}/continual-learning-geometry/#question)
- [Shared Model Spec]({{ site.baseurl }}/continual-learning-geometry/#shared-model-spec)
- [The Three Training Paths]({{ site.baseurl }}/continual-learning-geometry/#the-three-training-paths)
- [Path 1: The 100-Word Model]({{ site.baseurl }}/continual-learning-geometry/#path-1-the-100-word-model)
- [Path 2: Same Spec, More Text From Scratch]({{ site.baseurl }}/continual-learning-geometry/#path-2-same-spec-more-text-from-scratch)
- [What From-Scratch Training Appears To Do]({{ site.baseurl }}/continual-learning-geometry/#what-from-scratch-training-appears-to-do)
- [Left-To-Right Capacity Frontier]({{ site.baseurl }}/continual-learning-geometry/#left-to-right-capacity-frontier)
- [Capacity Geometry On The Original 100 Words]({{ site.baseurl }}/continual-learning-geometry/#capacity-geometry-on-the-original-100-words)
- [Capacity Geometry On The Full 500 Words]({{ site.baseurl }}/continual-learning-geometry/#capacity-geometry-on-the-full-500-words)
- [Path 3: Continual Update On Top Of 100 Words]({{ site.baseurl }}/continual-learning-geometry/#path-3-continual-update-on-top-of-100-words)
- [What Training On Top Appears To Do]({{ site.baseurl }}/continual-learning-geometry/#what-training-on-top-appears-to-do)
- [What The Images Show]({{ site.baseurl }}/continual-learning-geometry/#what-the-images-show)
- [Takeaway]({{ site.baseurl }}/continual-learning-geometry/#takeaway)

## Question

The simple story of continual learning is:

```text
old knowledge already exists
new data arrives
find a safe place to write the new data
protect the old knowledge
```

This experiment asks whether that story is enough. If successful learning only
needed a safe write location, then a protected update should be able to add the
second text while keeping the first text fixed. But if successful learning needs
the whole representation to rebase, then preserving the old geometry too rigidly
will block new learning.

The test is deliberately small so the geometry can be inspected.

## Shared Model Spec

Every model in this page uses the same transformer specification:

```text
vocabulary size = 2000
model width     = 128
layers          = 2
attention heads = 4
MLP width       = 256
sequence length = 32
```

The comparison is therefore not bigger model versus smaller model. It is the
same capacity trained under different data histories.

The measurements are:

| measurement | meaning |
|---|---|
| loss | next-token cross-entropy on the training span |
| token accuracy | fraction of next-token predictions that are correct |
| target margin | logit(correct token) minus logit(best competing token) |
| effective rank | how many dimensions the representation uses in practice |
| novelty angle | angular separation between two text spans inside a layer |
| drift | how far the same old text representation moved between models |
| CKA | representational similarity; lower means stronger geometry change |

This is not a held-out generalization test. It is a fixed-capacity fitting and
geometry test: how much text can this exact tiny model absorb, and what happens
to its representation as the amount of text increases?

## The Three Training Paths

The text was split into consecutive spans. The first 100 words are the old
span. Later words are extra data.

The three paths are:

```text
Path 1:
random model -> train on first 100 words -> base100

Path 2:
random model -> train from scratch on 200, 300, 400, or 500 words

Path 3:
base100 -> update afterward on the second 100 words
```

Path 2 is the important reference. It tells us what the model can do when it is
allowed to organize the whole representation from the beginning. Path 3 tells
us what happens when the old solution already exists and the model tries to add
new text afterward.

## Path 1: The 100-Word Model

The 100-word model fits the first span almost perfectly:

| model | final loss | best loss | token accuracy | mean margin | min margin |
|---|---:|---:|---:|---:|---:|
| 100 words | 0.014533 | 0.014110 | 0.9913 | +11.4909 | -1.3982 |

The final residual geometry on the original span is the baseline geometry that
later comparisons use.

![Final residual geometry for the 100-word model on the original 100-word span]({{ site.baseurl }}/assets/continual-geometry/base100-only-100w-final.png)

This is the model after it has already solved the old text. The continual
learning problem starts here: can we add new text without breaking this learned
behavior?

## Path 2: Same Spec, More Text From Scratch

Next, the same architecture was trained from scratch on longer spans: 100, 200,
300, 400, and 500 words.

| training span | final loss | best loss | token accuracy | mean margin | min margin |
|---:|---:|---:|---:|---:|---:|
| 100 words | 0.014533 | 0.014110 | 0.9913 | +11.4909 | -1.3982 |
| 200 words | 0.024659 | 0.024639 | 0.9867 | +11.6670 | -1.2406 |
| 300 words | 0.033142 | 0.033125 | 0.9839 | +11.8152 | -1.2813 |
| 400 words | 0.037832 | 0.037756 | 0.9826 | +12.1580 | -1.7029 |
| 500 words | 0.040771 | 0.040746 | 0.9821 | +12.4945 | -1.6374 |

The important pattern is:

```text
loss rises as more text is packed into the same model
accuracy stays high
mean margin stays strongly positive
```

That means the model is not simply failing. It still predicts most tokens
correctly and confidently. But the loss no longer reaches the same near-zero
level as the 100-word case. A small set of harder or conflicting positions keeps
contributing error while most of the sequence is memorized.

This is why accuracy alone hides the pressure. Accuracy says the model is still
mostly correct. Loss and geometry show that the same fixed capacity is being
repacked. In this page, this is treated as an overfit-like capacity strain, not
ordinary train/test overfitting, because the measurements are on the fitted
training span itself.

## What From-Scratch Training Appears To Do

The from-scratch runs are important because the model is free to organize all
text together. It does not have to protect an old solution. Based on the loss,
accuracy, rank, drift, and CKA measurements, the internal story appears to be:

```text
early text gives the model a first geometry
more text forces the geometry to expand
then the model starts repacking the same space
old states move, but old behavior can remain correct
```

This is not direct proof of the exact circuit mechanism. It is the most
consistent interpretation of the measured geometry.

### Reuse

The 200-to-500 word models are not learning every token with a separate isolated
route. Accuracy stays high while rank rises only up to a limited range and then
plateaus. That suggests the model reuses existing computations: similar contexts
share residual directions, MLP features, and attention routing rather than
allocating a completely new independent direction for every new phrase.

In practical terms:

```text
new text does not only create new directions
it also bends existing useful directions into a broader solution
```

### Rebase

The clearest signal is that the original 100-word representation moves when the
model is trained from scratch on 200, 300, 400, or 500 words. The old text is
still predicted well, but it is not represented in the same place.

That means successful learning can preserve function while changing geometry:

```text
same old behavior
different hidden-state arrangement
```

This is the core reason continual learning is difficult. If the old geometry is
protected too literally, the model may be prevented from finding the rebased
representation that a from-scratch solution would use.

### Dense Compression

As the text length grows, the model keeps fitting most positions, but loss
rises. This looks like dense compression: many token-specific constraints are
being packed into the same finite-dimensional residual space. Most predictions
remain correct, but a smaller number of difficult positions carry more loss.

The capacity frontier therefore looks like:

```text
100 words: enough room for a very sharp fit
200 words: rank expands and the geometry rebases
300-500 words: accuracy remains high, but loss rises under packing pressure
```

This is not clean modular storage. It is shared, dense, and entangled
representation.

### Capacity Packing

The full 500-word probe shows that middle-layer CKA drops hard as more text is
trained into the same architecture. That means the internal basis used by the
model changes substantially. The model is not just adding extra points at the
edge of the old cloud. It is changing the coordinate system of the cloud.

The important distinction is:

```text
append-only writing:
  old geometry stays fixed
  new data goes somewhere else

from-scratch fitting:
  old and new data co-determine the geometry
  old states are allowed to move
  behavior survives because the whole solution is coordinated
```

## Left-To-Right Capacity Frontier

These two strips show the same fixed architecture as the training span grows
from `100 -> 200 -> 300 -> 400 -> 500` words. Read each strip from left to
right. The point is not that the dots form a perfectly separable cluster. The
point is that the whole residual geometry keeps being rearranged while loss
rises and token accuracy stays high.

The first strip probes the original 100-word span through every model. This
shows what happens to the old representation when the model is trained from
scratch on more total text.

![Left-to-right final residual geometry on the original 100-word probe: 100, 200, 300, 400, and 500 word models]({{ site.baseurl }}/assets/continual-geometry/capacity-line-100w-final.png)

The second strip probes the full 500-word span through every model. This shows
the same model family being asked to organize a larger text window, even though
the parameter count and architecture never change.

![Left-to-right final residual geometry on the full 500-word probe: 100, 200, 300, 400, and 500 word models]({{ site.baseurl }}/assets/continual-geometry/capacity-line-500w-final.png)

The visual pattern matches the table above:

```text
loss:     0.0145 -> 0.0247 -> 0.0331 -> 0.0378 -> 0.0408
accuracy: 0.9913 -> 0.9867 -> 0.9839 -> 0.9826 -> 0.9821
```

The model still predicts most positions correctly, but the representation is
not static. The same-size network keeps changing how it uses the residual
space.

## Capacity Geometry On The Original 100 Words

The first visualization probes the original 100-word span through every
from-scratch model. The input is the same old text, but the model has been
trained on more total data.

If the old representation stayed fixed, the rank, drift, and CKA would remain
close to the 100-word model. They do not.

### Original 100-Word Probe: Embedding Layer

![Capacity frontier on the original 100-word span, embedding layer]({{ site.baseurl }}/assets/continual-geometry/capacity-frontier-100w-embed.png)

### Original 100-Word Probe: Block 0

![Capacity frontier on the original 100-word span, transformer block 0]({{ site.baseurl }}/assets/continual-geometry/capacity-frontier-100w-block-0.png)

### Original 100-Word Probe: Block 1

![Capacity frontier on the original 100-word span, transformer block 1]({{ site.baseurl }}/assets/continual-geometry/capacity-frontier-100w-block-1.png)

### Original 100-Word Probe: Final Residual

![Capacity frontier on the original 100-word span, final residual layer]({{ site.baseurl }}/assets/continual-geometry/capacity-frontier-100w-final.png)

On the original 100-word probe, the representation changes substantially as
more text is included in training.

| layer | 100-word rank | 200-word rank | 500-word rank | 500-word drift | 500-word CKA |
|---|---:|---:|---:|---:|---:|
| embed | 91.87 | 100.92 | 100.55 | 1.3156 | 0.5632 |
| block 0 | 62.74 | 82.45 | 81.65 | 1.7761 | 0.3305 |
| block 1 | 72.48 | 84.40 | 81.23 | 2.2986 | 0.4151 |
| final | 74.32 | 87.87 | 85.02 | 0.8829 | 0.3705 |

The old text is still handled well, but the internal geometry used to handle it
is no longer the same geometry. The model trained on more text has rebased even
the original span.

## Capacity Geometry On The Full 500 Words

The second visualization probes the full 500-word span through each model. The
colors split the early part of the text from the later part, so the plots show
how the same architecture organizes a larger input span as the training budget
increases.

### Full 500-Word Probe: Embedding Layer

![Capacity frontier on the full 500-word span, embedding layer]({{ site.baseurl }}/assets/continual-geometry/capacity-frontier-500w-embed.png)

### Full 500-Word Probe: Block 0

![Capacity frontier on the full 500-word span, transformer block 0]({{ site.baseurl }}/assets/continual-geometry/capacity-frontier-500w-block-0.png)

### Full 500-Word Probe: Block 1

![Capacity frontier on the full 500-word span, transformer block 1]({{ site.baseurl }}/assets/continual-geometry/capacity-frontier-500w-block-1.png)

### Full 500-Word Probe: Final Residual

![Capacity frontier on the full 500-word span, final residual layer]({{ site.baseurl }}/assets/continual-geometry/capacity-frontier-500w-final.png)

On the full 500-word probe, the geometry shows stronger capacity pressure.

| layer | 100-word rank | 200-word rank | 500-word rank | 500-word drift | 500-word CKA |
|---|---:|---:|---:|---:|---:|
| embed | 90.23 | 104.00 | 109.40 | 1.8685 | 0.5789 |
| block 0 | 65.99 | 90.36 | 87.48 | 2.7447 | 0.1214 |
| block 1 | 74.82 | 90.23 | 84.60 | 3.4766 | 0.1461 |
| final | 78.64 | 95.86 | 91.16 | 1.1968 | 0.0848 |

The first jump, from 100 to 200 words, expands the used representation strongly.
After that, rank does not keep increasing linearly. The model keeps fitting more
text, but the middle-layer geometry drifts heavily and CKA collapses.

This is the visible capacity frontier:

```text
more data does not just occupy more empty space
the existing representation is repacked
```

## Path 3: Continual Update On Top Of 100 Words

The last experiment starts from the already trained 100-word model and tries to
learn the second 100 words afterward.

The from-scratch 200-word model is the target behavior: it learns both spans.
The continual updates are the hard case: they have to add the new span while
preserving old behavior.

### Behavior On The Old 100 Words

| model | old loss | old accuracy | old mean margin | old min margin |
|---|---:|---:|---:|---:|
| 100-word base | 0.014533 | 0.9913 | +11.4909 | -1.3982 |
| 200-word from scratch | 0.023803 | 0.9881 | +11.8631 | -1.2406 |
| protected continual update | 0.014550 | 0.9913 | +11.3884 | -1.4061 |
| aggressive continual update | 0.031328 | 0.9901 | +9.0665 | -3.7476 |

### Behavior On The New 100 Words

| model | new loss | new accuracy | new mean margin | new min margin |
|---|---:|---:|---:|---:|
| 100-word base | 13.628014 | 0.0549 | -12.8071 | -30.5011 |
| 200-word from scratch | 0.024992 | 0.9863 | +11.5536 | -0.9081 |
| protected continual update | 12.466713 | 0.0661 | -11.4853 | -30.3192 |
| aggressive continual update | 10.083992 | 0.0614 | -9.0928 | -20.6879 |

The protected update preserves old behavior but barely learns the new text. The
aggressive update learns more than the protected update, but still fails badly
on the new span and weakens the old solution.

### Successful Reorganization: 100-Word Model Versus 200-Word Model

![Natural representation rebasing from the 100-word model to the 200-word model]({{ site.baseurl }}/assets/continual-geometry/natural-rebasing-100w-final.png)

This is the successful case. It is not a continual update; it is a fresh
200-word training run. The old text is represented differently, but behavior is
preserved because the whole solution was allowed to organize together.

### Protected Continual Update

![Protected continual update drift on the original 100-word span]({{ site.baseurl }}/assets/continual-geometry/safe-update-100w-final.png)

The protected update keeps the original geometry close enough that old behavior
stays stable. But it also fails to create the large representational change
needed to model the new span.

### Aggressive Continual Update

![Aggressive continual update drift on the original 100-word span]({{ site.baseurl }}/assets/continual-geometry/aggressive-update-100w-final.png)

The aggressive update moves the geometry more, but the movement is not the same
as the successful from-scratch reorganization. It damages old margins and still
does not solve the new text.

## What Training On Top Appears To Do

Training on top of the solved 100-word model behaves differently from training
from scratch because the first geometry is already load-bearing. The parameters
are no longer blank capacity. They already implement a sharp solution for the
old span.

The update now faces a conflict:

```text
move enough to learn the new span
but not so much that the old span loses its margins
```

The experiments show two bad regimes.

### Protected Update: Stable But Underpowered

The protected update keeps old behavior nearly unchanged:

```text
old loss:     0.014533 -> 0.014550
old accuracy: 0.9913   -> 0.9913
old margin:   11.4909  -> 11.3884
```

But the new text barely improves:

```text
new loss:     13.6280 -> 12.4667
new accuracy: 0.0549  -> 0.0661
```

Mechanistically, this looks like preserving the old coordinate system too
strongly. The update can make small local changes, but it cannot perform the
larger rebasing that the from-scratch 200-word model found.

### Aggressive Update: Movement Without Coordination

The aggressive update moves more and reduces new loss further:

```text
new loss: 13.6280 -> 10.0840
```

But it weakens old behavior:

```text
old loss:   0.014533 -> 0.031328
old margin: 11.4909  -> 9.0665
```

It also still fails to learn the new span well. This is the key part: moving the
old representation is not enough. The movement must be coordinated. The
from-scratch model changes old and new geometry together. The aggressive
continual update pushes on an already formed geometry and creates drift without
finding the better joint solution.

### Why This Is More Than A Write Problem

The failure is not only that the update chose the wrong individual weights. The
problem is that the model likely needs a new shared representation:

```text
reuse old features where they help
move old states where the global geometry requires it
compress overlapping structure into shared directions
separate conflicting contexts where reuse would cause errors
```

Training from scratch can do all of that implicitly because all constraints are
present together. Continual updating has to do it while old behavior is already
installed. That is a harder control problem than a local write.

## What The Images Show

The three paths separate three different effects.

### 1. Fitting More Text Uses More Representational Degrees Of Freedom

The 100-to-500 frontier shows a rise in effective rank:

```text
final layer rank on original 100-word probe:
74.32 -> 87.87 -> 87.38 -> 83.89 -> 85.02

final layer rank on full 500-word probe:
78.64 -> 95.86 -> 94.10 -> 90.91 -> 91.16
```

The model opens more representational directions early, then starts repacking.
That is why rank rises sharply at first and then plateaus or slightly drops.

### 2. Accuracy Can Stay High While Loss Reveals Strain

From 100 to 500 words:

```text
accuracy: 0.9913 -> 0.9821
loss:     0.0145 -> 0.0408
```

Accuracy falls only slightly. Loss almost triples. The model still gets most
tokens right, but the harder positions remain less perfectly fitted. This is a
capacity-frontier signal, not a simple accuracy failure.

### 3. Successful Learning Rebases Old Geometry

On the original 100-word probe, the 500-word model is not using the same hidden
geometry as the 100-word model:

```text
block 1 drift relative to 100-word model: 2.2986
block 1 CKA relative to 100-word model:   0.4151
final drift relative to 100-word model:   0.8829
final CKA relative to 100-word model:     0.3705
```

The old behavior is not preserved by freezing the old representation. It is
preserved by finding a different representation that supports more data.

### 4. Continual Updating Does Not Automatically Find That Reorganization

The protected update is too conservative:

```text
old accuracy stays: 0.9913 -> 0.9913
new loss only moves: 13.6280 -> 12.4667
```

The aggressive update is too destructive:

```text
old margin drops: 11.4909 -> 9.0665
new loss improves: 13.6280 -> 10.0840
new accuracy remains poor: 0.0614
```

Neither path recreates the coordinated geometry of the from-scratch 200-word
model.

## Takeaway

The useful conclusion is not that protected writing is useless. Protection is
necessary when old behavior matters. The useful conclusion is that protection
alone is not enough.

A fixed-size model that learns more data from scratch appears to do three
things together:

```text
increase the used representational rank
move old residual states into a new arrangement
preserve behavior through coordinated reorganization
```

A continual learner must somehow do the same thing after old behavior already
exists:

```text
preserve the function
while allowing the representation to rebase
```

That is much harder than choosing a safe write location. It suggests that the
core problem is not only:

```text
where can the new data be written?
```

It is also:

```text
how can the whole representation reorganize
without losing the old function?
```

This is the next problem to study.
