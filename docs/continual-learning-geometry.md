---
layout: default
title: Continual Learning Is A Geometry Problem
permalink: /continual-learning-geometry/
---

# Continual Learning Is A Geometry Problem

This is a small mechanistic case study, not a final claim about all
continual-learning systems. The goal is to look inside one tiny transformer and
ask a narrow question:

```text
When more data is learned, does the model merely write new facts somewhere,
or does the internal representation reorganize?
```

The short answer from this experiment is:

```text
successful extra-data learning rebases the representation
```

The model that learns both texts from scratch does not simply append the second
text. It increases effective rank, moves old residual states, and changes the
geometry of the original text while preserving behavior. That makes continual
learning harder than a simple write-location problem.

## Section Chooser

- [Setup]({{ site.baseurl }}/continual-learning-geometry/#setup)
- [The Three Training Paths]({{ site.baseurl }}/continual-learning-geometry/#the-three-training-paths)
- [Behavior Results]({{ site.baseurl }}/continual-learning-geometry/#behavior-results)
- [Geometry: The 200-Word Model Uses More Space]({{ site.baseurl }}/continual-learning-geometry/#geometry-the-200-word-model-uses-more-space)
- [Matched Drift: The Old Text Moves Too]({{ site.baseurl }}/continual-learning-geometry/#matched-drift-the-old-text-moves-too)
- [Continual Updates: Safe Drift And Damaging Drift]({{ site.baseurl }}/continual-learning-geometry/#continual-updates-safe-drift-and-damaging-drift)
- [What The Experiment Suggests]({{ site.baseurl }}/continual-learning-geometry/#what-the-experiment-suggests)
- [What This Does Not Prove]({{ site.baseurl }}/continual-learning-geometry/#what-this-does-not-prove)

## Setup

All runs used the same tiny transformer specification:

```text
vocabulary size = 2000
model width     = 128
layers          = 2
attention heads = 4
MLP width       = 256
sequence length = 32
```

Only the training path changed. This matters because the experiment is not
comparing a larger model against a smaller model. It is comparing how the same
architecture behaves when it learns the data in different orders.

The text source was split into two consecutive spans:

```text
old span = first 100 words
new span = second 100 words
full span = first 200 words
```

The measurements use next-token language-model loss, token accuracy, target
margin, residual-state geometry, and matched residual drift.

Target margin means:

```text
target margin = logit(correct token) - logit(best competing token)
```

Large positive margin means the model strongly prefers the correct next token.
Negative margin means some other token is preferred.

## The Three Training Paths

There are three different cases.

### 1. The 100-Word Base Model

The first model starts from random initialization and trains only on the first
100 words:

```text
random tiny transformer -> train on first 100 words -> base100
```

It learns the first span very well:

```text
loss: 7.7006 -> 0.0145
accuracy: 0.9913
mean target margin: +11.4909
```

This model gives us the old learned representation.

![Final residual geometry for the 100-word model on the original 100-word span]({{ site.baseurl }}/assets/continual-geometry/base100-only-100w-final.png)

This plot is the starting geometry for the later comparisons. It shows how the
trained 100-word model arranges residual states from the original text before
any second-span update is attempted.

### 2. Continual Updates From The 100-Word Model

The second case starts from the already trained 100-word model and then tries
to learn the second 100 words:

```text
base100 -> update on second 100 words -> continual-update model
```

This is the continual-learning setting. The model already has an old solution,
and the new update must add new data without damaging the old behavior.

Two update regimes were tested:

```text
safe continual update:
  small protected writes
  old behavior remains stable
  new text is barely learned

aggressive continual update:
  larger rewiring-style changes
  new loss improves more
  old behavior is damaged
  new text is still not learned well
```

### 3. The 200-Word Model From Scratch

The third model starts from random initialization and trains on both spans
together:

```text
random tiny transformer -> train on first 200 words -> base200
```

It sees old and new text together from the beginning. It does not need to
preserve a previously formed internal solution.

It also learns the data well:

```text
loss: 7.6761 -> 0.0247
accuracy: 0.9867
mean target margin on old span: +11.8631
mean target margin on new span: +11.5536
```

This model is important because it shows what a successful same-capacity
solution looks like when the model is allowed to organize freely.

## Behavior Results

The behavior table already shows the core tension.

### Old 100 Words

| model | loss | accuracy | mean margin | min margin |
|---|---:|---:|---:|---:|
| base100 | 0.014533 | 0.9913 | +11.4909 | -1.3982 |
| base200 | 0.023803 | 0.9881 | +11.8631 | -1.2406 |
| safe continual update | 0.014550 | 0.9913 | +11.3884 | -1.4061 |
| aggressive continual update | 0.031328 | 0.9901 | +9.0665 | -3.7476 |

The safe continual update preserves the old span almost perfectly. The
aggressive continual update still has high token accuracy, but the margin drops
from `+11.4909` to `+9.0665`. That is a real weakening of the old solution.

### New 100 Words

| model | loss | accuracy | mean margin | min margin |
|---|---:|---:|---:|---:|
| base100 | 13.628014 | 0.0549 | -12.8071 | -30.5011 |
| base200 | 0.024992 | 0.9863 | +11.5536 | -0.9081 |
| safe continual update | 12.466713 | 0.0661 | -11.4853 | -30.3192 |
| aggressive continual update | 10.083992 | 0.0614 | -9.0928 | -20.6879 |

The successful from-scratch 200-word model learns the new span. The continual
updates do not. The safe update barely moves enough to learn the new text. The
aggressive update improves the loss more, but not nearly enough, and it weakens
old behavior.

This is the central observation:

```text
protecting the old representation preserves old behavior
but does not create the successful new representation
```

## Geometry: The 200-Word Model Uses More Space

The next question is what changed internally.

The following plots show residual-state geometry. Each dot is a residual state
from a token position inside a sliding text window. The two colors split the
first and second text spans. The panels compare the 100-word model and the
200-word model under the same probe.

![Embedding geometry for the 100-word and 200-word models]({{ site.baseurl }}/assets/continual-geometry/base100-vs-base200-200w-embed.png)

At the embedding level, the 200-word model already shows higher effective rank:

```text
embedding layer
base100 rank = 92.18
base200 rank = 106.49
```

The more important changes appear inside the transformer blocks.

![Block 0 residual geometry for the 100-word and 200-word models]({{ site.baseurl }}/assets/continual-geometry/base100-vs-base200-200w-block-0.png)

```text
block 0
base100 rank = 64.84
base200 rank = 87.43
base100 novelty outside old span = 0.2569
base200 novelty outside old span = 0.3206
```

![Block 1 residual geometry for the 100-word and 200-word models]({{ site.baseurl }}/assets/continual-geometry/base100-vs-base200-200w-block-1.png)

```text
block 1
base100 rank = 74.52
base200 rank = 88.94
base100 novelty outside old span = 0.2541
base200 novelty outside old span = 0.3294
```

The final residual geometry shows the same pattern:

![Final residual geometry for the 100-word and 200-word models]({{ site.baseurl }}/assets/continual-geometry/base100-vs-base200-200w-final.png)

```text
final layer
base100 rank = 77.74
base200 rank = 92.96
base100 novelty outside old span = 0.2801
base200 novelty outside old span = 0.3467
base100 principal angle mean = 13.45 degrees
base200 principal angle mean = 19.13 degrees
```

The 200-word model uses a higher-rank representation and separates the two
text spans more strongly. This means the added data is not merely placed into a
small local patch of the old representation. The model reorganizes its residual
space.

The same architecture comparison can also be viewed only on the original
100-word span. This isolates what happened to the old text itself when the
200-word model learned both spans from scratch.

![Embedding geometry for both models on the original 100-word span]({{ site.baseurl }}/assets/continual-geometry/base100-vs-base200-100w-embed.png)

```text
original 100 words, embedding layer
base100 rank = 91.87
base200 rank = 100.92
base100 novelty = 0.4020
base200 novelty = 0.5096
```

![Block 0 residual geometry for both models on the original 100-word span]({{ site.baseurl }}/assets/continual-geometry/base100-vs-base200-100w-block-0.png)

```text
original 100 words, block 0
base100 rank = 62.74
base200 rank = 82.45
base100 novelty = 0.3265
base200 novelty = 0.4002
```

![Block 1 residual geometry for both models on the original 100-word span]({{ site.baseurl }}/assets/continual-geometry/base100-vs-base200-100w-block-1.png)

```text
original 100 words, block 1
base100 rank = 72.48
base200 rank = 84.40
base100 novelty = 0.3606
base200 novelty = 0.4073
```

![Final residual geometry for both models on the original 100-word span]({{ site.baseurl }}/assets/continual-geometry/base100-vs-base200-100w-final.png)

```text
original 100 words, final layer
base100 rank = 74.32
base200 rank = 87.87
base100 novelty = 0.3726
base200 novelty = 0.4478
base100 principal angle mean = 21.12 degrees
base200 principal angle mean = 27.06 degrees
```

Even when the probe contains only the original text, the 200-word model uses a
higher-rank geometry. The old text is not represented exactly the same way.

## Matched Drift: The Old Text Moves Too

The previous plots show different geometries, but they do not answer the most
important question:

```text
What happened to the original 100-word representation?
```

To test that, both models were evaluated on the exact same original 100-word
span. The 200-word model's residual states were aligned into the 100-word
model's coordinates with an orthogonal Procrustes map. The arrows show how
matched residual points moved after alignment.

![Embedding matched drift from the 100-word model to the 200-word model]({{ site.baseurl }}/assets/continual-geometry/natural-rebasing-100w-embed.png)

For the embedding layer:

```text
rank: 91.87 -> 100.92
relative matched drift: 0.6624
CKA: 0.6021
aligned cosine: 0.7409
centroid shift: 0.4336
```

![Final residual matched drift from the 100-word model to the 200-word model]({{ site.baseurl }}/assets/continual-geometry/natural-rebasing-100w-final.png)

For the final residual layer:

```text
rank: 74.32 -> 87.87
relative matched drift: 0.7880
CKA: 0.4906
aligned cosine: 0.6783
centroid shift: 1.5824
```

CKA means centered kernel alignment. It measures representational similarity.
The value `0.4906` is only moderate. The relative matched drift of `0.7880`
means the old residual states moved by a large fraction of their natural scale.

The same movement appears inside the transformer blocks:

![Block 0 matched drift from the 100-word model to the 200-word model]({{ site.baseurl }}/assets/continual-geometry/natural-rebasing-100w-block-0.png)

```text
block 0
rank: 62.74 -> 82.45
relative matched drift: 0.7960
CKA: 0.3936
aligned cosine: 0.6240
```

![Block 1 matched drift from the 100-word model to the 200-word model]({{ site.baseurl }}/assets/continual-geometry/natural-rebasing-100w-block-1.png)

```text
block 1
rank: 72.48 -> 84.40
relative matched drift: 0.7351
CKA: 0.5328
aligned cosine: 0.6728
```

This is the key result:

```text
the successful 200-word model does not preserve exact old residual coordinates
```

It preserves behavior while changing the internal coordinate system. The old
text still works, but its representation has been rebased.

That is why the continual-learning problem cannot be reduced to freezing exact
activation anchors. If the old coordinates are protected too rigidly, the model
may be blocked from making the same kind of successful reorganization seen in
the from-scratch 200-word solution.

## Continual Updates: Safe Drift And Damaging Drift

Now compare that natural rebasing with the two continual-update attempts.

The safe continual update preserves the old geometry very closely:

![Embedding drift after the safe continual update]({{ site.baseurl }}/assets/continual-geometry/safe-update-100w-embed.png)

```text
safe continual update, embedding layer
rank: 91.87 -> 91.91
relative matched drift: 0.0556
CKA: 0.9977
aligned cosine: 0.9983
```

![Block 0 drift after the safe continual update]({{ site.baseurl }}/assets/continual-geometry/safe-update-100w-block-0.png)

```text
safe continual update, block 0
rank: 62.74 -> 62.68
relative matched drift: 0.0669
CKA: 0.9949
aligned cosine: 0.9969
```

![Block 1 drift after the safe continual update]({{ site.baseurl }}/assets/continual-geometry/safe-update-100w-block-1.png)

```text
safe continual update, block 1
rank: 72.48 -> 72.38
relative matched drift: 0.0769
CKA: 0.9936
aligned cosine: 0.9961
```

![Final residual drift after the safe continual update]({{ site.baseurl }}/assets/continual-geometry/safe-update-100w-final.png)

```text
safe continual update, final layer
rank: 74.32 -> 74.18
relative matched drift: 0.0779
CKA: 0.9930
aligned cosine: 0.9961
old loss: 0.014533 -> 0.014550
new loss: 13.628014 -> 12.466713
```

The old model is preserved, but the new text is barely learned.

The aggressive continual update moves more:

![Embedding drift after the aggressive continual update]({{ site.baseurl }}/assets/continual-geometry/aggressive-update-100w-embed.png)

```text
aggressive continual update, embedding layer
rank: 91.87 -> 93.04
relative matched drift: 0.1106
CKA: 0.9650
aligned cosine: 0.9932
```

![Block 0 drift after the aggressive continual update]({{ site.baseurl }}/assets/continual-geometry/aggressive-update-100w-block-0.png)

```text
aggressive continual update, block 0
rank: 62.74 -> 63.29
relative matched drift: 0.1180
CKA: 0.9772
aligned cosine: 0.9894
```

![Block 1 drift after the aggressive continual update]({{ site.baseurl }}/assets/continual-geometry/aggressive-update-100w-block-1.png)

```text
aggressive continual update, block 1
rank: 72.48 -> 73.16
relative matched drift: 0.1383
CKA: 0.9755
aligned cosine: 0.9871
```

![Final residual drift after the aggressive continual update]({{ site.baseurl }}/assets/continual-geometry/aggressive-update-100w-final.png)

```text
aggressive continual update, final layer
rank: 74.32 -> 74.87
relative matched drift: 0.1348
CKA: 0.9760
aligned cosine: 0.9872
old loss: 0.014533 -> 0.031328
old mean margin: +11.4909 -> +9.0665
new loss: 13.628014 -> 10.083992
```

This update does learn slightly more about the new text, but it weakens the
old solution and still does not produce the successful 200-word behavior.

The contrast is sharp:

| case | old final-layer drift | final-layer CKA | old behavior | new behavior |
|---|---:|---:|---|---|
| safe continual update | 0.0779 | 0.9930 | preserved | barely learned |
| aggressive continual update | 0.1348 | 0.9760 | weakened | still poor |
| 200-word from scratch | 0.7880 | 0.4906 | still good | learned well |

The successful solution is not just “more drift.” It is a different kind of
drift: a large, coordinated representational rebasing that keeps behavior
working.

## What The Experiment Suggests

The naive write framing is:

```text
old knowledge is stored somewhere
new knowledge must be written somewhere else
```

This experiment suggests a more difficult picture:

```text
old and new behavior may need a shared rebasing of the representation
```

The 200-word model did not simply add a second memory next to the first. It
changed the basis used by the first memory too.

The main findings are:

1. **More data increased effective rank.** The 200-word model used more
   representational degrees of freedom than the 100-word model.
2. **Successful learning moved old states.** The original 100-word residual
   states moved substantially in the successful 200-word model.
3. **Small protected updates were too rigid.** They preserved old behavior and
   old geometry but failed to learn the new span.
4. **Aggressive local updates were not enough.** They damaged old behavior
   without reproducing the useful global reorganization.
5. **The core problem is behavior-preserving rebasing.** Continual learning
   needs a way to move representations coherently while preserving old
   behavior.

A better mechanistic target is therefore:

```text
preserve useful behavior and useful relations
while allowing internal coordinates to change
```

That is different from:

```text
freeze old activations exactly
```

It is also different from:

```text
freely fine-tune and hope old behavior survives
```

The real object to preserve may be relational and behavioral: margins, causal
routes, readout compatibility, and the ability to decode old behavior after the
representation has moved.

## What This Does Not Prove

This experiment is intentionally small. It does not prove a general theory of
continual learning.

The limitations are:

- one tiny transformer architecture;
- one short text source;
- one main seed for the reported comparison;
- a near-memorization regime rather than a broad benchmark;
- PCA and Procrustes views are diagnostics, not complete mechanistic proofs.

Still, the result is useful because the control is clean:

```text
same architecture
same width
same layer count
same tokenizer
same text source
different training path
```

Under that control, successful from-scratch learning of more data caused large
representational rebasing, while continual updates either preserved old
geometry too strongly or damaged old behavior without learning the new text.

The next mechanistic question is:

```text
Can a model update old and new representations together,
so old behavior survives while the internal geometry is allowed to rebase?
```
