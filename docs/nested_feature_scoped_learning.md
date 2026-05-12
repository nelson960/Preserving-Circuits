# Research Proposal

## Title

**Nested Latent-Tangent Write Attention for Continual Learning**  
**A fixed-capacity optimizer framework that learns where to write new knowledge by tracking latent concept geometry and update-induced representation drift**

---

## 1. Executive Summary

Modern neural networks learn new information by applying global optimizer updates to many parameters at once. This is powerful, but it creates a major failure mode in continual learning: when a model learns new information, it can unintentionally overwrite older knowledge. This is known as **catastrophic forgetting**. Existing continual-learning methods reduce forgetting through replay, regularization, parameter isolation, task masks, adapters, or gradient projection, but they still do not fully answer the central question:

> **How should an optimizer know where new knowledge should be written inside a network?**

This proposal investigates a new answer:

> A continual-learning optimizer should not decide updates only from gradients in parameter space. It should decide updates by measuring how candidate parameter updates move latent concept representations across layers.

The proposed framework is called:

# Nested Latent-Tangent Write Attention

The core idea is to connect three spaces that are usually studied separately:

1. **Input/embedding space**  
   How data enters the model through token/image embeddings and positional encodings.

2. **Latent representation space**  
   How concepts such as “cat,” “dog,” “fur,” “motion,” or “object” appear and move through hidden layers.

3. **Tangent/update space**  
   How optimizer updates change the model’s function and hidden representations.

The central mathematical bridge is:

\[
\Delta h_{l}(x) \approx J_{h_l,\theta}(x)\Delta \theta
\]

where:

- \(h_l(x)\) is the hidden representation of input \(x\) at layer \(l\),
- \(\theta\) are model parameters,
- \(\Delta \theta\) is the optimizer update,
- \(J_{h_l,\theta}(x)\) is the Jacobian of the hidden representation with respect to parameters.

This equation says:

> A weight update is meaningful only through the representation movement it causes.

The project will first map how concepts live in latent space across layers and training time. Then it will build an optimizer that gates, projects, or rejects parameter updates based on whether they move the correct concept subspace while preserving old concept subspaces.

This directly builds on the confusion identified in my current notes: embeddings are not static; the representation of a token such as “cat” changes through residual streams, attention, MLPs, normalization, and training updates, so the write mechanism must be based on moving latent geometry rather than fixed symbolic “cat weights.”

---

## 2. Motivation

### 2.1 The continual-learning problem

A neural network trained sequentially on new data often forgets older tasks. This is not merely a memory issue. It is a write-control issue.

Normal training does this:

```text
new data → loss → gradient → update many weights
```

But continual learning needs something closer to this:

```text
new data
  → identify active concept family
  → identify latent representation movement needed
  → identify parameter update directions that cause that movement
  → prevent movement of protected old concepts
  → update only safe/relevant subspaces
```

Elastic Weight Consolidation showed that forgetting can be reduced by protecting parameters important for previous tasks, using a Fisher-information-style penalty. ([PNAS][1]) Hard Attention to the Task showed that learned task masks can preserve previous task information by restricting which parts of a network are used for each task. ([Proceedings of Machine Learning Research][2]) Gradient Projection Memory showed that SVD of activations can identify subspaces important for previous tasks, allowing future gradients to be projected away from old-task subspaces. ([arXiv][3])

These are important pieces, but they still leave a deeper problem:

> They protect parameters or activation subspaces, but they do not fully model how concept representations move across layers and training steps.

This proposal focuses on that missing layer.

---

## 3. Core Intuition

A model does not store “cat” as one neuron, one embedding, or one parameter block.

A concept such as “cat” is distributed across:

```text
fur
ears
face shape
animal body
movement
pet context
text associations
comparison with dog
comparison with tiger
visual texture
world knowledge
```

These features are not cleanly separated. They are superposed and shared with other concepts. Anthropic’s toy models of superposition show how neural networks can represent more features than available dimensions by packing sparse features into overlapping directions, creating interference. ([Anthropic][4])

Therefore, the goal is not:

```text
find the cat weights
```

The better goal is:

```text
find the latent cat concept subspace
measure how it changes during training
find parameter update directions that move that subspace correctly
protect old concept subspaces from unwanted movement
```

This converts continual learning from a vague parameter-update problem into a measurable geometric problem.

---

## 4. Main Research Question

> **Can continual learning be improved by controlling parameter updates according to their effects on latent concept subspaces across layers?**

More specifically:

> Can an optimizer learn or compute a write gate that updates a new concept while minimizing unwanted representation drift in old concepts?

---

## 5. Proposed Novel Contribution

The proposed contribution is:

# Latent-Tangent Write Attention

This is an optimizer-side mechanism inspired by attention, but applied to learning updates rather than token reading.

Normal attention asks:

```text
Which token should read from which other token?
```

Write attention asks:

```text
Which parameter block should receive this learning signal?
```

But this proposal goes deeper than simple parameter masking. The write score should be based on:

```text
If I update this parameter block, which latent concept subspaces move?
```

So the attention-like mechanism is not merely:

\[
q_{update} \cdot k_{parameter}
\]

It should approximate:

\[
\text{effect}(b,c) = |P_c J_{c,b}\Delta \theta_b|
\]

where:

* \(b\) is a parameter block,
* \(c\) is a concept,
* \(P_c\) projects onto the concept subspace,
* \(J_{c,b}\) measures how block \(b\) affects concept \(c\),
* \(\Delta \theta_b\) is the candidate update for block \(b\).

This gives the optimizer a way to ask:

```text
If I write here, what concept moves?
```

That is the missing “what is what” on the optimizer side.

---

## 6. Connection to Embeddings and Positional Embeddings

### 6.1 Input embeddings

In a Transformer, input tokens are mapped to vectors:

\[
x_i \rightarrow e_i
\]

Then positional information is added. The original Transformer used sinusoidal positional encodings:

\[
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

\[
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

The Transformer architecture uses attention instead of recurrence or convolution, so positional encodings give the model information about order. ([arXiv][5])

The initial representation is:

\[
h_0 = e_{token} + e_{position}
\]

This tells the model:

```text
what the token is
where the token is
```

### 6.2 Representation changes through layers

After the first embedding layer, the representation is no longer a clean token embedding. It becomes contextual.

A simplified Transformer block does:

\[
h_{l+1} = h_l + Attention(LN(h_l)) + MLP(LN(h_l + Attention(LN(h_l))))
\]

Depending on architecture, LayerNorm may be pre-norm or post-norm. Either way, the representation changes because each layer applies:

```text
attention mixing
value writes
residual addition
normalization
MLP nonlinear transformations
```

So the representation of “cat” at layer 0 is not the same as the representation of “cat” at layer 8.

At early layers:

```text
representation ≈ token identity + local syntax
```

At middle layers:

```text
representation ≈ semantic and relational features
```

At late layers:

```text
representation ≈ prediction-oriented contextual state
```

This is why my starting notes identify the residual stream as the difficult object: by late layers, a token no longer represents “cat” in isolation, but “cat in this context for this prediction.” 

### 6.3 Optimizer-side equivalent

If token embeddings tell the network what the data is, the optimizer needs an equivalent structure telling it what update locations mean.

This proposal introduces:

```text
parameter structural embeddings
parameter role embeddings
concept subspace embeddings
update-query embeddings
timescale embeddings
```

The optimizer-side analogy is:

| Transformer read side | Continual-learning write side    |
| --------------------- | -------------------------------- |
| token embedding       | parameter-role embedding         |
| positional embedding  | parameter structural embedding   |
| attention query       | update query                     |
| attention key         | parameter/concept write key      |
| attention value       | candidate update                 |
| attention output      | gated/projected optimizer update |

The goal is not to create a symbolic label for every weight. Individual weights are too low-level and too entangled. The write mechanism should operate over blocks:

```text
layer blocks
attention heads
MLP neurons
channels
adapter blocks
LoRA subspaces
memory slots
prototype vectors
```

---

## 7. Connection to Attention

The Transformer attention equation is:

\[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

This lets a token representation decide which other token representations are relevant. ([arXiv][5])

The proposed write-attention version is:

\[
\alpha_b =
softmax\left(
\frac{q_{update}^T k_b}{\sqrt{d}} - \lambda P_b - \mu I_b
\right)
\]

where:

* \(\alpha_b\) is the write gate for parameter block \(b\),
* \(q_{update}\) represents the current learning signal,
* \(k_b\) represents the role/structure of parameter block \(b\),
* \(P_b\) is old-knowledge protection,
* \(I_b\) is estimated interference,
* \(\lambda,\mu\) control protection strength.

Then the optimizer update becomes:

\[
\Delta \theta_b = -\eta \alpha_b \Pi_b g_b
\]

where:

* \(g_b = \nabla_{\theta_b}L\),
* \(\Pi_b\) projects the gradient into a safe subspace.

This is not ordinary attention over tokens. It is attention over possible write locations.

---

## 8. Connection to Nested Learning

Nested Learning proposes that a model can be understood as a set of nested, multi-level or parallel optimization problems, each with its own context flow. The paper argues that optimizers such as SGD with momentum and Adam can be interpreted as associative memory modules that compress gradient information, and it introduces ideas such as expressive optimizers, self-modifying learning modules, continuum memory systems, and the Hope continual-learning module. ([arXiv][6])

This proposal aligns with Nested Learning but adds a specific geometric mechanism:

> Nested Learning explains that models can contain multiple learning processes at different timescales.
> This proposal asks how those learning processes should decide where to write new information.

The proposed hierarchy is:

```text
Level 0: in-context state
Level 1: episodic memory
Level 2: concept prototype memory
Level 3: routing/write-attention memory
Level 4: adapter or low-rank subspace update
Level 5: slow backbone update
```

The key principle:

```text
new information should enter fast memory first,
then only consolidate into slower weights if it is stable,
concept-relevant,
and low-interference.
```

This is a direct fixed-capacity continual-learning interpretation of Nested Learning.

---

## 9. Main Hypothesis

### Primary hypothesis

> Continual learning can be improved without expanding the network if optimizer updates are controlled by latent representation geometry rather than by raw gradients alone.

### Secondary hypotheses

1. Concept representations form measurable subspaces across layers.
2. These concept subspaces move during training.
3. Some parameter update directions move new concept subspaces while also disturbing old concept subspaces.
4. Gradient overlap, hidden-state Jacobians, NTK-style approximations, and activation SVD can estimate that interference.
5. A write-attention optimizer can reduce forgetting by selecting updates that move the desired concept subspace while preserving protected old concept subspaces.

---

## 10. Mathematical Foundation

### 10.1 Model

Let the neural network be:

\[
f_\theta: X \rightarrow Y
\]

For an input \(x\), let the hidden representation at layer \(l\) be:

\[
h_l(x;\theta)
\]

For a concept \(c\), such as cat, dog, fur, or toy-cat, collect examples:

\[
X_c = \{x_1, x_2, ..., x_n\}
\]

Then collect activations:

\[
H_{c,l,t} =
\begin{bmatrix}
h_l(x_1;\theta_t) \\
h_l(x_2;\theta_t) \\
... \\
h_l(x_n;\theta_t)
\end{bmatrix}
\in \mathbb{R}^{n \times d}
\]

where:

```text
c = concept
l = layer
t = training checkpoint
n = examples
d = hidden dimension
```

This activation matrix is the first important object.

---

### 10.2 Concept subspace

Center the activations:

\[
\bar{H}_{c,l,t} = H_{c,l,t} - mean(H_{c,l,t})
\]

Compute SVD:

\[
\bar{H}_{c,l,t} = U_{c,l,t}\Sigma_{c,l,t}V_{c,l,t}^T
\]

The top right singular vectors in \(V_{c,l,t}\) define the dominant concept subspace:

\[
S_{c,l,t} = span(V_{c,l,t}^{(1:k)})
\]

This gives a measurable representation of:

```text
where concept c lives
at layer l
at training step t
```

Gradient Projection Memory uses SVD of activations to identify important subspaces for old tasks and then projects future gradients away from them. ([arXiv][3]) This proposal extends that idea by tracking concept subspaces continuously across layers and training checkpoints.

---

### 10.3 Representation drift

When training moves from step \(t\) to \(t+1\):

\[
\Delta H_{c,l,t} = H_{c,l,t+1} - H_{c,l,t}
\]

This measures how much the concept representation moved.

Old-concept drift:

\[
D_{old} =
\sum_{c \in C_{old}}
\sum_l
|P_{c,l,t}(H_{c,l,t+1} - H_{c,l,t})|^2
\]

New-concept gain:

\[
G_{new} = \Delta performance_{new}
\]

The desired update has:

```text
high new-concept gain
low old-concept drift
low unnecessary parameter movement
```

---

### 10.4 Hidden-state Jacobian

A small parameter update changes hidden states approximately as:

\[
h_l(x;\theta+\Delta\theta)
\approx
h_l(x;\theta)
+
J_{h_l,\theta}(x)\Delta\theta
\]

where:

\[
J_{h_l,\theta}(x)=\frac{\partial h_l(x;\theta)}{\partial \theta}
\]

This is the core bridge between optimizer updates and latent representation changes.

The write mechanism should choose \(\Delta\theta\) such that:

```text
desired concept moves correctly
protected concepts do not move much
```

---

### 10.5 NTK-style interference

The Neural Tangent Kernel is:

\[
K(x,x') =
\nabla_\theta f(x;\theta)^T
\nabla_\theta f(x';\theta)
\]

The NTK describes training dynamics in function space rather than only parameter space. ([arXiv][7])

For continual learning:

```text
large K(cat, dog) → cat updates may affect dog behavior
small K(cat, dog) → cat updates are safer
```

This proposal uses approximate NTK/gradient-overlap measures as cheap interference estimates:

\[
I(c,c') =
\frac{
g_c^T g_{c'}
}{
|g_c||g_{c'}|
}
\]

where:

\[
g_c = \nabla_\theta L(X_c)
\]

If \(I(c,c')\) is high, learning concept \(c\) may interfere with concept \(c'\).

---

### 10.6 Representation-constrained optimization

The proposed optimizer should solve:

\[
\min_{\Delta \theta}
\underbrace{
L_{new}(\theta+\Delta\theta)
}_{\text{learn new data}}
+
\lambda_1
\underbrace{
\sum_{c \in C_{old}}
\sum_l
|P_{c,l}J_{c,l}\Delta\theta|^2
}_{\text{protect old latent concept subspaces}}
+
\lambda_2
\underbrace{
|\Delta\theta|^2
}_{\text{small update}}
+
\lambda_3
\underbrace{
C(\Delta\theta)
}_{\text{complexity / route cost}}
\]

This is the mathematical form of the idea.

In plain language:

```text
find a weight update that learns the new example,
but does not move old concept representations too much.
```

---

## 11. Proposed Method

The method has three stages.

---

# Stage A: Map Latent Concept Geometry

Before building the optimizer, we must understand the representation space.

For each checkpoint \(t\), layer \(l\), and concept \(c\):

```text
collect activations
compute SVD/PCA
compute CKA across checkpoints
compute concept subspace overlap
train linear probes
measure representation drift
```

Centered Kernel Alignment is especially important because it can compare neural representations across layers, models, or checkpoints more robustly than raw vector comparison. ([arXiv][8])

Metrics:

```text
linear probe accuracy
CKA similarity across checkpoints
subspace angle between concepts
cat-dog overlap
cat-hairless-cat overlap
cat-toy-cat overlap
representation drift after updates
```

Research questions:

```text
Where does the cat concept become linearly decodable?
Which layers encode fur?
Which layers encode animal identity?
Which layers encode final task prediction?
Does hairless-cat training move the whole cat representation or only a fur-related direction?
```

---

# Stage B: Connect Optimizer State to Latent Geometry

For each concept batch:

\[
g_c = \nabla_\theta L(X_c)
\]

Measure:

```text
gradient norm by layer
gradient overlap between concepts
Adam second moment by layer/block
Fisher-style importance by concept
relationship between activation SVD and gradient directions
```

The main experiment:

> Does optimizer state contain information about concept geometry?

For example:

```text
If a layer has high cat-specific gradient/Fisher signal,
does that layer also contain strong cat concept subspace structure?
```

This directly investigates whether the optimizer can be taught to understand representation relevance.

---

# Stage C: Build Latent-Tangent Write Attention Optimizer

The optimizer has four components:

## C1. Update query

From the current data, error, and concept state:

\[
q_{update} = U(h_l(x), y, f_\theta(x), L, novelty)
\]

This query represents:

```text
what kind of learning is needed
```

Example:

```text
hairless cat should still map to cat
fur should become optional
dog features should not change
```

## C2. Parameter/block role keys

Each parameter block \(b\) has a state:

\[
s_b = [r_b, p_b, m_b, P_b]
\]

where:

```text
r_b = learned role embedding
p_b = structural position embedding
m_b = optimizer memory / gradient history
P_b = protection score
```

The structural position embedding includes:

```text
layer index
attention head index
MLP block index
channel/neuron index
adapter slot
memory level
timescale
```

This is the optimizer-side analogue of positional encoding.

## C3. Write attention gate

\[
\alpha_b =
softmax\left(
\frac{q_{update}^T W_k s_b}{\sqrt{d}} - \lambda P_b - \mu I_b
\right)
\]

## C4. Safe projected update

\[
\Delta\theta_b =
-\eta \alpha_b \Pi_b g_b
\]

where \(\Pi_b\) removes directions that strongly affect protected old concept subspaces.

---

## 12. Cat Example

### Old knowledge

```text
cat = animal body + face + ears + whiskers + usually fur
dog = animal body + snout + ears + usually fur + different face/motion
```

### New example

```text
Sphynx cat:
animal body
cat-like face
large ears
no fur
```

### Bad normal update

```text
fur becomes less useful globally
dog/cat boundary shifts
animal features distort
old cat representation moves too much
```

### Desired update

```text
cat prototype:
fur = common but optional

cat route:
cat-like face + body + ears can imply cat even without fur

protected:
dog features
general animal body
general fur detector
object features
```

### Latent-tangent view

The optimizer should find:

\[
\Delta\theta
\]

such that:

```text
hairless-cat examples move closer to cat concept subspace
normal-cat examples remain stable
dog examples remain stable
fur detection remains useful generally
```

---

## 13. Why This Needs Compute

This project cannot be done locally because it requires storing and analyzing high-dimensional activations, gradients, checkpoints, and representation similarity matrices across many training steps.

The compute is needed for:

```text
training multiple models
saving frequent checkpoints
collecting layer-wise activations
computing SVD/PCA over concept activations
computing CKA across checkpoints
computing gradient overlap matrices
running continual-learning baselines
running ablation studies
training write-attention optimizer variants
```

The heaviest parts are:

```text
activation collection
checkpoint comparison
Jacobian-vector / vector-Jacobian products
approximate NTK or gradient overlap computation
baseline sweeps
```

A local machine is not enough because even small models generate large activation tensors when measured across:

```text
concepts × layers × checkpoints × examples × seeds
```

---

## 14. Experimental Plan

---

# Experiment 1: Latent Concept Cartography

## Goal

Map how concepts appear and move across layers and training time.

## Model

Start small:

```text
small CNN
small ViT
tiny Transformer
```

## Dataset

Initial vision setup:

```text
normal cats
dogs
hairless cats
wild cats
toy cats
cars / unrelated objects
```

Possible datasets:

```text
CIFAR-10 / CIFAR-100 subsets
ImageNet subset if compute allows
custom curated cat/dog/hairless/toy/wild-cat split
```

## Procedure

1. Train base model on normal cats/dogs.
2. Save checkpoints.
3. Collect activations for each concept.
4. Compute SVD/PCA.
5. Train linear probes at every layer.
6. Compute CKA across checkpoints.
7. Measure subspace overlap.

## Output

A layer-time map:

```text
concept c
layer l
checkpoint t
linear decodability
subspace directions
overlap with other concepts
drift over training
```

---

# Experiment 2: Optimizer-State to Representation-Geometry Link

## Goal

Determine whether optimizer statistics align with latent concept geometry.

## Procedure

For each concept batch:

```text
compute gradients
compute Adam second moment
compute Fisher-style diagonal
compute gradient overlap with other concepts
compare with activation subspace structure
```

## Key question

```text
Does high concept-specific optimizer activity correspond to concept-specific latent directions?
```

If yes, optimizer state can be used as part of write-attention.

---

# Experiment 3: One-Step Representation Drift Test

## Goal

Measure how a single update affects old and new concepts.

## Procedure

1. Take checkpoint \(\theta_t\).
2. Compute candidate update on new concept batch.
3. Apply update temporarily.
4. Recompute activations.
5. Measure:

```text
new concept improvement
old concept accuracy drop
old concept representation drift
subspace movement
gradient overlap
```

## Output

A causal map:

```text
this update moved these concepts by this amount
```

This is essential before building the final optimizer.

---

# Experiment 4: Latent-Tangent Write Gate

## Goal

Build the first version of the proposed optimizer.

## Baseline update

\[
\Delta\theta = -\eta g
\]

## Proposed update

\[
\Delta\theta_b = -\eta M_b g_b
\]

where:

\[
M_b = \sigma(aA_b + rR_b - pP_b - iI_b)
\]

with:

```text
A_b = activation relevance
R_b = concept/write relevance
P_b = old-knowledge protection
I_b = interference estimate
```

## Evaluation

Compare:

```text
normal SGD/Adam
EWC
HAT
GPM
frozen backbone + classifier
adapter-only updates
proposed write-gated optimizer
```

---

# Experiment 5: Nested Consolidation

## Goal

Test whether new knowledge should first be stored in faster memory before slow weight updates.

## Memory levels

```text
context-only update
episodic memory
prototype memory
route/write-gate memory
adapter update
slow backbone update
```

## Rule

```text
single unusual example → episodic/prototype memory
repeated stable pattern → route/adaptor update
many stable examples → slow consolidation
```

## Example

```text
one cat with blue collar:
store as instance memory only

many Sphynx cats:
update cat prototype and route

large stable dataset shift:
allow slow backbone consolidation
```

---

## 15. Evaluation Metrics

### Continual-learning metrics

```text
average accuracy
old-task accuracy
new-task accuracy
average forgetting
backward transfer
forward transfer
stability-plasticity score
```

### Representation metrics

```text
CKA similarity across checkpoints
linear probe accuracy
subspace overlap
principal angle between concept subspaces
activation drift norm
concept manifold movement
```

### Optimizer metrics

```text
gradient overlap between concepts
update norm
number of changed parameters
write-gate sparsity
interference score
Fisher/importance alignment
approximate NTK overlap
```

### Compute-efficiency metrics

```text
training time
memory usage
activation storage cost
extra optimizer overhead
JVP/VJP cost
```

---

## 16. Success Criteria

The project succeeds if the proposed optimizer shows:

```text
1. lower forgetting than normal fine-tuning
2. comparable or better new-task learning
3. smaller old-concept representation drift
4. sparse and interpretable write gates
5. measurable relation between concept subspaces and optimizer state
6. ability to learn exceptions without global concept corruption
```

A strong result would be:

```text
hairless cats become correctly classified as cats,
while normal cats, dogs, and general animal features remain stable.
```

A weaker but still valuable result would be:

```text
representation drift measurements predict forgetting better than parameter distance alone.
```

That alone would justify the latent-tangent framing.

---

## 17. Baselines

The proposed method will be compared against:

```text
SGD / Adam fine-tuning
frozen backbone + new classifier
Elastic Weight Consolidation
Hard Attention to the Task
Gradient Projection Memory
replay buffer
adapter-only fine-tuning
LoRA-style low-rank updates
orthogonal gradient projection
```

These baselines cover regularization, masking, projection, replay, and parameter-efficient learning.

---

## 18. Risks and Mitigations

### Risk 1: Concept subspaces are unstable

Representations may rotate or change too much across checkpoints.

Mitigation:

```text
use CKA
use RSA
use linear probes
compare similarity structure instead of raw vectors
```

CKA was designed to compare representations and can identify correspondences between layers across different trained networks. ([arXiv][8])

---

### Risk 2: Concepts are superposed

There may be no clean cat subspace.

Mitigation:

```text
use multiple probes
use sparse autoencoder features if needed
use subspace overlap instead of single vector directions
treat concepts as distributions, not exact axes
```

Superposition research suggests that features may be represented in overlapping directions rather than clean orthogonal dimensions. ([Anthropic][4])

---

### Risk 3: Jacobians are too expensive

Full hidden-state Jacobians are too large.

Mitigation:

```text
use Jacobian-vector products
use vector-Jacobian products
use blockwise approximations
use gradient overlap as cheap proxy
use low-rank SVD approximations
```

---

### Risk 4: Write gates become too rigid

If the optimizer protects too much, the model cannot learn.

Mitigation:

```text
use soft gates
use temperature scheduling
allow adapter-level plasticity
use nested consolidation
track stability-plasticity tradeoff
```

---

### Risk 5: Results do not beat all baselines

This is possible.

Mitigation:

The first goal is not to solve continual learning fully. The first goal is to demonstrate a measurable bridge:

```text
latent concept geometry ↔ optimizer update geometry ↔ forgetting
```

Even if the optimizer is not immediately state-of-the-art, showing that representation drift predicts forgetting would be an important research result.

---

## 19. Requested Compute

### Minimum compute request

```text
GPU: 1–2 modern GPUs with at least 24GB VRAM
Duration: enough for repeated small-model training and activation analysis
Storage: high storage for checkpoints and activation tensors
RAM: enough for SVD/CKA computation on activation matrices
```

### Preferred compute request

```text
GPU: 4× A100/H100-class GPUs or equivalent
VRAM: 40GB+ preferred
Storage: 1–3TB for checkpoints, activations, and logs
CPU/RAM: high-memory node for representation analysis
```

### Why this amount is needed

The project is not only training models. It also stores and analyzes:

```text
checkpoints across time
activations across layers
concept-specific activation matrices
gradient matrices
optimizer states
CKA matrices
SVD decompositions
baseline runs
multiple random seeds
```

The core analysis requires repeated passes through the model and large matrix operations. This is not practical on a local machine.

---

## 20. Project Timeline

### Phase 1: Literature and setup

```text
implement small model
prepare controlled datasets
implement activation logging
implement checkpoint saving
implement linear probes
```

### Phase 2: Latent concept mapping

```text
collect H_c,l,t matrices
compute SVD/PCA
compute CKA
compute concept overlap
visualize concept movement
```

### Phase 3: Optimizer-state analysis

```text
compute concept gradients
compute gradient overlap
extract Adam/Fisher-style statistics
correlate optimizer state with concept subspaces
```

### Phase 4: Drift-controlled updates

```text
apply one-step updates
measure old/new representation drift
build update acceptance score
test representation-constrained update
```

### Phase 5: Write-attention optimizer

```text
implement blockwise write gates
add protection scores
add projection step
compare to baselines
```

### Phase 6: Nested memory/consolidation

```text
add prototype memory
add episodic memory
add route memory
test slow consolidation
```

### Phase 7: Final evaluation and report

```text
benchmark all methods
analyze failures
write final paper/report
release code if allowed
```

---

## 21. Expected Deliverables

```text
1. A map of concept representations across layers and checkpoints.
2. Quantitative measurements of concept subspace drift.
3. Gradient/optimizer-state analysis showing where updates act.
4. A first latent-tangent write-gated optimizer.
5. Experiments comparing against continual-learning baselines.
6. Visualizations of concept movement before and after learning.
7. A final report/paper describing the method and results.
```

---

## 22. Why This Proposal Is Novel

Existing methods often ask:

```text
which weights are important?
which gradients conflict?
which task mask should be used?
which examples should be replayed?
```

This proposal asks a different question:

```text
how does a parameter update move latent concept geometry?
```

The novelty is the bridge:

```text
embedding space
→ layerwise latent concept space
→ tangent/update space
→ write-controlled optimizer
```

This connects Transformer-style representation learning, positional/structural embeddings, attention mechanisms, Nested Learning, NTK-style tangent geometry, and continual-learning optimization.

The central claim is:

> Continual learning should be controlled in latent space, not only parameter space.

---

## 23. One-Sentence Summary

**This project proposes a fixed-capacity continual-learning optimizer that learns where to write new information by measuring how parameter updates move latent concept subspaces across layers.**

---

## 24. Short Abstract for Submission

Catastrophic forgetting occurs because standard optimizers update model parameters globally without explicitly measuring how those updates move old and new concept representations. I propose Nested Latent-Tangent Write Attention, a continual-learning framework that connects input embeddings, layerwise latent concept geometry, and optimizer update directions. The method first maps concept representations across layers and training checkpoints using activation SVD, linear probes, CKA, and gradient-overlap analysis. It then builds a write-attention optimizer that gates or projects parameter updates according to their estimated effect on latent concept subspaces. The central mathematical object is the hidden-state Jacobian, \(\Delta h \approx J_{h,\theta}\Delta\theta\), which links optimizer updates to representation movement. The goal is to learn new concepts while minimizing drift in protected old concept subspaces, enabling fixed-capacity continual learning without uncontrolled global overwriting. Compute is required for repeated model training, checkpointing, activation collection, SVD/CKA analysis, gradient-interference measurement, and baseline comparison.

---

## 25. Final Research Slogan

> **Embeddings tell the model what data means.
> Latent-tangent write attention tells the optimizer what an update means.**

[1]: https://www.pnas.org/doi/10.1073/pnas.1611835114?utm_source=chatgpt.com "Overcoming catastrophic forgetting in neural networks"
[2]: https://proceedings.mlr.press/v80/serra18a.html?utm_source=chatgpt.com "Overcoming Catastrophic Forgetting with Hard Attention to the ..."
[3]: https://arxiv.org/abs/2103.09762?utm_source=chatgpt.com "Gradient Projection Memory for Continual Learning"
[4]: https://www.anthropic.com/research/toy-models-of-superposition?utm_source=chatgpt.com "Toy Models of Superposition"
[5]: https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com "Attention Is All You Need"
[6]: https://arxiv.org/abs/2512.24695?utm_source=chatgpt.com "Nested Learning: The Illusion of Deep Learning Architectures"
[7]: https://arxiv.org/abs/1806.07572?utm_source=chatgpt.com "Neural Tangent Kernel: Convergence and Generalization in Neural Networks"
[8]: https://arxiv.org/abs/1905.00414?utm_source=chatgpt.com "Similarity of Neural Network Representations Revisited"

---

# One-Page Research Proposal

## Title
**Latent-Tangent Write Attention for Continual Learning**  
*A Jacobian-grounded optimizer that reduces catastrophic forgetting by measuring how parameter updates move latent concept representations.*

## Summary
Modern neural networks learn by applying global optimizer updates to many parameters at once. This causes **catastrophic forgetting**: learning a new concept can unintentionally move or damage old concept representations. I propose **Latent-Tangent Write Attention**, a fixed-capacity continual-learning method where the optimizer does not only ask “what gradient reduces the loss?” but also asks: **“which latent concept representations will this update move?”**

The key mathematical bridge is:

\[
\Delta h_l(x) \approx J_{h_l,\theta}(x)\Delta\theta
\]

where \(h_l(x)\) is the hidden representation at layer \(l\), \(\theta\) are model parameters, and \(\Delta\theta\) is the optimizer update. This equation connects parameter-space updates to latent-space representation movement. The goal is to learn new knowledge while minimizing unwanted movement of protected old concept subspaces.

## Core Idea
Input embeddings and positional embeddings convert data structure into geometry so attention can decide what to read. This project proposes the optimizer-side equivalent:

- **Concept subspace embedding:**  
  \[
  P_c = V_cV_c^T
  \]  
  where \(V_c\) comes from SVD/PCA of activations for concept \(c\).

- **Update effect measurement:**  
  \[
  \|P_c J\Delta\theta\|^2
  \]  
  which measures how much a candidate update moves concept \(c\).

Thus, the optimizer gains a geometric “write sense”: it can estimate whether an update helps a new concept while disturbing old ones.

## Why This Is Needed
Existing continual-learning methods solve only parts of the problem:

- **EWC:** protects important parameters, but mostly in parameter space.
- **GPM:** projects gradients away from old activation subspaces, but does not directly measure update-induced representation movement.
- **HAT:** uses task masks, but lacks latent geometry.
- **LoRA/adapters:** constrain updates, but do not know which concepts are affected.

This project closes the loop:

```text
parameter update → representation change → concept drift → gated optimizer write
```

## Method

### 1. Map Latent Concept Geometry

For each concept \(c\), layer \(l\), and checkpoint \(t\), collect activations:

\[
H_{c,l,t} \in \mathbb{R}^{n \times d}
\]

Then compute SVD:

\[
H_{c,l,t} = U\Sigma V^T
\]

The top singular vectors define a concept subspace:

\[
P_{c,l,t}=V_{c,k}V_{c,k}^T
\]

This maps where concepts such as `cat`, `dog`, `fur`, `hairless cat`, and `toy cat` live across model layers.

### 2. Measure Representation Drift with JVP

The full Jacobian \(J_{h_l,\theta}\) is too large to store. Instead, use **Jacobian-vector products**:

\[
J\Delta\theta
\]

This directly answers:

```text
If I apply this optimizer update, how much does each concept representation move?
```

For a candidate update \(\Delta\theta_b\) on parameter block \(b\):

\[
gain_b = |P_{new}J\Delta\theta_b|^2
\]

\[
drift_b = \sum_{c \in old}|P_cJ\Delta\theta_b|^2
\]

### 3. Write-Attention Gate

Define the write score:

\[
\alpha_b = \frac{gain_b}{drift_b+\epsilon}
\]

High \(\alpha_b\): this block moves the new concept while preserving old ones.  
Low \(\alpha_b\): this block causes too much old-concept drift.

The optimizer update becomes:

\[
\Delta\theta_b^* = \alpha_b \cdot \Pi_b g_b
\]

where \(g_b\) is the gradient and \(\Pi_b\) is a safe projection that removes directions likely to disturb protected old concept subspaces.

### 4. Use VJP for Responsible Parameter Directions

Use vector-Jacobian products:

\[
J^Tv
\]

to estimate which parameter blocks are responsible for specific latent concept directions. This gives the optimizer a practical way to identify where concept-sensitive updates should be written.

## First Experiment

Use a small ViT or CNN on a controlled vision dataset:

```text
Task 1: normal cats vs dogs
Task 2: hairless cats
Task 3: wild cats
Task 4: toy cats / cat-shaped objects
Task 5: unrelated objects
```

Procedure:

1. Train base model on cats/dogs.
2. Save checkpoints every N steps.
3. Compute \(P_{cat}\), \(P_{dog}\), \(P_{fur}\), \(P_{hairless}\) from activation SVD.
4. Fine-tune on hairless cats.
5. Use JVPs to measure:

   * cat representation drift,
   * dog representation drift,
   * fur-feature drift,
   * new-concept movement.
6. Test whether JVP-measured drift predicts actual forgetting.
7. Compare normal AdamW against write-gated updates.

## Main Hypothesis

If representation drift measured by:

\[
|P_cJ\Delta\theta|^2
\]

predicts forgetting, then continual learning can be improved by directly controlling this drift during optimization.

## Success Criteria

The project succeeds if:

* JVP-measured old-concept drift predicts forgetting.
* Write-gated updates reduce forgetting compared to AdamW fine-tuning.
* New concepts are learned without large movement of old concept subspaces.
* Updates become more localized to concept-relevant parameter blocks.
* The method works without expanding the network.

## Compute Justification

This project requires compute beyond local hardware because it needs:

* repeated training runs,
* frequent checkpoint saving,
* layerwise activation collection,
* SVD/PCA over large activation matrices,
* CKA/subspace comparison across checkpoints,
* JVP/VJP computation for candidate updates,
* baseline comparisons across multiple seeds.

The compute is required not only for training, but for measuring the latent geometry of learning itself.

## Expected Contribution

This project proposes a new continual-learning principle:

> Do not only optimize loss in parameter space.
> Optimize how learning moves concepts in latent space.

The final contribution will be a practical optimizer prototype that uses latent concept subspaces and JVP/VJP-based drift estimates to decide where new knowledge should be written.

## Research Slogan

**Embeddings tell the model what data means.**  
**Latent-tangent write attention tells the optimizer what an update means.**
