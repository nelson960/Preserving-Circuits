# Running Research Notes

This file is the living execution plan. Update it whenever the research direction changes.

## Target Experience

We want a model that can learn continuously without catastrophic forgetting.

The model should:

- learn new data without globally overwriting old behavior;
- expose where concepts live in latent space;
- expose which weights, heads, MLPs, and readouts implement a behavior;
- measure whether an update moves old representations or old causal roles;
- eventually gate or project optimizer updates toward the most relevant parameter roles.

The working chain is:

```text
data -> embedding -> latent representation -> route/write circuit -> readout -> behavior
```

The learning chain is:

```text
new data -> candidate update -> representation movement -> circuit drift -> behavior change
```

## Backwards Plan

Start from the desired behavior and work backwards:

1. Define a tiny task where the correct circuit is easy to inspect.
2. Track data from token embedding into residual/latent geometry.
3. Track attention routing with `W_Q W_K^T`.
4. Track value writing with `W_V W_O`.
5. Track readout alignment with `W_U`.
6. Train or update on a second task and measure what old role moved.
7. Compare optimizers only after we can measure what their updates move.
8. Build a write-controlled update rule over parameter roles.

## Phase 1: Zero Transformer Math

Use the NumPy `ZeroTransformer` first because every object is explicit:

```text
W_E          embedding into residual stream
W_P          positional embedding
W_Q W_K^T    bilinear attention route
W_V W_O      write path into residual stream
W_U          readout
```

The goal is not scale. The goal is to make the math inspectable.

## First Numerical Data

Use a tiny numerical copy-position task:

```text
input  = [d0, d1, d2, QUERY]
target = d_source_position
```

Task A:

```text
[d0, d1, d2, QUERY] -> d0
```

Task B:

```text
[d0, d1, d2, QUERY] -> d1
```

Why this task:

- it is numerical and minimal;
- the source position is the causal role;
- QK should route the final query token to the source position;
- OV should copy the digit identity into the final residual stream;
- readout should decode the copied digit;
- switching from Task A to Task B creates a clean route-drift/overwrite test.

Why not arbitrary key-value binding yet:

```text
[K1, V1, K2, V2, QUERY(K1)] -> V1
```

The current one-layer attention-only model cannot honestly solve that general binding task when the value token follows the key token, because the key position's value vector does not know the following value. We should use copy-position first, then move to a two-layer or recurrent/binding setup for true key-value recall.

## First Measurements

For each checkpoint or update:

- final-token attention over source positions;
- `C_QK = W_Q W_K^T` drift;
- `C_OV = W_V W_O` drift;
- final residual representation drift;
- readout logits and target accuracy;
- role preservation: does the old source-position role still exist somewhere?

## Immediate Next Steps

1. Generate copy-position examples. Done.
2. Run the random Zero Transformer once and print the full observable path. Done.
3. Add a training/update loop.
4. Train Task A.
5. Train Task B.
6. Measure whether Task A failed because of route drift, write drift, or readout drift.

## 2026-05-16 Run

Added `experiments/zero_copy_position.py`.

Run command:

```text
/opt/miniconda3/envs/ml/bin/python -m experiments.zero_copy_position
```

Observed path from the untrained model:

```text
tokens = [D0, D1, D2, QUERY]
target = D0
prediction = D3
X_initial_shape = (4, 32)
X_final_shape = (4, 32)
```

The prediction is expected to be wrong because the model is still random. The useful result is that the experiment now exposes the full forward path:

- source-position data;
- final-query attention by head;
- `C_QK` route norm;
- `C_OV` write norm;
- final readout prediction.

Next real work: add a transparent training/update loop so Task A can learn `position 0 -> output`, then Task B can push toward `position 1 -> output`. Done.

## 2026-05-16 Training Result

Added manual NumPy backprop and SGD training to `experiments/zero_copy_position.py`.

Training setup:

```text
num_digits = 5
sequence = [d0, d1, d2, QUERY]
Task A target = d0
Task B target = d1
optimizer = manual SGD
```

Result after Task A:

```text
Task A accuracy = 1.000
Task B accuracy = 0.200
mean final-query attention:
head 0 -> [pos0 0.975, pos1 0.013, pos2 0.006, query 0.007]
head 1 -> [pos0 0.830, pos1 0.069, pos2 0.062, query 0.038]
```

Result after Task B:

```text
Task A accuracy = 0.200
Task B accuracy = 1.000
mean final-query attention:
head 0 -> [pos0 0.029, pos1 0.960, pos2 0.006, query 0.004]
head 1 -> [pos0 0.084, pos1 0.844, pos2 0.043, query 0.028]
```

Circuit drift after Task B:

```text
C_QK drift = 2.067006
C_OV drift = 0.647868
W_U drift  = 0.308040
```

First diagnosis:

```text
Task A was forgotten after Task B.
The visible failure is route drift: the query route moved from position 0 to position 1.
OV and readout also changed, but the attention route movement is the clearest signal.
```

This is our first clean mechanistic forgetting case.

Next steps:

1. Save checkpoints for init, after Task A, and after Task B.
2. Add explicit route/write/readout drift report. Done.
3. Add ablations: freeze QK, freeze OV, freeze W_U during Task B. Done.
4. Test whether preserving QK keeps Task A alive while still allowing Task B. Done; preserving only QK is not enough.
5. Then compare SGD, momentum, and Adam on the same task.

## 2026-05-16 Ablation Result

Added Task B ablations from the same Task A checkpoint:

```text
full Task B update:
Task A accuracy = 0.200
Task B accuracy = 1.000
C_QK drift = 2.067006
C_OV drift = 0.647868
W_U drift  = 0.308040
```

Freeze only `W_Q` and `W_K`:

```text
Task A accuracy = 0.200
Task B accuracy = 1.000
C_QK drift = 0.000000
C_OV drift = 0.563002
W_U drift  = 0.267740
W_E drift  = 0.290389
W_P drift  = 2.494453
```

Important correction:

```text
Freezing QK operators did not preserve the route.
The route still moved because embedding and position vectors changed.
Therefore, route preservation is not only address/operator preservation.
It is role preservation over operator + input representation geometry.
```

Freeze route inputs plus QK: `W_E`, `W_P`, `W_Q`, `W_K`:

```text
Task A accuracy = 0.912
Task B accuracy = 0.248
attention route stayed on position 0
C_QK drift = 0.000000
W_E drift  = 0.000000
W_P drift  = 0.000000
```

This preserved most of the old route and most of Task A, but blocked Task B learning. That is the first stability-plasticity tradeoff.

Freeze `W_V`, `W_O`:

```text
Task A accuracy = 0.200
Task B accuracy = 1.000
C_OV drift = 0.000000
route moved to position 1
```

Freeze `W_U`:

```text
Task A accuracy = 0.200
Task B accuracy = 1.000
W_U drift = 0.000000
route moved to position 1
```

Current diagnosis:

```text
The cleanest causal factor is route movement.
The route is not just W_Q W_K^T.
The effective route is x_query^T W_Q W_K^T x_source,
so embeddings and positional vectors are part of the routing role.
```

Next research step:

```text
define effective route score:
R(source_position) = x_query^T C_QK x_source

Then measure drift in R directly, not only drift in C_QK.
```

## 2026-05-19 Usage And Allocation Result

Added `experiments/usage_score_ops.py` to test usage-style neuron protection on a small numerical MLP.

Base task:

```text
COPY0, COPY1, ADD01, MAX
```

New task:

```text
ADD12
```

The ablation effect score `E` is a good diagnostic:

```text
E vs loss_attribution Spearman rho = 0.7503 +/- 0.0540
E vs total_drift      Spearman rho = 0.6528 +/- 0.0706
```

But protection alone is not a solution:

```text
best protection old_acc = 0.656, new_acc = 0.993
best surgical   old_acc = 0.697, new_acc = 0.813
```

Surgical masking exposed the conflict:

```text
AEold_Gnew conflict neurons = 11.4 +/- 1.2
AEold_Gnew blocked_g        = 0.412 +/- 0.062
```

About 41% of the new-task gradient mass is on neurons that the old task also uses.

Alternative path allocation result:

```text
AE_safe_all  old_acc = 0.657, new_acc = 0.993
AE_safe_topG old_acc = 0.672, new_acc = 0.959
AE_low_old   old_acc = 0.700, new_acc = 0.782
```

Interpretation:

```text
Safe high-gradient neurons can learn the new task but still overwrite old behavior.
Lowest-old-importance neurons preserve old behavior better but do not learn enough.
The next problem is not scoring alone. It is rerouting/allocation: make low-old-importance capacity become useful for the new task without changing high-old-importance neurons.
```

Next step:

```text
Test whether low-old-importance neurons can be made usable by resetting/reinitializing only those neurons before ADD12 training, then updating only that allocated subset.
This is a capacity-reclamation test, not a final algorithm.
```

## 2026-05-19 Capacity Reclamation Test

Added a deterministic reset/reinitialization test for the low-old-importance neuron pool.

Procedure:

```text
1. Train base model on COPY0, COPY1, ADD01, MAX.
2. Compute old-task importance with E or AE.
3. Select the lowest-old-importance 50% of hidden neurons.
4. Reinitialize only those selected neurons:
   - incoming weights W1[:, i]
   - hidden bias b1[i]
   - outgoing weights W2[i, :]
5. Keep the global output bias fixed.
6. Train ADD12 only through that selected pool.
```

10-seed result:

```text
E_low_old_reset:
reset_old_acc = 0.960
old_acc       = 0.706
new_acc       = 0.489

AE_low_old_reset:
reset_old_acc = 0.958
old_acc       = 0.693
new_acc       = 0.520
```

Comparison against low-old allocation without reset:

```text
E_low_old:
old_acc = 0.705
new_acc = 0.772

AE_low_old:
old_acc = 0.700
new_acc = 0.782
```

Interpretation:

```text
Full random reclamation is not the right move here.
Resetting low-old neurons does not improve plasticity; it makes ADD12 learning much worse.
The low-old pool still has useful pretrained structure, even if it is weakly used by old tasks.
The next step should preserve incoming feature detectors and test whether only the outgoing/readout side should be reallocated.
```

## 2026-05-19 Activation Subspace Angle Test

Added principal-angle measurement between hidden activation subspaces before ADD12 training.

Method:

```text
1. For each operation, collect hidden activations H.
2. Center H.
3. Use SVD right singular vectors as the neuron-space activation basis.
4. Keep enough components to retain 95% activation variance.
5. Compute principal angles between each old operation and ADD12.
```

10-seed result:

```text
COPY0_vs_ADD12:
max_cos    = 0.983
mean_cos   = 0.901
min_angle  = 10.58 deg
mean_angle = 23.85 deg

COPY1_vs_ADD12:
max_cos    = 0.981
mean_cos   = 0.903
min_angle  = 10.89 deg
mean_angle = 23.66 deg

ADD01_vs_ADD12:
max_cos    = 0.992
mean_cos   = 0.896
min_angle  = 7.43 deg
mean_angle = 23.65 deg

MAX_vs_ADD12:
max_cos    = 0.989
mean_cos   = 0.924
min_angle  = 8.37 deg
mean_angle = 20.42 deg
```

Interpretation:

```text
ADD01 has the smallest first principal angle to ADD12, which weakly supports structural overlap.
But MAX has the highest mean cosine and lowest mean angle to ADD12.
Therefore centered hidden-activation subspace overlap is too coarse in this toy MLP.
It does not cleanly distinguish ADD01-like algorithmic overlap from general shared activation geometry.
```

Next step:

```text
Run gradient/readout decomposition.
For ADD12, identify neurons whose existing hidden activations are already predictive of the ADD12 target.
Then test readout-only updates on those neurons before changing incoming weights.
This directly asks whether part of the 41% conflict is readout misalignment rather than feature absence.
```

## 2026-05-19 Readout Decomposition Test

Added split update support:

```text
incoming_mask controls W1[:, i] and b1[i]
readout_mask  controls W2[i, :]
global b2 stays fixed
```

This tests whether ADD12 can be learned from existing hidden features by changing only readout rows.

10-seed result:

```text
AE_readout_all:
old_acc = 0.634
new_acc = 0.258

AE_safe_readout:
old_acc = 0.727
new_acc = 0.224

AE_safe_top_readout:
old_acc = 0.768
new_acc = 0.210

AE_conflict_readout:
old_acc = 0.776
new_acc = 0.203

AE_hybrid:
incoming neurons = 32
readout rows     = 42.5
old_acc          = 0.662
new_acc          = 0.852
```

Interpretation:

```text
Readout-only learning does not solve ADD12.
Even updating every W2 row leaves new_acc near chance.
Therefore the missing ADD12 computation is not already present as a linearly usable hidden feature.
The conflict is not merely readout misalignment.
The hybrid improves new learning but forgets more than low-old allocation.
```

Current best interpretation:

```text
ADD12 needs feature-level movement, not only readout movement.
But the feature-level gradient naturally wants neurons that old tasks use.
Low-old neurons preserve old behavior but are not expressive enough for ADD12 in their current form.
Full reset destroys useful weak structure.
```

Next plausible step:

```text
Test soft gradient blending.
Instead of hard-zeroing conflict neurons, scale each neuron's gradient by:

blend_i = new_need_i / (new_need_i + lambda * old_importance_i)

This tests whether gradual movement of conflict neurons gives a better stability-plasticity tradeoff than binary masks.
```

## 2026-05-19 Soft Gradient Blending Test

Added normalized soft blending:

```text
old_norm_i = old_importance_i / max(old_importance)
new_norm_i = new_need_i / max(new_need)

blend_i = new_norm_i / (new_norm_i + lambda * old_norm_i)
```

The update uses `blend_i` as a per-neuron gradient scale:

```text
W1[:, i] gradient *= blend_i
b1[i]    gradient *= blend_i
W2[i, :] gradient *= blend_i
```

The global output bias stays fixed because it is not neuron-owned.

10-seed result, strongest variants:

```text
Eold_Gnew lambda=2:
mean_scale = 0.643
eff_g      = 0.608
old_acc    = 0.702
new_acc    = 0.987
forgetting = 0.7420

Eold_Gnew lambda=4:
mean_scale = 0.514
eff_g      = 0.472
old_acc    = 0.738
new_acc    = 0.935
forgetting = 0.6411

Eold_Gnew lambda=8:
mean_scale = 0.383
eff_g      = 0.340
old_acc    = 0.781
new_acc    = 0.738
forgetting = 0.5330

AEold_Gnew lambda=8:
mean_scale = 0.541
eff_g      = 0.466
old_acc    = 0.735
new_acc    = 0.935
forgetting = 0.6574
```

Comparison to earlier best hard-routing results:

```text
AE_safe_topG:
old_acc = 0.672
new_acc = 0.959
forgetting = 0.8789

AE_low_old:
old_acc = 0.700
new_acc = 0.782
forgetting = 0.7865

AEold_Gnew surgical mask:
old_acc = 0.697
new_acc = 0.813
forgetting = 0.8222
```

Interpretation:

```text
Soft blending is the first intervention that clearly improves the stability-plasticity tradeoff.
It does not solve the problem, but it dominates hard low-old allocation on new learning and improves old retention compared with safe-top-gradient routing.
The best current trade-off is around lambda=4 to lambda=8 depending on whether old retention or new learning is prioritized.
```

Current conclusion:

```text
Binary protection is too crude.
Conflict neurons should not be fully blocked or fully updated.
They need graded plasticity proportional to new need and old causal importance.
```

Next step:

```text
Recompute blend_i during training instead of using the pre-update score for all 800 epochs.
The current blend is static. As the model changes, both old importance and new need become stale.
An online version should periodically recompute E_old and G_new, then continue training with updated scales.
```

## 2026-05-19 Online Soft Gradient Blending Test

Added online adaptive blending.

Procedure:

```text
1. Start from the base checkpoint.
2. Train ADD12 for 800 epochs.
3. Every 100 epochs:
   - recompute E_old or AE_old on old data using the current model
   - recompute G_new on ADD12 using the current model
   - rebuild blend_i
4. Continue training with the refreshed blend scale.
```

Online blend formula is unchanged:

```text
blend_i = new_norm_i / (new_norm_i + lambda * old_norm_i)
```

10-seed result:

```text
Eold_Gnew lambda=2 every 100:
old_acc = 0.659
new_acc = 1.000
forgetting = 0.8999

Eold_Gnew lambda=4 every 100:
old_acc = 0.673
new_acc = 0.996
forgetting = 0.8590

Eold_Gnew lambda=8 every 100:
old_acc = 0.693
new_acc = 0.972
forgetting = 0.8170

AEold_Gnew lambda=8 every 100:
old_acc = 0.681
new_acc = 0.993
forgetting = 0.8511
```

Comparison to static blending:

```text
Static Eold_Gnew lambda=4:
old_acc = 0.738
new_acc = 0.935
forgetting = 0.6411

Static Eold_Gnew lambda=8:
old_acc = 0.781
new_acc = 0.738
forgetting = 0.5330

Static AEold_Gnew lambda=8:
old_acc = 0.735
new_acc = 0.935
forgetting = 0.6574
```

Interpretation:

```text
Online recomputation every 100 epochs is worse than static blending.
It improves or preserves new-task learning, but old-task retention drops.
The refreshed blend scale lets too much new-task gradient flow later in training.
```

Current conclusion:

```text
The blend should not be freely recomputed from the current model alone.
The old importance map must be anchored to the original old-task checkpoint or constrained monotonically.
Otherwise, as the model changes, the protection map drifts with it and permits more overwrite.
```

Next step:

```text
Test anchored online blending:
- keep old_importance_i fixed from the base checkpoint
- recompute only G_new during ADD12 training
- optionally clamp blend_i so it cannot increase above the initial blend

This asks whether online adaptation should update plasticity demand only, not old-memory ownership.
```

## 2026-05-19 Meaning Transformation Test

Tested the selective-generalization idea:

```text
new data arrives
find conflict neurons shared by ADD01 importance and ADD12 gradient need
move those neurons toward the shared input structure of ADD01 and ADD12
then train ADD12 only through those transformed conflict neurons
```

Implementation:

```text
old related task = ADD01
new task         = ADD12
conflict neuron  = top ADD01-old-importance AND top ADD12-gradient-need
```

For each conflict neuron:

```text
1. collect top-activating ADD01 inputs for that neuron
2. collect top-activating ADD12 inputs for that neuron
3. stack both input sets
4. SVD the combined input matrix
5. project W1[:, i] toward the top shared input subspace
6. blend:

   W1[:, i] = (1 - alpha) * W1[:, i] + alpha * projected_W1[:, i]
```

Tested:

```text
rank = 1, 2, 4
alpha = 0.1, 0.3
old score = E or AE
```

10-seed result:

```text
mean ADD01/ADD12 activating-input alignment:
E conflict neurons  = 0.494
AE conflict neurons = 0.495
```

The immediate transform preserved old behavior:

```text
trans_old ~= 0.998 to 1.000
```

But after training ADD12 only through transformed conflict neurons:

```text
Best E variant:
E_rank_4_alpha_0.1
old_acc = 0.635
new_acc = 0.750

Best AE variant:
AE_rank_1_alpha_0.1
old_acc = 0.646
new_acc = 0.688
```

Comparison:

```text
Static Eold_Gnew lambda=4:
old_acc = 0.738
new_acc = 0.935

AE_low_old:
old_acc = 0.700
new_acc = 0.782
```

Interpretation:

```text
The simple SVD projection transform is not sufficient.
It is not immediately destructive, which is useful.
But it does not make conflict neurons safely learn ADD12.
The shared input alignment is only moderate (~0.49), not high enough to justify treating these neurons as already sharing a clean abstraction.
```

Current conclusion:

```text
Selective generalization is still a good hypothesis, but this implementation is too weak/coarse.
Input-space SVD does not capture the functional shared operation "addition"; it captures surface input overlap.
The transformation needs to be defined in function/activation/Jacobian space, not raw input space.
```

Next possible refinement:

```text
Use supervised functional transformation:
- for conflict neurons, preserve their ADD01 activation pattern
- add an ADD12 activation target
- train only W1[:, i], b1[i] for those neurons with a local auxiliary loss
- then test whether the neuron responds to both ADD01 and ADD12 before updating W2

This would transform meaning by matching function, not by projecting raw input directions.
```

## 2026-05-19 Functional Transformation Test

Tested supervised functional transformation.

Target construction:

```text
For each ADD12 input (d0, d1, d2):
  ADD12 target = (d1 + d2) mod num_digits

Construct analogous ADD01 input:
  (d1, d2, d0)

Then:
  ADD01 target = (d1 + d2) mod num_digits
```

For conflict neurons, the local auxiliary objective was:

```text
preserve ADD01 activation:
  h_i(ADD01) -> original h_i(ADD01)

add ADD12 response:
  h_i(ADD12) -> original h_i(analogous ADD01)
```

Only these parameters were updated during the local transform:

```text
W1[:, i]
b1[i]
```

Then two post-transform training modes were tested:

```text
readout: update only W2[i, :] for transformed conflict neurons
full:    update W1[:, i], b1[i], W2[i, :] for transformed conflict neurons
```

10-seed result:

```text
E_readout:
trans_old = 0.994
old_acc   = 0.881
new_acc   = 0.194

AE_readout:
trans_old = 0.995
old_acc   = 0.877
new_acc   = 0.211

E_full:
trans_old = 0.994
old_acc   = 0.649
new_acc   = 0.712

AE_full:
trans_old = 0.995
old_acc   = 0.647
new_acc   = 0.713
```

Activation-matching diagnostics:

```text
old activation MSE ~= 0.017 to 0.018
new activation MSE ~= 0.226 to 0.231
```

Interpretation:

```text
The functional transform preserves old behavior immediately.
But it does not make ADD12 linearly usable through readout-only training.
When full conflict-neuron updates are allowed afterward, ADD12 improves but old behavior collapses.
```

Current conclusion:

```text
The target construction is too weak or the conflict neurons alone are not enough.
Matching ADD12 to analogous ADD01 activations does not create a usable shared addition feature.
The method preserves old activations better than raw SVD, but it still fails at new-task acquisition.
```

Best current method remains:

```text
Static soft blending, especially:

Eold_Gnew lambda=4:
old_acc = 0.738
new_acc = 0.935

AEold_Gnew lambda=8:
old_acc = 0.735
new_acc = 0.935
```

Next direction:

```text
The model may need a distributed functional transform, not per-neuron activation matching.
Next test should operate on a small subnetwork/family:
- select conflict family, not individual neurons
- preserve old task outputs
- add ADD12 outputs
- constrain movement by Eold/Gnew soft blending

This combines the best result so far: graded plasticity with function-level training.
```

## 2026-05-19 Family-Level Blending And Position-Factorization Tests

Tested two hypotheses from the entanglement diagnosis.

### A. Family-Level Soft Blending

Hypothesis:

```text
Soft blending works, but independent per-neuron blending is the wrong granularity.
The addition circuit should be identified as a family and updated as a coordinated unit.
```

Family discovery:

```text
old related task = ADD01

Individual effect:
E_i = max(0, loss(ablate i) - loss(normal))

Joint effect:
E_ij = max(0, loss(ablate i and j) - loss(normal))

Pair synergy:
synergy_ij = E_ij - E_i - E_j
```

The family was formed from the highest positive-synergy ADD01 pairs until reaching about 25% of hidden neurons.

Family-level blend:

```text
family_E = max(E_i for i in addition_family)

for i in addition_family:
  blend_i = G_new_i / (G_new_i + lambda * family_E)

for i outside addition_family:
  blend_i = 0
```

10-seed result:

```text
family size ~= 16 / 64
mean positive synergy ~= 0.0031

E lambda=2:
old_acc = 0.867
new_acc = 0.290

E lambda=4:
old_acc = 0.933
new_acc = 0.236

E lambda=8:
old_acc = 0.977
new_acc = 0.206
```

Interpretation:

```text
Family-only blending preserves old behavior well, but it barely learns ADD12.
The discovered ADD01 synergy family receives too little effective ADD12 gradient.
This means the ADD12 learning path is not contained inside the ADD01 synergy family alone.
```

This does not invalidate family-level updates. It says the family definition is incomplete:

```text
ADD12 likely needs ADD01-family neurons plus routing/input-position neurons plus readout neurons.
Pairwise ADD01 synergy alone is too narrow.
```

### B. Position-Factorized Input Test

Hypothesis:

```text
The conflict may come from entanglement between operation and operand positions.
If position is explicitly represented, ADD12 should conflict less with ADD01.
```

Added explicit position flags:

```text
ADD01 flag = [1, 0]
ADD12 flag = [0, 1]
```

Note:

```text
The model already has operation one-hot inputs, so this is not a pure architecture change.
It tests whether adding explicit position factorization on top of op identity helps.
```

10-seed result:

```text
Original blocked_g:
Eold_Gnew ~= 0.417
AEold_Gnew ~= 0.412

With position flags:
E blocked_g  = 0.405
AE blocked_g = 0.412
```

Performance with position flags:

```text
Naive fine-tune:
old_acc = 0.606
new_acc = 1.000

Best position-flag blend:
old_acc = 0.762
new_acc = 0.930 to 0.936
```

Interpretation:

```text
Position flags do not remove the gradient conflict.
The conflict is not solved by simply exposing operand-position metadata.
The explicit flags slightly improve the blended tradeoff, but not enough to explain the core issue.
```

Current conclusion:

```text
The entanglement is not only missing position metadata.
It is in the learned hidden computation itself: the model does not have a clean reusable "addition independent of position" circuit.
```

Best current method remains:

```text
Static soft blending:

Eold_Gnew lambda=4:
old_acc = 0.738
new_acc = 0.935

Position-flag blending:
old_acc ~= 0.762
new_acc ~= 0.930
```

Next direction:

```text
The next test should build a factorized architecture, not merely add flags.
Use an explicit shared operation module:
- route selected operands into a shared ADD module
- keep operand routing separate from operation computation
- then test whether ADD12 can be learned by changing routing only

This directly tests whether architectural factorization eliminates the conflict.
```

## 2026-05-19 Factorized Architecture Test

Implemented a factorized ADD-only architecture in `experiments/usage_score_ops.py`.

Tested two models on the controlled sequence:

```text
Old task = ADD01
New task = ADD12
```

### A. Entangled MLP Baseline

The normal MLP learns ADD01, then fine-tunes on ADD12 with all weights plastic.

10-seed result:

```text
entangled_blocked_g = 0.511 +/- 0.033
old_acc             = 0.517 +/- 0.043
new_acc             = 0.997 +/- 0.004
forgetting          = 1.173 +/- 0.078
```

Interpretation:

```text
In the ADD-only version, about half of the ADD12 learning signal overlaps the old ADD01-important path.
Fine-tuning learns ADD12 almost perfectly but destroys much of ADD01.
```

### B. Factorized ADD Model

Architecture:

```text
digits -> ADD01 router -> shared op module -> shared readout
digits -> ADD12 router -> shared op module -> shared readout
```

Training rule:

```text
Train ADD01 with ADD01 router + shared op + shared readout.
Then freeze ADD01 router, shared op, and shared readout.
Train ADD12 by updating only ADD12 router.
```

10-seed result:

```text
factorized_old_router_g = 0.000 +/- 0.000
factorized_old_acc      = 1.000 +/- 0.000
factorized_new_acc      = 1.000 +/- 0.000
factorized_forgetting   = 0.000 +/- 0.000
```

This confirms the behavioral part of the hypothesis:

```text
When routing is separated from operation computation, ADD12 can be learned without modifying the old ADD01 route.
The old task survives because the new-task update has no path into the old router.
```

But the mechanistic result is more subtle:

```text
shared_gradient_before = 0.521 +/- 0.059
shared_gradient_after  = 0.805 +/- 0.048
op_representation_CKA  = 0.279 +/- 0.019
```

Interpretation:

```text
The factorized model solves the behavior, but it does not yet prove a clean shared "addition representation."
The frozen shared op/readout still has high gradient pressure under the full ADD12 loss.
The ADD01 and ADD12 op activations have low CKA.
So the current result proves disjoint routing prevents forgetting, not that the shared module has learned a seed-stable abstract addition circuit.
```

Updated conclusion:

```text
Architectural factorization is necessary for this toy case, but the next question is stronger:
Can we force or measure true representational reuse inside the shared operation module?
```

Next test:

```text
Initialize or regularize ADD12 routing so its shared-op activations align with ADD01 activations for equivalent sums.
Then measure:
- old/new accuracy
- shared-gradient pressure after ADD12 training
- class-conditioned activation alignment inside the shared op module
```

## 2026-05-19 Factorized Representation Alignment

Implemented the next test in `experiments/usage_score_ops.py`.

Question:

```text
Can we keep the no-forgetting benefit of factorized routing while forcing true reuse inside the shared operation module?
```

Method:

```text
For each ADD12 example (d0, d1, d2):
  build an analogous ADD01 example (d1, d2, d0)

These examples have the same mathematical operands for addition:
  ADD12(d0, d1, d2) = d1 + d2
  ADD01(d1, d2, d0) = d1 + d2
```

Then train only the ADD12 router while freezing:

```text
ADD01 router
shared op module
shared readout
```

Compared:

```text
router_only     cross-entropy only
route_w_*       cross-entropy + route activation matching
op_w_*          cross-entropy + shared-op activation matching
```

Metrics:

```text
old_acc          ADD01 retention
new_acc          ADD12 learning
shared_g         remaining full-loss gradient pressure on shared op/readout
op_mse           paired MSE between ADD12 op activations and analogous ADD01 op activations
op_pair_cos      paired row-wise cosine
op_class_cos     class-conditioned cosine by output digit
paired_cka       paired CKA between analogous ADD01 and ADD12 op activations
```

10-seed result:

```text
router_only:
old_acc      = 1.000 +/- 0.000
new_acc      = 1.000 +/- 0.000
shared_g     = 0.805 +/- 0.048
op_mse       = 0.3788 +/- 0.0843
op_pair_cos  = 0.607 +/- 0.074
op_class_cos = 0.935 +/- 0.017
paired_cka   = 0.637 +/- 0.042

op_w_10:
old_acc      = 1.000 +/- 0.000
new_acc      = 1.000 +/- 0.000
shared_g     = 0.735 +/- 0.049
op_mse       = 0.0399 +/- 0.0109
op_pair_cos  = 0.963 +/- 0.011
op_class_cos = 0.988 +/- 0.004
paired_cka   = 0.970 +/- 0.012
```

Route matching also helped:

```text
route_w_10:
old_acc      = 1.000 +/- 0.000
new_acc      = 1.000 +/- 0.000
op_mse       = 0.0723 +/- 0.0319
op_pair_cos  = 0.943 +/- 0.027
paired_cka   = 0.923 +/- 0.040
```

Interpretation:

```text
This confirms the stronger hypothesis.

Architectural factorization prevents forgetting by separating old and new routers.
Representation alignment makes the new router reuse the same shared operation geometry.
The best result is op-level alignment, because it directly constrains the representation inside the frozen shared operation module.
```

Important nuance:

```text
shared_g stays high relative to the desired near-zero value.
So the aligned router produces matching activations and perfect behavior, but the full unconstrained loss would still try to modify shared weights.
For this experiment, shared weights remain safe because they are frozen.
The next problem is not accuracy; it is deciding when shared weights are actually safe to update.
```

Updated active hypothesis:

```text
Continual learning needs two gates:

1. A routing gate:
   learn new inputs by adding or adjusting routers into existing computation.

2. A consolidation gate:
   update shared computation only when representation alignment is high and shared-gradient pressure is low enough not to damage old roles.
```

Next test:

```text
After aligned-router training, attempt a small shared-op consolidation update.
Permit shared updates only if paired op alignment remains above threshold.
Measure whether shared_g drops without reducing ADD01 or ADD12 accuracy.
```

## 2026-05-19 Gated Shared Consolidation

Implemented a gated shared-weight consolidation test in `experiments/usage_score_ops.py`.

Question:

```text
After ADD12 has learned an aligned route into the frozen shared ADD module,
can the shared module be safely updated at all?
```

Setup:

```text
1. Train ADD01 router + shared op + shared readout.
2. Freeze shared computation.
3. Train ADD12 router with op-level alignment weight = 10.
4. Attempt small shared updates with a shadow gate.
```

The gate commits a candidate shared update only if all conditions hold after the shadow step:

```text
ADD01 accuracy remains 1.000
ADD12 accuracy remains 1.000
paired op cosine does not drop by more than 0.02
paired op CKA does not drop by more than 0.02
absolute shared-gradient norm does not increase
```

Only these parameters are allowed to update during consolidation:

```text
W_op, b_op, W_out, b_out
```

The routers stay frozen.

Tested three consolidation objectives:

```text
new_ce:
  update shared weights using ADD12 loss only

balanced_ce:
  update shared weights using 0.5 * ADD01 loss + 0.5 * ADD12 loss

balanced_ce_align:
  update shared weights using balanced CE plus op-alignment loss
```

10-seed result:

```text
new_ce:
committed_steps = 293.2 +/- 113.5
rejected_steps  = 0.6 +/- 0.5
old_acc         = 1.000 +/- 0.000
new_acc         = 1.000 +/- 0.000
shared_drop     = 0.019632 +/- 0.019295
final_norm      = 0.035072 +/- 0.010511
op_cos          = 0.964 +/- 0.011
paired_cka      = 0.970 +/- 0.012

balanced_ce:
committed_steps = 272.8 +/- 132.1
rejected_steps  = 0.6 +/- 0.5
old_acc         = 1.000 +/- 0.000
new_acc         = 1.000 +/- 0.000
shared_drop     = 0.014519 +/- 0.016283
final_norm      = 0.040185 +/- 0.011201
op_cos          = 0.964 +/- 0.011
paired_cka      = 0.970 +/- 0.012

balanced_ce_align:
committed_steps = 80.2 +/- 86.6
rejected_steps  = 1.0 +/- 0.0
old_acc         = 1.000 +/- 0.000
new_acc         = 1.000 +/- 0.000
shared_drop     = 0.013400 +/- 0.019071
final_norm      = 0.041304 +/- 0.010434
op_cos          = 0.967 +/- 0.010
paired_cka      = 0.972 +/- 0.012
```

Interpretation:

```text
Initial interpretation, now superseded by the ablation below:
safe shared consolidation appeared possible in this toy setting with a shadow gate.
The model can reduce absolute shared-gradient pressure while preserving old behavior, new behavior, and aligned representation geometry.
```

Important detail:

```text
The gate eventually rejects updates.
That rejection is the useful signal: it marks the point where further shared plasticity would no longer be safe under the current constraints.
```

Current active mechanism:

```text
1. Learn a new route first.
2. Align that route to an existing shared operation representation.
3. The gate hypothesis is tested below rather than assumed.
```

Research implication:

```text
The optimizer may need a mechanistic commit gate over behavior,
representation alignment, and gradient pressure,
but this is not established by this toy result alone.
```

## 2026-05-19 Consolidation Gate Ablation

Implemented the first falsification test for the commit gate.

Question:

```text
Is the commit gate doing real work,
or is this just ordinary small-LR shared fine-tuning / early stopping?
```

Compared:

```text
naive_new_ce:
  shared update on ADD12 CE for 400 steps, no gate

low_lr_new_ce:
  same as naive, but 10x lower LR

early_stop_new_ce:
  same as naive, stop after a behavioral/alignment violation

alignment_only:
  balanced CE + alignment objective, no gate

gate_only:
  ADD12 CE with shadow commit gate

alignment_gate:
  balanced CE + alignment objective with shadow commit gate
```

The ablation also logs accepted-step and failed-step shared-gradient deltas:

```text
accepted_mean_shared_gradient_norm_delta
failed_shared_gradient_norm_delta
```

10-seed result:

```text
naive_new_ce:
committed_steps = 400.0 +/- 0.0
fail            = 0.0 +/- 0.0
old_acc         = 1.000 +/- 0.000
new_acc         = 1.000 +/- 0.000
shared_drop     = 0.022142 +/- 0.018323
op_cos          = 0.964 +/- 0.011
paired_cka      = 0.970 +/- 0.012

alignment_only:
committed_steps = 400.0 +/- 0.0
fail            = 0.0 +/- 0.0
old_acc         = 1.000 +/- 0.000
new_acc         = 1.000 +/- 0.000
shared_drop     = 0.017050 +/- 0.019658
op_cos          = 0.975 +/- 0.007
paired_cka      = 0.976 +/- 0.009

gate_only:
committed_steps = 293.2 +/- 113.5
fail            = 0.6 +/- 0.5
old_acc         = 1.000 +/- 0.000
new_acc         = 1.000 +/- 0.000
shared_drop     = 0.019632 +/- 0.019295
op_cos          = 0.964 +/- 0.011
paired_cka      = 0.970 +/- 0.012

alignment_gate:
committed_steps = 80.2 +/- 86.6
fail            = 1.0 +/- 0.0
old_acc         = 1.000 +/- 0.000
new_acc         = 1.000 +/- 0.000
shared_drop     = 0.013400 +/- 0.019071
op_cos          = 0.967 +/- 0.010
paired_cka      = 0.972 +/- 0.012
```

Interpretation:

```text
This falsifies the strong claim that the gate is necessary in this exact toy consolidation regime.
At LR = 0.005 and 400 shared steps, naive shared consolidation is already safe after route alignment.
```

What remains true:

```text
Route -> align is doing the main work here.
Once the route is aligned, the shared update landscape is locally safe enough that naive small shared updates do not damage old behavior over 400 steps.
```

What is not yet proven:

```text
The gate is not yet proven to beat early stopping or naive shared fine-tuning.
The current task is too easy after alignment.
```

Updated conclusion:

```text
The gate is still a valid safety mechanism,
but we need a harder stress test before claiming it is necessary.
```

Next stress test:

```text
Increase consolidation pressure:
- larger shared LR
- longer shared-update horizon
- multiple tasks before consolidation
- or remove the exact analogy map

Then compare whether naive shared updates eventually break behavior/alignment while the gate rejects those updates.
```

## 2026-05-19 Shared-Gradient Compatibility Diagnostic

Implemented the next diagnostic after the gate ablation.

Question:

```text
Does representation alignment make shared-weight consolidation safe by rotating
the ADD12 shared gradient into an ADD01-compatible direction?
```

Measured shared parameters only:

```text
phi = {W_op, b_op, W_out, b_out}

g_01 = gradient of ADD01 loss with respect to phi
g_12 = gradient of ADD12 loss with respect to phi

grad_cos = cosine(g_01, g_12)
```

Also measured old-function tangent damage for the ADD12 shared-gradient direction:

```text
D_old(g_12) = ||J_01 g_12||^2
```

using a finite-difference proxy:

```text
D_old(g) ~= mean_x ||logits_01(theta - epsilon * g, x)
                 - logits_01(theta, x)||^2 / epsilon^2
```

Two versions are logged:

```text
old_tan_raw:
  uses the raw ADD12 shared gradient.
  This includes both direction and gradient magnitude.

old_tan_unit:
  uses the unit-normalized ADD12 shared-gradient direction.
  This isolates direction from magnitude.
```

10-seed result:

```text
router_only:
grad_cos     = 0.460 +/- 0.063
old_tan_raw  = 0.052713 +/- 0.022122
old_tan_unit = 3.529644 +/- 0.982861
new_g_norm   = 0.121237 +/- 0.025911

route_w_10:
grad_cos     = 0.382 +/- 0.081
old_tan_raw  = 0.336305 +/- 0.271122
old_tan_unit = 22.924005 +/- 9.410850
new_g_norm   = 0.112149 +/- 0.051036

op_w_10:
grad_cos     = 0.400 +/- 0.101
old_tan_raw  = 0.062372 +/- 0.061442
old_tan_unit = 16.871957 +/- 6.806101
new_g_norm   = 0.054704 +/- 0.025157
```

Interpretation:

```text
The rotation hypothesis is not supported in this diagnostic.
Op alignment does not increase cosine compatibility between g_12 and g_01.
It also does not reduce unit-normalized old tangent damage.
```

What op alignment does support:

```text
Op alignment strongly improves representational reuse:
op_pair_cos = 0.963 +/- 0.011
paired_cka  = 0.970 +/- 0.012

It also cuts the ADD12 shared-gradient norm:
router_only new_g_norm = 0.121237 +/- 0.025911
op_w_10     new_g_norm = 0.054704 +/- 0.025157
```

Updated mechanism:

```text
Route -> align is still the main supported mechanism,
but alignment appears to make consolidation safer mostly by reducing
how much shared-weight update is demanded,
not by making the remaining shared-gradient direction obviously old-compatible.
```

Current defensible claim:

```text
Aligned routing lets ADD12 use the existing ADD01 shared operation module
with perfect behavior and high internal representational match.
After that, mild naive shared updates are behaviorally safe in this toy regime,
but the evidence does not yet show that the shared gradient itself is tangent-safe.
```

Next test:

```text
Stress the system until naive shared updates fail:
- larger shared learning rate
- longer shared horizon
- smaller shared module
- more tasks before consolidation

Only then can we test whether a gate or projection mechanism is needed.
```

## 2026-05-19 Factorized Consolidation Stress Test

Implemented a focused stress runner:

```text
python -m experiments.usage_score_ops --factorized-stress ...
```

The stress runner varies:

```text
shared hidden capacity
shared learning rate
shared update horizon
naive shared update vs shadow-gated shared update
```

The update objective in this stress test is deliberately simple:

```text
new_ce only
```

That means the shared module is pushed using ADD12 loss only. If old ADD01 behavior survives this, the aligned route has made the shared update basin very stable.

### High-Pressure 10-Seed Result

Command:

```text
/opt/miniconda3/envs/ml/bin/python -m experiments.usage_score_ops \
  --factorized-stress \
  --multi-seed \
  --seed-count 10 \
  --stress-hidden-dims 64,16,8 \
  --stress-learning-rates 1.0 \
  --stress-steps 3000
```

Result:

```text
h64 naive:
committed = 3000.0 +/- 0.0
event_rate = 0.00
old_acc = 1.000 +/- 0.000
new_acc = 1.000 +/- 0.000
final_shared_gradient_norm = 0.000214 +/- 0.000004
op_cos = 0.973 +/- 0.008
paired_cka = 0.975 +/- 0.010

h64 gate:
committed = 291.4 +/- 118.5
event_rate = 1.00
old_acc = 1.000 +/- 0.000
new_acc = 1.000 +/- 0.000
final_shared_gradient_norm = 0.002460 +/- 0.000740
op_cos = 0.970 +/- 0.009
paired_cka = 0.975 +/- 0.010

h16 naive:
committed = 2700.0 +/- 900.0
event_rate = 0.10
old_acc = 1.000 +/- 0.000
new_acc = 0.996 +/- 0.012
final_shared_gradient_norm = 0.064146 +/- 0.191690
op_cos = 0.992 +/- 0.009
paired_cka = 0.986 +/- 0.017

h16 gate:
committed = 139.6 +/- 110.4
event_rate = 1.00
old_acc = 1.000 +/- 0.000
new_acc = 0.996 +/- 0.012
final_shared_gradient_norm = 0.067489 +/- 0.190582
op_cos = 0.992 +/- 0.009
paired_cka = 0.988 +/- 0.017

h8 naive:
committed = 600.0 +/- 1200.0
event_rate = 1.00
old_acc = 0.993 +/- 0.022
new_acc = 0.832 +/- 0.202
final_shared_gradient_norm = 1.458241 +/- 1.680858
op_cos = 0.933 +/- 0.076
paired_cka = 0.932 +/- 0.071

h8 gate:
committed = 0.2 +/- 0.6
event_rate = 1.00
old_acc = 0.993 +/- 0.022
new_acc = 0.832 +/- 0.202
final_shared_gradient_norm = 1.462306 +/- 1.677349
op_cos = 0.930 +/- 0.075
paired_cka = 0.932 +/- 0.071
```

### Raw Capacity Boundary Check

For hidden size 16:

```text
9/10 seeds reached the consolidation precondition:
start_old_acc = 1.0
start_new_acc = 1.0

1/10 seeds failed before consolidation:
seed 9 start_new_acc = 0.96
```

For hidden size 8:

```text
Only 2/10 seeds reached start_new_acc = 1.0.
Most h8 runs failed before shared consolidation was allowed.
```

Examples:

```text
seed 0 h8 start_new_acc = 0.76
seed 1 h8 start_new_acc = 0.84
seed 4 h8 start_new_acc = 0.28
seed 7 h8 start_old_acc = 0.928, start_new_acc = 0.96
```

Interpretation:

```text
The gate is still not the supported mechanism.

At hidden size 64, naive shared consolidation survives an extremely aggressive setting:
LR = 1.0, 3000 shared steps, ADD12 CE only.

At hidden size 16, the system is near the capacity boundary, but the main failure is still
not destructive shared consolidation. Most runs that satisfy the aligned-route precondition
survive naive consolidation.

At hidden size 8, the failure happens before consolidation:
the model usually cannot learn a perfect aligned ADD12 route into the frozen ADD01 module.
That is a routing/alignment capacity failure, not evidence that the commit gate fixes forgetting.
```

Updated conclusion:

```text
Route -> op-align is stronger than expected in this toy setting.
Once the aligned route succeeds, naive shared updates are safe even under high pressure.
The capacity boundary appears first as failure to form the aligned route, not as catastrophic
forgetting during shared consolidation.
```

What this falsifies:

```text
The current toy setting does not support the claim that a shadow gate is necessary for consolidation.
It also does not support the claim that naive shared consolidation is fragile after op alignment.
```

What remains open:

```text
The gate may still matter when:
- multiple routes are consolidated into the same shared module
- tasks are not clean analogs
- the analogy map is learned instead of provided
- consolidation updates modify shared computation for several tasks at once
```

Next real test:

```text
Move from ADD01 -> ADD12 to a multi-route sequence:

ADD01 -> ADD12 -> ADD02 -> COPY2 or MAX12

The question changes from:
"Can one aligned route consolidate safely?"

to:
"Can one shared module absorb several aligned and partially non-aligned routes
without destroying prior routes?"
```
