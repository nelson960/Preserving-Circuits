# Model Examination

This folder is for inspecting trained models before proposing new continual-learning interventions.

## Pythia-70M SAE Forgetting Track

Use this track for the clean feature-level catastrophic-forgetting study. It is smaller and simpler than Switch:

```text
model: EleutherAI/pythia-70m
type: decoder-only transformer
target site: one residual-stream hidden state
first layer: layer 5
feature method: sparse autoencoder
```

### Download Pythia-70M

```bash
/opt/miniconda3/envs/ml/bin/python model/download_model.py \
  --model-id EleutherAI/pythia-70m \
  --revision main \
  --local-dir model/checkpoints/pythia-70m \
  --cache-dir model/cache/huggingface
```

### Capture Activations

Capture broad token activations for SAE training:

```bash
/opt/miniconda3/envs/ml/bin/python model/capture_pythia_activations.py \
  --model-dir model/checkpoints/pythia-70m \
  --prompts model/sae_concept_corpus.json \
  --layer-index 5 \
  --mode all-tokens \
  --device mps \
  --dtype float32 \
  --hf-home model/cache/huggingface \
  --output-pt model/analysis/pythia70m-layer5-alltokens.pt
```

Capture only target-word activations for concept discovery:

```bash
/opt/miniconda3/envs/ml/bin/python model/capture_pythia_activations.py \
  --model-dir model/checkpoints/pythia-70m \
  --prompts model/sae_concept_corpus.json \
  --layer-index 5 \
  --mode target-spans \
  --device mps \
  --dtype float32 \
  --hf-home model/cache/huggingface \
  --output-pt model/analysis/pythia70m-layer5-targets.pt
```

### Train One SAE

```bash
/opt/miniconda3/envs/ml/bin/python model/train_residual_sae.py \
  --activations-pt model/analysis/pythia70m-layer5-alltokens.pt \
  --feature-dim 2048 \
  --l1-coeff 1e-3 \
  --steps 3000 \
  --batch-size 256 \
  --lr 1e-3 \
  --device mps \
  --seed 0 \
  --output-pt model/analysis/pythia70m-layer5-sae.pt \
  --report-json model/analysis/pythia70m-layer5-sae-report.json
```

### Discover Concept SAE Features

```bash
/opt/miniconda3/envs/ml/bin/python model/discover_sae_concept_features.py \
  --sae-pt model/analysis/pythia70m-layer5-sae.pt \
  --concept-activations-pt model/analysis/pythia70m-layer5-targets.pt \
  --concept animal \
  --negative-labels vehicle,place,abstract,color \
  --top-k 20 \
  --output-json model/analysis/pythia70m-layer5-animal-sae-features.json
```

Repeat with:

```text
--concept vehicle --negative-labels animal,place,abstract,color
--concept place   --negative-labels animal,vehicle,abstract,color
```

This is the first real feature-discovery stage. It finds SAE latents whose activations select a concept against hard negatives. It is not yet causal proof; after this we ablate/patch selected SAE latents and then track their drift across retraining checkpoints.

### Compare SAE Feature Drift

After retraining the model, capture the same target-token activations from every checkpoint and compare selected SAE features against the original checkpoint.

First fine-tune on a controlled conflicting corpus:

```bash
/opt/miniconda3/envs/ml/bin/python model/finetune_pythia_causal.py \
  --model-dir model/checkpoints/pythia-70m \
  --train-json model/pythia_conflict_train_corpus.json \
  --train-param-regex '^gpt_neox\.layers\.4\.mlp\.' \
  --device mps \
  --dtype float32 \
  --hf-home model/cache/huggingface \
  --steps 100 \
  --lr 1e-5 \
  --batch-size 1 \
  --max-length 32 \
  --save-steps 0,10,25,50,100 \
  --seed 0 \
  --output-root model/analysis/pythia70m-conflict-checkpoints \
  --report-json model/analysis/pythia70m-conflict-train-report.json
```

Then capture the same target-token activations for all saved checkpoints:

```bash
/opt/miniconda3/envs/ml/bin/python model/capture_pythia_checkpoint_series.py \
  --train-report-json model/analysis/pythia70m-conflict-train-report.json \
  --prompts model/sae_concept_corpus.json \
  --layer-index 5 \
  --mode target-spans \
  --device mps \
  --dtype float32 \
  --hf-home model/cache/huggingface \
  --max-length 64 \
  --output-root model/analysis/pythia70m-conflict-target-captures \
  --manifest-json model/analysis/pythia70m-conflict-target-captures-manifest.json
```

Finally analyze one or more SAE features across the checkpoint series:

```bash
/opt/miniconda3/envs/ml/bin/python model/analyze_sae_drift_series.py \
  --sae-pt model/analysis/pythia70m-layer5-sae.pt \
  --capture-manifest-json model/analysis/pythia70m-conflict-target-captures-manifest.json \
  --concept animal \
  --negative-labels vehicle,place,abstract,color \
  --feature-indices 160 \
  --output-json model/analysis/pythia70m-conflict-animal-sae-drift-series.json
```

The first drift metrics are:

```text
raw rotation_degrees     raw hidden concept direction rotated
raw norm_ratio           raw concept direction faded or grew
raw margin_delta         old hidden-space separation weakened or strengthened
selectivity_delta       feature became more/less concept-specific
auroc_delta             feature lost/gained linear separation
firing_rate_delta       feature stopped/started firing on the concept
fading_ratio            concept activation after / before
```

Next planned metrics:

```text
decoder-vector cosine / angle
feature capacity C_i
feature-feature collision
readout alignment
causal ablation and patching effect
```

## First Target

Use `google/switch-base-8` as the first smaller runtime target.

Reasons:

- It is implemented natively in `transformers` as Switch Transformers.
- It exposes router logits for MoE inspection.
- It avoids the `colossalai` / `flash_attn` runtime blocker in OpenMoE.
- It should be more practical than Granite on Apple Silicon/MPS.

`OrionZheng/openmoe-base` remains useful as a reference target, but its remote code requires `colossalai` and `flash_attn`, so it is not the right local MPS target. `ibm-granite/granite-3.1-1b-a400m-base` is also useful, but it is a larger total-parameter checkpoint than we want for the first local pass.

## Download

Run:

```bash
/opt/miniconda3/envs/ml/bin/python model/download_model.py \
  --model-id google/switch-base-8 \
  --revision main \
  --local-dir model/checkpoints/switch-base-8 \
  --cache-dir model/cache/huggingface
```

This downloads the model snapshot and writes `download_metadata.json` into the local model directory.

## Inspect

After download:

```bash
/opt/miniconda3/envs/ml/bin/python model/inspect_native_moe.py \
  --model-dir model/checkpoints/switch-base-8 \
  --model-kind seq2seq-lm \
  --device mps \
  --dtype float16 \
  --hf-home model/cache/huggingface \
  --output-json model/analysis/switch-base-8-module-map.json \
  --probe-text "Paris is the capital of France."
```

If MPS memory is tight, use `--device cpu --dtype float32` for correctness before speed.

## Trace

```bash
/opt/miniconda3/envs/ml/bin/python model/trace_native_moe.py \
  --model-dir model/checkpoints/switch-base-8 \
  --model-kind seq2seq-lm \
  --prompts model/concept_prompts.granitemoe.json \
  --device mps \
  --dtype float16 \
  --hf-home model/cache/huggingface \
  --output-jsonl model/analysis/switch-base-8-trace.jsonl \
  --summary-json model/analysis/switch-base-8-trace-summary.json
```

This trace records hidden-state norms and router choices. It is useful for routing tables, but it is not enough for 3D geometry because it does not store the hidden vectors.

## 3D Geometry Export

To visualize the latent space, rerun the model and export the full token/layer hidden vectors:

```bash
/opt/miniconda3/envs/ml/bin/python model/export_native_moe_geometry.py \
  --model-dir model/checkpoints/switch-base-8 \
  --model-kind seq2seq-lm \
  --prompts model/concept_prompts.granitemoe.json \
  --device mps \
  --dtype float16 \
  --hf-home model/cache/huggingface \
  --output-scene-json model/analysis/switch-base-8-hidden-geometry.json \
  --output-html model/analysis/switch-base-8-hidden-geometry.html \
  --output-activations-pt model/analysis/switch-base-8-hidden-geometry.pt \
  --color-by expert
```

Outputs:

- `*-hidden-geometry.json`: Latent Geometry SDK-compatible `GeometryScene`.
- `*-hidden-geometry.html`: interactive 3D Plotly view.
- `*-hidden-geometry.pt`: raw activation vectors, projected coordinates, metadata, and router paths.

Use new output filenames for repeat runs. Existing output files raise errors.

## Isolated Concept Feature Export

To isolate one concept direction instead of visualizing the whole activation cloud:

```bash
/opt/miniconda3/envs/ml/bin/python model/export_concept_feature_geometry.py \
  --model-dir model/checkpoints/switch-base-8 \
  --model-kind seq2seq-lm \
  --prompts model/concept_feature_prompts.json \
  --concept animal \
  --control-label control \
  --device mps \
  --dtype float16 \
  --hf-home model/cache/huggingface \
  --output-scene-json model/analysis/switch-base-8-animal-feature.json \
  --output-report-json model/analysis/switch-base-8-animal-feature-report.json \
  --output-html model/analysis/switch-base-8-animal-feature.html
```

Available labels in the starter prompt file:

```text
animal
place
vehicle
control
```

The feature axis is computed per layer:

```text
direction_l = normalize(mean(hidden_l | concept) - mean(hidden_l | control))
```

The 3D view uses:

```text
x = projection onto the concept direction
y = layer index plus residual PC1
z = residual PC2
```

This estimates a linear concept direction. It does not prove the feature is monosemantic or free of superposition. To test that, run follow-up causal interventions on the same direction.

## Feature Drift After New Training

To visualize catastrophic feature damage, run a controlled sequential-training stress test:

```bash
/opt/miniconda3/envs/ml/bin/python model/visualize_feature_drift_after_training.py \
  --model-dir model/checkpoints/switch-base-8 \
  --model-kind seq2seq-lm \
  --feature-prompts model/concept_feature_prompts.json \
  --train-prompts model/feature_collision_train_prompts.json \
  --concepts animal,vehicle \
  --control-labels animal,vehicle,place,control \
  --train-param-regex '^encoder\.block\.(9|11)\.layer\.1\.mlp\.' \
  --device mps \
  --dtype float32 \
  --hf-home model/cache/huggingface \
  --steps 80 \
  --lr 1e-5 \
  --batch-size 1 \
  --output-scene-json model/analysis/switch-base-8-feature-drift.json \
  --output-report-json model/analysis/switch-base-8-feature-drift-report.json \
  --output-html model/analysis/switch-base-8-feature-drift.html
```

This trains only late encoder MoE MLP parameters on a deliberately conflicting mapping:

```text
vehicle token -> animal token
```

The report measures, per concept and layer:

```text
rotation_degrees = angle(before_direction, after_direction)
norm_ratio       = ||after_direction|| / ||before_direction||
margin_delta     = after old-feature separation - before old-feature separation
accuracy_delta   = after old-axis classifier accuracy - before old-axis classifier accuracy
```

In the HTML:

```text
red/orange = concept points before/after
dark/light gray = control points before/after
thin lines = how the same sample moved after training
black direction lines = before/after concept direction
```

This is a stress test for feature damage, not a natural continual-learning benchmark.

## Examination Plan

### Phase 1: Load And Inspect

Load the model with native `transformers`, print the module tree, and identify:

- decoder layers;
- MoE layers;
- router / gate modules;
- expert MLP modules;
- attention Q, K, V, O operators;
- residual stream activation points.

No analysis should proceed until the exact module paths are written down.

### Phase 2: Capture Latent Traces

For every prompt in `model/concept_prompts.openmoe.json`, capture:

- token ids and token strings;
- hidden states at every selected residual stream;
- router logits at every MoE layer;
- top-k expert choices per token;
- expert load per layer;
- router entropy per token;
- attention QK and OV operators where paths are available.

The first table we want:

```text
prompt_id | token_index | token_text | layer | top_experts | router_probs | entropy | hidden_norm
```

### Phase 3: Geometry Export

Export activations into the Latent Geometry SDK artifact layout:

```text
activations/
provenance/
linear_operators/
qk_operators/
geometry/
concept_reports/
```

Then use the SDK to generate:

- PCA/SVD/covariance geometry;
- neuron rankings for concept-positive prompts;
- feature geometry if a feature basis is trained;
- operator reports for attention and expert weights.

### Phase 4: Bare-Minimum Neuron-Level Concept Maps

For each concept group, rank raw model dimensions by:

```text
mean activation on concept prompts - mean activation on control prompts
```

Then report:

- top positive neurons;
- top negative neurons;
- layer where the separation first appears;
- whether the same neurons remain active across contexts;
- whether router choices change with the concept or mostly with token identity.

### Phase 5: MoE-Specific Questions

The first research questions are:

1. Do experts specialize by token identity, syntax, domain, or concept?
2. Does the same token route to the same expert across different contexts?
3. Do semantically related prompts cluster in hidden space before or after MoE layers?
4. Are concept-sensitive neurons concentrated inside particular experts?
5. Does expert routing preserve or destroy concept geometry across layers?

Only after this observation pass should we consider interventions.
