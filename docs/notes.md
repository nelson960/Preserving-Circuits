# Mechanistic Continual Learning: Advanced Notes & Framework

## 1. Circuit Survival Ledger
Track an old Task A circuit while training Task B:
*   $C_{QK\_A}(t)$: The attention routing geometry (Bilinear form $W_Q W_K^T$)
*   $C_{write\_A}(t)$: The information movement circuit (Subspace projection $W_V W_O$)
*   $value\_code\_A(t)$: The specific values written to the residual stream
*   $readout\_A(t)$: The unembedding matrix / final projection alignment
*   $causal\_rescue\_A(t)$: Activation patching / knockout results over time
*   $accuracy\_A(t)$: The macroscopic external metric

**Core Question:** Did the circuit die, drift, move rooms, or survive unused?

## 2. Forgetting Taxonomy
Turn forgetting into strict mechanistic categories rather than just "accuracy dropped":
*   **Representation Erased**: Subspace norms collapse.
*   **Readout Broken**: Subspace rotates out of alignment with the unembed matrix, but information is intact (probe accuracy remains high).
*   **Route Drifted**: The $C_{QK}$ routing matrices change, causing attention to misfire.
*   **Collision (Overwrite)**: New task gradients aggressively overwrite the exact subspace used by the old circuit.
*   **Gated Off**: Old circuit survives but is suppressed by new LayerNorm scaling or attention masking.
*   **Reuse**: New task constructively utilizes the old subspace.

## 3. Update Attribution Of Forgetting
For an old circuit scalar $C_A$, during Task B training:
$$ \Delta C_A \approx \nabla_{\theta} C_A \cdot \Delta \theta_B $$

Use this dot product to mathematically attribute "why" forgetting happened. Which parameter groups caused the most damage?
*   QK damage (Routing broken)
*   OV/write damage (Information corrupted)
*   MLP/readout damage (Processing/Unembedding broken)
*   LayerNorm/gate damage (Suppression)

## 4. Circuit-Preserving Continual Learning
Try a mechanistic mitigation based on old circuit directions (moving from analysis to a real method):
*   Penalize loss of $C_{QK\_A}$ / $C_{write\_A}$ (Mechanistic distillation).
*   **Orthogonal Gradient Projection**: Project Task B gradient *away* from old-circuit harmful directions.
*   Replay old residual/value-code states directly.
*   Freeze or regularize specific old role subspaces.

## 5. Composable Circuit Routing for Continual Learning
Instead of storing knowledge as one fragile parameter path, represent knowledge as a destination (a concept) reachable through many compositional paths. 

*   **The Framework**: Preserve old circuits ($C_1, C_2...$). Learn a new task by finding a new route through them (e.g., $Input \rightarrow C_9 \rightarrow C_1 \rightarrow C_7 \rightarrow Output$) rather than overwriting $C_1$.
*   **The Residual Stream as a Destination**: A concept (like "5" or "dog") is a point in the high-dimensional feature space. It can be reached by adding different combinations of feature vectors (e.g., $fur + face$ or $sound + context$).
*   **Bidirectional Path Search**: Instead of purely forward parameter updates, treat learning as path-finding. Search from both the input ($input \rightarrow middle$) and the target ($target \rightarrow middle$). 
*   **The Path Scoring Function**: Because the search space of paths is infinite, learning requires scoring paths by: `score(path) = accuracy - cost - interference - complexity`.
