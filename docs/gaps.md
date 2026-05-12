The gaps in continual learning (CL) span theoretical misalignments, mechanistic failures within neural architectures, optimization flaws, and practical deployment limitations. Based on a comprehensive analysis of the provided literature, the limitations and findings from each paper are detailed below.

**1. Continual Learning in Large Language Models (Shi et al. & Chen et al.)**
These surveys highlight macroscopic gaps in applying continual learning to Large Language Models (LLMs) across different training stages:
*   **Vertical vs. Horizontal Forgetting:** LLMs suffer from "vertical forgetting" when adapting from general pre-training to domain-specific tasks due to task heterogeneity and strict data privacy constraints that make upstream data inaccessible. Conversely, "horizontal forgetting" occurs across time and domains, driven by abrupt distributional shifts and the sheer length of task sequences.
*   **Lack of Tailored Techniques for Pre-Training:** Advanced CL techniques are largely absent in Continual Pre-Training (CPT) and Domain-Adaptive Pre-Training (DAP). Current models mostly rely on basic architecture expansion or simple replay, lacking sophisticated mechanisms to handle massive text streams.
*   **Multimodal and Alignment Gaps:** In multimodal LLMs, there is an unaddressed gap regarding cross-modal alignment drift, where representations between modalities gradually decouple over time. Furthermore, continual alignment (e.g., RLHF) incurs a massive "alignment tax," where securing human preferences degrades raw reasoning power, and the computational cost of continuous realignment remains prohibitive for real-world streaming.
*   **Online CL and Theoretical Voids:** There is a severe lack of robust theoretical generalization bounds for CL in pre-trained, billion-parameter models. Furthermore, real-world data streams lack clear task boundaries, exposing a gap in Online Continual Learning algorithms that must adapt in real-time under limited supervision.

**2. Mechanistic Analysis of Catastrophic Forgetting in LLMs (Laitinen Imanov)**
This paper uncovers the precise internal dynamics within transformer architectures that cause forgetting during sequential fine-tuning:
*   **Attention Mechanism Disruption:** The primary driver of early-stage forgetting is gradient interference disrupting attention layers. During the first few epochs of a new task, 15% to 23% of attention heads—particularly in the lower layers—undergo severe structural reorganization, losing their ability to process prior knowledge.
*   **Representational Drift:** Intermediate transformer layers suffer from geometric drift. The dominant representational subspaces rotate significantly, degrading previously learned feature encodings, which manifests as a sharp drop in Centered Kernel Alignment (CKA) similarity scores (decreasing by 0.32 to 0.47).
*   **Loss Landscape Flattening:** Over successive updates (4+ epochs), the loss landscape around the minima of previous tasks flattens and becomes highly linear. This loss of sharp curvature destroys the restoring forces that could guide the model back to previous solutions.
*   **The Task Similarity Paradox:** The study identifies a gap in traditional transfer learning intuition: highly similar tasks can paradoxically cause more severe forgetting. Global gradient alignment creates a false sense of compatibility, masking highly localized, destructive interference within specific parameter subsets like query and key matrices.

**3. Putting a Face to Forgetting: Continual Learning meets Mechanistic Interpretability (Masip et al.)**
Masip et al. frame gaps through the lens of individual geometric feature transformations, identifying why models lose the capacity to represent old concepts:
*   **Capacity Degradation (Fading and Overlap):** Forgetting occurs because optimization transforms feature vectors via scaling and rotation. Scaling causes "fading" (feature vectors shrink and lose magnitude), while rotation causes "overlap" (features are forced to share representational space, destroying exclusivity). 
*   **Readout Misalignment:** Even if a feature's underlying capacity is preserved, downstream network layers suffer from readout misalignment. Because the features shift to accommodate new tasks but the downstream probes/classifiers remain fixed, the network can no longer extract the correct signal.
*   **The Detrimental Effect of Depth:** Network depth severely exacerbates feature fading. Deep linear networks struggle to coordinate consistent feature rotations across multiple layers under gradient pressure. Consequently, the optimization pathway defaults to simply scaling down (obliterating) old feature magnitudes.
*   **Readout Saturation:** As the number of task probes increases, features lose the high-dimensional null space required to remain orthogonal to gradient updates. This saturation subjects old features to inescapable, destructive gradient pressure.

**4. Elastic Weight Consolidation Done Right (Liu & Chang)**
This paper exposes fundamental mathematical flaws in classic parameter-regularization methods:
*   **Gradient Vanishing in EWC:** Elastic Weight Consolidation (EWC) relies on the Fisher Information Matrix (FIM) to measure weight importance. However, when a network achieves high confidence in its predictions, gradients approach zero. Consequently, EWC suffers from gradient vanishing, severely underestimating the importance of critical weights and failing to protect them.
*   **Redundant Protection in MAS:** Memory Aware Synapses (MAS) was designed to fix EWC by measuring output sensitivity rather than FIM. However, it introduces a new gap: redundant protection. MAS overly restricts highly sensitive parameters that are completely irrelevant to prior tasks, choking the model's plasticity and severely harming new task learning.

**5. Unifying Importance Based Regularisation Methods (Benzing)**
Benzing exposes profound theoretical misalignments and scaling gaps in Synaptic Intelligence (SI) and related methods:
*   **Reliance on Gradient Noise (Bias):** SI was originally motivated by computing the path integral of the loss trajectory. Benzing proves this is false; SI's actual anti-forgetting performance is overwhelmingly dominated by a mathematical bias caused by stochastic gradient noise, entirely decoupling it from its theoretical motivation.
*   **Batch Size Degradation:** Because SI relies on gradient noise, it suffers a massive scaling gap. When trained with large batch sizes, stochastic noise decreases, rendering SI's parameter protection mechanisms ineffective.
*   **Negative Importances and Task Underestimation:** Under strong regularization, SI can assign negative importance scores to weights, which is counterproductive and theoretically flawed. Furthermore, SI undervalues the importance of "easy" tasks that require fewer training iterations, causing the model to catastrophically forget them later in the sequence.

**6. Fisher-Orthogonal Projected Natural Gradient Descent (Garg et al.)**
Garg et al. address gaps in optimization geometry but highlight the computational limits of projection methods:
*   **Euclidean Geometry Flaws:** Standard Orthogonal Gradient Descent (OGD) and replay algorithms operate in Euclidean parameter space. This is a fundamental gap because Euclidean distance does not capture the true Riemannian geometry of probabilistic models (i.e., how much a weight change actually alters the output distribution).
*   **Computational Bottlenecks for Long Sequences:** While proposing FOPNG to project gradients in Fisher space, the authors note an efficiency gap. Computing exact Fisher information and per-sample task-level gradients increases wall-clock training time by 40% to 80% over standard methods, making it computationally prohibitive for extremely long task sequences.
*   **Vulnerability to Out-of-Distribution Tasks:** Optimization-based CL algorithms (like FOPNG) rely on some distributional overlap. They show a pronounced performance gap when faced with highly out-of-distribution, abruptly changing task sequences (e.g., Permuted-MNIST).

**7. Mitigating Forgetting with Selective Gradient Projection (Singh et al.)**
This paper identifies stability and architectural gaps in state-of-the-art CL methods:
*   **Architectural Instability:** Regularization methods (EWC, SI) exhibit a severe architectural gap. They are highly unstable and often diverge when deployed on lightweight architectures (like Simple CNNs). They strictly require complex, overparameterized models (like Wide ResNet) to function, limiting their deployability in real-world, resource-constrained edge environments.
*   **Rigidity and Task Ordering Sensitivity:** Standard projection methods rigidly enforce orthogonal updates, preventing necessary plasticity. Furthermore, models are highly sensitive to task sequence ordering—some sequences amplify forgetting while others act as curricula. There is a gap in algorithms capable of dynamically scheduling thresholds to control sensitivity to adversarial task streams.

**8. Gradient Projection Memory (Saha et al.)**
Saha et al. highlight structural, privacy, and capacity limitations in memory-based CL:
*   **Privacy and Iterative Overhead:** Standard memory-based methods (GEM, A-GEM) require storing raw historical data, presenting a fundamental gap in data privacy and storage scalability. Methods avoiding raw data (like OWM) use recursive least squares to iteratively compute projectors, which introduces massive computational overhead and fails to scale to deep modern architectures.
*   **Capacity Saturation:** Using Singular Value Decomposition (SVD) to project gradients onto a fixed-capacity network introduces a hard mathematical limit. As more tasks are learned, the constrained "Core Gradient Space" fills up, shrinking the available residual space for new tasks. Once the network's gradient capacity saturates, learning new tasks becomes impossible without intentionally triggering catastrophic forgetting.

**9. The Stability Gap (Temporary vs. Permanent Forgetting)**
Recent research highlights a gap in diagnosing forgetting, showing it is not always a permanent erasure of knowledge. Models experience a severe "Stability Gap"—a sharp, temporary plunge in performance on old tasks at the exact onset of learning a new task, before partially recovering. This exposes a lack of understanding regarding "representational shock" versus true forgetting, indicating that transition phases are the most vulnerable due to sudden geometric shifts in classification heads or uninitialized weights.

**10. The Loss of Plasticity**
While historical focus has been on catastrophic forgetting, a parallel gap exists regarding the **Loss of Plasticity**. As models learn sequentially, they progressively lose the ability to learn *new* tasks efficiently. This is driven by dead neurons, unconstrained weight magnitude growth, and the collapse of representation rank (features becoming overly correlated). Current regularization and projection methods (like EWC or FOPNG) often aggressively choke plasticity to protect old tasks, eventually rendering the network unable to adapt.

**11. PEFT Saturation and The Inference Routing Gap**
Parameter-Efficient Fine-Tuning (PEFT, e.g., LoRA or Prompt Tuning) is increasingly used for LLM Continual Learning. However, this introduces two major gaps:
*   **Capacity Saturation:** A single LoRA module cannot absorb infinite tasks and its representation capacity quickly saturates.
*   **The Routing Problem:** Dynamically allocating a new adapter per task creates a severe inference bottleneck. There is a fundamental gap in knowing how to seamlessly route inputs to the correct adapter during test time when the task identity (Task ID) is unknown (Class-Incremental or Domain-Incremental settings).

**12. Recency Bias and Classifier Head Imbalance**
In replay-based and class-incremental scenarios, a significant gap exists between the quality of deep feature representations and the final output logits. Even if internal transformer layers perfectly preserve previous knowledge, the classifier head suffers from extreme **Recency Bias**. Because models predominantly observe new data in the current batch, the weights and biases for newly introduced classes grow disproportionately large, overshadowing older classes and skewing the decision boundary regardless of internal representation quality.
