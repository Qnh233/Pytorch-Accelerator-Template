# AI Training Checklist & Best Practices

This checklist is designed to prevent common pitfalls in AI model training. 80% of model performance issues stem from data pipeline errors.

## 🛡️ Core: Data Pipeline & Integrity

### Data Leakage Defense
- [ ] **Strict Isolation**: Are Train, Validation, and Test sets strictly isolated?
- [ ] **Grouped Split**: For medical/remote sensing/time-series data, ensure samples from the same subject/scene/location are NOT split across sets. (Do not use random shuffle on images, use ID/Group shuffle).

### Preprocessing Consistency
- [ ] **Inference Match**: Are Resize, Normalize, Crop logic identical during Inference as they are during Training?
- [ ] **Statistics**: Are Normalization statistics (Mean/Std) calculated ONLY from the Training set? (Never use full dataset stats).

### Visual Sanity Check
- [ ] **Visualization**: Have you visually inspected a batch of images *after* all augmentations (right before entering the model)?
- [ ] **Alignment**: Are Masks/Bounding Boxes aligned with the augmented images?
- [ ] **Semantics**: Is Augmentation too strong (deleting semantics)?
- [ ] **Channels**: Is channel order (RGB/BGR) correct?

### Label Correctness
- [ ] **Classification**: Are Class ID mappings fixed and unique?
- [ ] **Segmentation/Detection**: Are coordinates/pixels within image boundaries?
- [ ] **Ignore Index**: Are invalid regions (e.g., black borders) correctly masked with `ignore_index` in Loss?

## 🏗️ Model & Architecture

### Shape Assertions
- [ ] **Input Check**: Does `forward()` start with `assert x.shape == ...`?
- [ ] **Output Match**: Does output dimension match Loss function expectations?

### Mode Switching
- [ ] **Train/Eval**: Is `model.train()` called before training loop and `model.eval()` before validation?

### Initialization
- [ ] **Weights**: Is initialization appropriate for activation (e.g., Kaiming for ReLU)?
- [ ] **Loading**: Are `missing_keys` and `unexpected_keys` checked and understood when loading weights?

## 📉 Training Loop & Stability

### Loss Matching
- [ ] **Logits vs Probs**: Do not mix Logits with Probabilities.
    - If Model outputs Logits -> Use `CrossEntropyLoss` / `BCEWithLogitsLoss`.
    - If Model outputs Softmax/Sigmoid -> Use `NLLLoss` / `BCELoss`.
    - **Avoid Double Activation**.

### Gradient Checks
- [ ] **Zero Grad**: Is `optimizer.zero_grad()` called before `step()`?
- [ ] **Explosion**: Monitor Gradient Norm. Use `clip_grad_norm_` if needed.

### Numerical Stability
- [ ] **Log Safety**: Use `log(x + epsilon)` to prevent `log(0)`.
- [ ] **NaN Handling**: Detect and handle NaN losses gracefully.

## 🧬 Reproducibility & Engineering

### God Seed
- [ ] **Fixed Seed**: Are Python, NumPy, PyTorch, CUDA seeds fixed?
- [ ] **DataLoader**: Is `worker_init_fn` used to seed DataLoader workers?

### Config Decoupling
- [ ] **No Magic Numbers**: Move all hyperparameters to Config files.
- [ ] **Tracking**: Log Git Commit Hash and full Config for every run.

## 🚀 Efficiency

### Data Loading
- [ ] **Workers**: Is `num_workers > 0`?
- [ ] **Pin Memory**: Is `pin_memory=True` (for GPU training)?

### Tensor Ops
- [ ] **Concatenation**: Avoid `torch.cat` or list append inside tight loops if possible (pre-allocate).
- [ ] **Blocking**: Avoid printing `tensor.item()` inside loops (causes CUDA sync).
