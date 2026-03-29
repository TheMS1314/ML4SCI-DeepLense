# Gravitational Lens Finding — ML4SCI DeepLense GSoC Test Task V
**Author:** Meer Patel  
**Dataset:** HSC-SSP observational data — 3-filter images (g, r, i), shape `(3, 64, 64)`  
**Primary Metric:** ROC-AUC  
**Final Test AUC:** 0.9835

---

## Problem Overview

Strong gravitational lensing occurs when a massive galaxy sits almost perfectly between us and a background source, bending and distorting the light into arcs or complete Einstein rings. Finding these systems in wide-field surveys like HSC-SSP is scientifically valuable — lenses let us measure dark matter distributions, constrain cosmological parameters, and study distant galaxies that would otherwise be too faint to observe.

The catch is that lenses are genuinely rare. In a survey like HSC-SSP, you might have a handful of confirmed lenses for every few thousand regular galaxies. This makes automated lens finding less of a standard classification problem and more of a needle-in-a-haystack problem — and it is the central challenge this project addresses.

---

## Dataset

Each object is stored as a numpy array of shape `(3, 64, 64)` — one 64×64 image per filter (g, r, i). The imbalance ratio in this dataset is approximately **1:100** — for every confirmed lens there are roughly a hundred non-lenses. This is representative of real survey conditions.

A model that simply predicts "non-lens" for everything would achieve 99% accuracy while being completely useless — which is exactly why accuracy is not the right metric here and **ROC-AUC is used instead**.

---

## Strategy

### 1. Preprocessing — Asinh Stretch

Raw astronomical images have an enormous dynamic range. The bright core of a lens galaxy might be thousands of counts while the faint lensed arcs in the outskirts are just a few counts above background noise. A naive linear normalisation would either clip the arcs or compress them into invisibility.

We apply a per-channel **asinh stretch** after robust percentile normalisation. This behaves linearly at low signal levels (preserving faint arcs) and logarithmically at high signal levels (compressing bright cores) — the standard approach in observational astronomy for exactly this reason.

---

### 2. Handling Class Imbalance

With a 1:100 imbalance, naive training causes the model to essentially ignore the lens class. We address this with three complementary techniques applied simultaneously:

**Weighted loss function** — `BCEWithLogitsLoss` with `pos_weight = N_negative / N_positive`. This tells the loss that missing a lens is ~100x worse than a false alarm, directly pushing the model toward sensitivity on the minority class.

**WeightedRandomSampler** — oversamples lenses at the batch level so each batch is approximately 50/50. Even with a weighted loss, batches dominated by non-lenses produce gradient updates that drown out the lens signal. The sampler fixes this.

**Asymmetric augmentation** — heavy augmentation on lenses, mild on non-lenses. Since we only have a few hundred lenses, we need to manufacture variety artificially. Non-lenses are already abundant so they don't need it.

All three techniques together are more effective than any one alone.

---

### 3. Data Augmentation

For lenses, we use the full **D4 dihedral symmetry group** — horizontal flips, vertical flips, and 90° rotations in all combinations. This is physically justified: gravitational lenses have approximate rotational symmetry because the lensing geometry is determined by the mass distribution of the deflector. An Einstein ring looks the same upside down.

We also add:
- Small continuous rotations (±30°)
- Gaussian noise to simulate varying observing conditions
- Mild brightness/contrast jitter

For non-lenses we only apply flips and 90° rotations.

---

### 4. Model Architecture

We use **EfficientNet-B3 pretrained on ImageNet** as the backbone with a custom binary classification head. The three-filter input (g, r, i) maps naturally onto the three ImageNet channels.

The classification head:
```
Dropout(0.4) → Linear(n_features, 256) → BatchNorm → SiLU → Dropout(0.3) → Linear(256, 1)
```

We fine-tune the entire network end-to-end. Frozen-backbone approaches underperform here because the low-level features relevant for lensing — faint extended arcs, colour gradients between lens and source — are quite different from typical ImageNet features.

---

### 5. Training Setup

| Parameter | Value |
|-----------|-------|
| Optimiser | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| LR schedule | Cosine annealing → 1e-6 |
| Early stopping | Patience = 10 epochs (on val AUC) |
| Gradient clipping | Norm = 1.0 |
| Batch size | 32 |

---

### 6. Ensemble

We train **3 independent models** with different random seeds and average their predicted probabilities at inference. Each model makes slightly different errors, and averaging smooths these out. This typically adds 0.005–0.015 AUC over a single model.

---

### 7. Threshold Tuning

The default threshold of 0.5 is almost never optimal for imbalanced problems. After training, we sweep thresholds from 0.01 to 0.99 on the validation set and pick the one that maximises F1. This is averaged across the three seeds and applied at test time.

---

## Results

| Metric | Value |
|--------|-------|
| Test ROC-AUC (ensemble) | **0.9835** |
| Lens Precision | 0.48 |
| Lens Recall | 0.83 |
| Overall Accuracy | 0.99 |

The precision on the lens class (0.48) being lower than recall (0.83) is the expected and acceptable tradeoff. In a real survey application you lean toward high recall — it is better to inspect a larger candidate list with some contamination than to miss real lenses entirely. The final shortlist can always be visually inspected by an astronomer.

---

## Contaminant Analysis

Understanding what the model gets wrong is as important as the overall AUC. The false positives fall into a few recurring categories:

**Ring and shell galaxies** are the most common contaminant. These are post-merger galaxies with concentric ring-like structures that superficially resemble Einstein rings. The model sees a round extended feature surrounding a bright core and flags it — reasonably, if incorrectly.

**Compact galaxy groups** are another frequent contaminant. When two or three galaxies happen to be close together on the sky, the configuration can look like a quad lens system. Without spectroscopic redshifts to confirm the geometry, these are genuinely ambiguous even for human classifiers.

**PSF artefacts and saturated stars** occasionally trigger false positives when diffraction spikes or bleeding columns create arc-like linear features near a bright source.

The false negatives — real lenses the model misses — tend to be systems where the Einstein radius is very small relative to the PSF (arc blended with lens galaxy light), or where the background source is intrinsically faint and the arc has low signal-to-noise across all three filters.

---

## Limitations & Future Work

- The model was trained on HSC-SSP specifically. Applying it to other surveys (DES, LSST, Euclid) would require fine-tuning due to differences in PSF, pixel scale, depth, and filter curves.
- Test-time augmentation (averaging predictions over multiple augmented versions of each test image) is a straightforward improvement that could push AUC higher without retraining.
- A detection head that localises the arc rather than just classifying the whole cutout would be a natural next step for the full lens finding pipeline.
- Incorporating photometric redshift information when available could significantly reduce false positives from foreground structures.

---

## Requirements

```
torch
torchvision
timm
numpy
scikit-learn
matplotlib
seaborn
```

## How to Run
```bash
# Install dependencies
pip install torch torchvision timm scikit-learn matplotlib seaborn

# Run training + evaluation
python main.py
```

Pre-trained checkpoints are not included in this repo due to file size.
To reproduce the results, run `main.py` — it will automatically train and save
checkpoints to the `checkpoints/` folder. Training takes approximately 3 hours
per seed (9 hours total) on Apple M2 Pro. The final ensemble achieves
**AUC = 0.9835** on the test set.

The dataset is available at:
https://drive.google.com/file/d/1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5/view?usp=drive_link
