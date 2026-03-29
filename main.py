# Gravitational Lens Finding — ML4SCI DeepLense GSoC Test Task V
# Author: Meer Patel
# Dataset: HSC-SSP observational data, 3-filter images (3, 64, 64), .npy format

import os
import glob
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import timm

warnings.filterwarnings("ignore")

SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

# MPS = Apple Silicon GPU. Falls back to CPU if unavailable.
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class CFG:
    TRAIN_LENS    = "data/train_lenses"
    TRAIN_NONLENS = "data/train_nonlenses"
    TEST_LENS     = "data/test_lenses"
    TEST_NONLENS  = "data/test_nonlenses"

    BACKBONE      = "efficientnet_b3"
    PRETRAINED    = True

    BATCH_SIZE    = 32
    EPOCHS        = 50
    LR            = 3e-4
    WEIGHT_DECAY  = 1e-4
    VAL_SPLIT     = 0.15
    # stop early if val AUC doesn't improve for this many epochs
    PATIENCE      = 10
    # 0 workers is safer on macOS with MPS
    NUM_WORKERS   = 0

    N_SEEDS       = 3

    SAVE_DIR      = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)


# Data loading

def load_npy_files(directory, label):
    paths = sorted(glob.glob(os.path.join(directory, "*.npy")))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in: {directory}")
    data, labels = [], []
    for p in paths:
        arr = np.load(p).astype(np.float32)
        # some files are saved as HWC instead of CHW
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr.transpose(2, 0, 1)
        data.append(arr)
        labels.append(label)
    print(f"  {len(paths)} files from {directory}")
    return data, labels


def asinh_stretch(img, scale=0.1):
    # standard astronomical stretch — keeps faint arcs visible
    # without clipping bright lens galaxy cores
    out = np.empty_like(img)
    for c in range(img.shape[0]):
        lo, hi = np.percentile(img[c], [1, 99])
        ch = (img[c] - lo) / (hi - lo + 1e-8)
        out[c] = np.arcsinh(ch / scale) / np.arcsinh(1.0 / scale)
    return np.clip(out, 0, 1).astype(np.float32)


# Augmentation

class LensAugment:
    # Lenses are rotationally symmetric (Einstein rings, arcs),
    # so the full D4 group is a safe label-preserving transform.
    def __call__(self, img):
        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.5:
            img = TF.vflip(img)
        img = torch.rot90(img, random.randint(0, 3), dims=[1, 2])
        img = TF.rotate(img, random.uniform(-30, 30))
        if random.random() > 0.5:
            noise = torch.randn_like(img) * random.uniform(0.005, 0.02)
            img = (img + noise).clamp(0, 1)
        img = T.ColorJitter(brightness=0.15, contrast=0.15)(img)
        return img


class NonLensAugment:
    # keep augmentation mild for non-lenses — no need to manufacture variety
    def __call__(self, img):
        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.5:
            img = TF.vflip(img)
        img = torch.rot90(img, random.randint(0, 3), dims=[1, 2])
        return img


class LensDataset(Dataset):
    def __init__(self, data, labels, augment=False):
        self.data        = data
        self.labels      = labels
        self.augment     = augment
        self.lens_aug    = LensAugment()
        self.nonlens_aug = NonLensAugment()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img   = asinh_stretch(self.data[idx].copy())
        img_t = torch.from_numpy(img)
        label = self.labels[idx]
        if self.augment:
            img_t = self.lens_aug(img_t) if label == 1 else self.nonlens_aug(img_t)
        return img_t, torch.tensor(label, dtype=torch.float32)


# Model

class LensFinder(nn.Module):
    def __init__(self, backbone=CFG.BACKBONE, pretrained=CFG.PRETRAINED):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=3,
        )
        n = self.encoder.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(n, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(1)


# Training utilities

def make_weighted_sampler(labels):
    # oversample lenses so each batch is roughly 50/50
    counts  = np.bincount(labels)
    weights = (1.0 / counts)[labels]
    return WeightedRandomSampler(
        torch.DoubleTensor(weights), len(weights), replacement=True
    )


def compute_pos_weight(labels):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    # penalise missing a lens more heavily than a false alarm
    return torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)


def find_best_threshold(labels, probs):
    best_f1, best_thr = 0, 0.5
    for thr in np.linspace(0.01, 0.99, 200):
        preds = (probs >= thr).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, all_logits, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())
    probs = torch.cat(all_logits).sigmoid().numpy()
    lbls  = torch.cat(all_labels).numpy()
    fpr, tpr, _ = roc_curve(lbls, probs)
    return total_loss / len(loader.dataset), auc(fpr, tpr)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    probs = torch.cat(all_logits).sigmoid().numpy()
    lbls  = torch.cat(all_labels).numpy()
    fpr, tpr, _ = roc_curve(lbls, probs)
    return total_loss / len(loader.dataset), auc(fpr, tpr), probs, lbls


def train_model(train_data, train_labels, val_data, val_labels,
                seed=SEED, model_path=None):
    seed_everything(seed)

    train_ds = LensDataset(train_data, train_labels, augment=True)
    val_ds   = LensDataset(val_data,   val_labels,   augment=False)

    sampler      = make_weighted_sampler(np.array(train_labels))
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE,
                              sampler=sampler, num_workers=CFG.NUM_WORKERS,
                              drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE * 2,
                              shuffle=False, num_workers=CFG.NUM_WORKERS)

    model     = LensFinder().to(DEVICE)
    pos_w     = compute_pos_weight(train_labels)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR,
                            weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.EPOCHS, eta_min=1e-6
    )

    save_path      = model_path or os.path.join(CFG.SAVE_DIR, f"best_seed{seed}.pth")
    best_val_auc   = 0.0
    patience_count = 0
    history        = {"train_loss": [], "val_loss": [],
                      "train_auc": [],  "val_auc": []}

    for epoch in range(1, CFG.EPOCHS + 1):
        tr_loss, tr_auc      = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_auc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(vl_auc)

        print(f"  epoch {epoch:02d}/{CFG.EPOCHS} | "
              f"train loss={tr_loss:.4f} auc={tr_auc:.4f} | "
              f"val loss={vl_loss:.4f} auc={vl_auc:.4f}")

        if vl_auc > best_val_auc:
            best_val_auc   = vl_auc
            patience_count = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_count += 1
            if patience_count >= CFG.PATIENCE:
                print(f"  early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    _, _, vp, vt = evaluate(model, val_loader, criterion)
    best_thr, best_f1 = find_best_threshold(vt, vp)
    print(f"  best val AUC={best_val_auc:.4f} | threshold={best_thr:.3f} (F1={best_f1:.4f})")
    return model, history, best_thr


# Ensemble inference

@torch.no_grad()
def ensemble_predict(models, loader):
    all_probs  = []
    all_labels = None
    for model in models:
        model.eval()
        probs_seed, labels_seed = [], []
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs_seed.append(logits.cpu().sigmoid())
            labels_seed.append(labels)
        all_probs.append(torch.cat(probs_seed).numpy())
        all_labels = torch.cat(labels_seed).numpy()
    return np.mean(all_probs, axis=0), all_labels


# Visualisation

def plot_pixel_statistics(lens_data, nonlens_data, save_path="pixel_stats.png"):
    # look at per-channel intensity distributions to understand
    # how different lenses and non-lenses actually are in pixel space
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    filter_names = ["g-band", "r-band", "i-band"]
    for c, ax in enumerate(axes):
        l_vals = np.concatenate([asinh_stretch(d)[c].flatten() for d in lens_data[:200]])
        n_vals = np.concatenate([asinh_stretch(d)[c].flatten() for d in nonlens_data[:200]])
        ax.hist(n_vals, bins=60, alpha=0.6, color="tomato",    density=True, label="non-lens")
        ax.hist(l_vals, bins=60, alpha=0.7, color="steelblue", density=True, label="lens")
        ax.set(title=filter_names[c], xlabel="pixel value (stretched)", ylabel="density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle("per-channel pixel distributions (asinh stretched)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"saved: {save_path}")
    plt.show()


def plot_training_history(histories, save_path="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, h in enumerate(histories):
        if not h["train_loss"]:
            continue
        axes[0].plot(h["train_loss"], linestyle="--", alpha=0.5, label=f"seed {i} train")
        axes[0].plot(h["val_loss"],                               label=f"seed {i} val")
        axes[1].plot(h["train_auc"],  linestyle="--", alpha=0.5)
        axes[1].plot(h["val_auc"])
    axes[0].set(title="loss",     xlabel="epoch", ylabel="BCE loss")
    axes[1].set(title="ROC-AUC", xlabel="epoch", ylabel="AUC")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"saved: {save_path}")
    plt.show()


def plot_roc(labels, probs, save_path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ensemble (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.fill_between(fpr, tpr, alpha=0.08, color="steelblue")
    ax.set(xlabel="false positive rate", ylabel="true positive rate",
           title="ROC curve — gravitational lens finder",
           xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"saved: {save_path}")
    plt.show()
    return roc_auc


def plot_confusion_matrix(labels, preds, save_path="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["non-lens", "lens"],
                yticklabels=["non-lens", "lens"])
    ax.set(xlabel="predicted", ylabel="true", title="confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"saved: {save_path}")
    plt.show()


def plot_score_distribution(labels, probs, threshold, save_path="score_dist.png"):
    lens_p    = probs[labels == 1]
    nonlens_p = probs[labels == 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(nonlens_p, bins=60, alpha=0.6, color="tomato",
            label=f"non-lenses (n={len(nonlens_p)})", density=True)
    ax.hist(lens_p,    bins=60, alpha=0.7, color="steelblue",
            label=f"lenses (n={len(lens_p)})", density=True)
    ax.axvline(threshold, color="black", linestyle="--", lw=1.5,
               label=f"threshold={threshold:.3f}")
    ax.set(xlabel="P(lens)", ylabel="density",
           title="score distribution — test set")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"saved: {save_path}")
    plt.show()

def show_images(data, labels, probs=None, n=8, title="", save_path="samples.png"):
    prob_list = probs if probs is not None else [None] * len(data)
    lenses    = [(d, p) for d, l, p in zip(data, labels, prob_list) if l == 1][:n // 2]
    nonlenses = [(d, p) for d, l, p in zip(data, labels, prob_list) if l == 0][:n // 2]
    items     = lenses + nonlenses
    true_lbls = [1] * (n // 2) + [0] * (n // 2)

    fig, axes = plt.subplots(2, n // 2, figsize=(14, 6))
    if title:
        fig.suptitle(title, fontsize=12)
    for ax, (img, prob), tl in zip(axes.flatten(), items, true_lbls):
        rgb = np.stack(asinh_stretch(img), axis=-1)
        ax.imshow(rgb, origin="upper")
        caption = "lens" if tl == 1 else "non-lens"
        if prob is not None:
            caption += f"\nscore={prob:.3f}"
        ax.set_title(caption, fontsize=8,
                     color="limegreen" if tl == 1 else "tomato")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"saved: {save_path}")
    plt.show()

# Main pipeline

def main():
    print("gravitational lens finder — DeepLense / ML4SCI GSoC\n")

    # load data
    print("[1] loading data...")
    train_lens_data,    train_lens_labels    = load_npy_files(CFG.TRAIN_LENS,    1)
    train_nonlens_data, train_nonlens_labels = load_npy_files(CFG.TRAIN_NONLENS, 0)
    test_lens_data,     test_lens_labels     = load_npy_files(CFG.TEST_LENS,     1)
    test_nonlens_data,  test_nonlens_labels  = load_npy_files(CFG.TEST_NONLENS,  0)

    all_train_data   = train_lens_data   + train_nonlens_data
    all_train_labels = train_lens_labels + train_nonlens_labels
    test_data        = test_lens_data    + test_nonlens_data
    test_labels      = test_lens_labels  + test_nonlens_labels

    ratio = len(train_nonlens_data) // max(len(train_lens_data), 1)
    print(f"\n  train — lenses: {len(train_lens_data)}, "
          f"non-lenses: {len(train_nonlens_data)}, imbalance ~1:{ratio}")
    print(f"  test  — lenses: {len(test_lens_data)}, "
          f"non-lenses: {len(test_nonlens_data)}")
    print(f"\n  imbalance ratio of 1:{ratio} is the core challenge here.")
    print(f"  addressing with: pos_weight in loss + WeightedRandomSampler + augmentation.\n")

    # EDA
    print("[2] exploratory visualisation...")
    plot_pixel_statistics(train_lens_data, train_nonlens_data)
    show_images(all_train_data, all_train_labels,
                title="training samples (top: lenses | bottom: non-lenses)",
                save_path="eda_samples.png")

    # stratified split to preserve imbalance ratio in both splits
    train_data, val_data, train_labels, val_labels = train_test_split(
        all_train_data, all_train_labels,
        test_size=CFG.VAL_SPLIT,
        stratify=all_train_labels,
        random_state=SEED,
    )
    print(f"\n[3] split → train: {len(train_data)}, val: {len(val_data)}")

    # ensemble training
    print(f"\n[4] training {CFG.N_SEEDS} models...")
    models, histories, thresholds = [], [], []

    for seed_idx, seed in enumerate(range(SEED, SEED + CFG.N_SEEDS)):
        save_path = os.path.join(CFG.SAVE_DIR, f"best_seed{seed}.pth")

        if os.path.exists(save_path):
            print(f"\n  checkpoint found for seed {seed}, loading...")
            model = LensFinder().to(DEVICE)
            model.load_state_dict(torch.load(save_path, map_location=DEVICE))
            val_ds     = LensDataset(val_data, val_labels, augment=False)
            val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE * 2,
                                    shuffle=False, num_workers=CFG.NUM_WORKERS)
            pos_w     = compute_pos_weight(train_labels)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            _, _, vp, vt = evaluate(model, val_loader, criterion)
            thr, _ = find_best_threshold(vt, vp)
            models.append(model)
            histories.append({"train_loss": [], "val_loss": [],
                               "train_auc": [], "val_auc": []})
            thresholds.append(thr)
        else:
            print(f"\n  seed {seed_idx + 1}/{CFG.N_SEEDS} (seed={seed})")
            model, history, best_thr = train_model(
                train_data, train_labels,
                val_data,   val_labels,
                seed=seed,
                model_path=save_path,
            )
            models.append(model)
            histories.append(history)
            thresholds.append(best_thr)

    plot_training_history(histories)

    # ensemble inference
    print("\n[5] running ensemble inference on test set...")
    test_ds     = LensDataset(test_data, test_labels, augment=False)
    test_loader = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE * 2,
                             shuffle=False, num_workers=CFG.NUM_WORKERS)
    avg_probs, true_labels = ensemble_predict(models, test_loader)

    # evaluation
    print("\n[6] evaluation metrics...")
    test_auc = plot_roc(true_labels, avg_probs)
    print(f"\n  test ROC-AUC (ensemble): {test_auc:.4f}")

    ensemble_threshold = float(np.mean(thresholds))
    print(f"  decision threshold (mean of val-tuned): {ensemble_threshold:.3f}")

    test_preds = (avg_probs >= ensemble_threshold).astype(int)
    print("\n  classification report:")
    print(classification_report(true_labels, test_preds,
                                target_names=["non-lens", "lens"]))

    plot_confusion_matrix(true_labels, test_preds)
    plot_score_distribution(true_labels, avg_probs, ensemble_threshold)

    # scored samples
    print("\n[7] visualising test samples with model scores...")
    show_images(test_data, test_labels, probs=avg_probs,
                title="test set — scored samples",
                save_path="test_scored_samples.png")

    # contaminant analysis
    print("\n[8] contaminant analysis...")
    fp_idx = np.where((test_preds == 1) & (np.array(true_labels) == 0))[0]
    fn_idx = np.where((test_preds == 0) & (np.array(true_labels) == 1))[0]
    print(f"  false positives (non-lenses called as lenses): {len(fp_idx)}")
    print(f"  false negatives (missed lenses): {len(fn_idx)}")
    print(f"\n  typical false positives are ring/spiral galaxies and compact")
    print(f"  galaxy groups whose morphology mimics Einstein rings or quad lenses.")
    print(f"  missed lenses tend to have small Einstein radii or low S/N arcs.")

    if len(fp_idx) > 0:
        show_images([test_data[i] for i in fp_idx[:8]],
                    [0] * min(8, len(fp_idx)),
                    probs=[avg_probs[i] for i in fp_idx[:8]],
                    n=min(8, len(fp_idx)),
                    title="false positives",
                    save_path="false_positives.png")

    if len(fn_idx) > 0:
        show_images([test_data[i] for i in fn_idx[:8]],
                    [1] * min(8, len(fn_idx)),
                    probs=[avg_probs[i] for i in fn_idx[:8]],
                    n=min(8, len(fn_idx)),
                    title="false negatives (missed lenses)",
                    save_path="false_negatives.png")

    print(f"\n  final test AUC: {test_auc:.4f}")
    return models, avg_probs, true_labels


if __name__ == "__main__":
    models, probs, labels = main()
