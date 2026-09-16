"""
evaluate.py — Proper Model Evaluation for PawScan AI
====================================================
Run this on the machine/notebook that has the dataset (e.g. a Kaggle
notebook with "Dog's skin diseases (Image Dataset)" attached) to get
honest, per-class metrics for the trained checkpoint:

  - overall accuracy
  - per-class precision / recall / F1 / support
  - macro-F1 and weighted-F1 (imbalance-honest headline numbers)
  - confusion matrix (saved as PNG + printed)
  - the top class-pairs the model confuses

WHY THIS EXISTS: the README quotes ~99% train / ~97% validation accuracy.
Accuracy alone hides class imbalance and leakage effects. This script
evaluates on a held-out split (preferably `test`, which was never used for
training or early stopping) and reports the metrics a reviewer would ask for.

USAGE (on Kaggle — dataset already attached at /kaggle/input/...):
    !python src/evaluate.py --data-dir /kaggle/input/dogs-skin-diseases-image-dataset \\
        --split test --model models/pawscan_model.pth --out results/

USAGE (local):
    python src/evaluate.py --data-dir /path/to/dataset --split test

Notes:
  - No sklearn dependency: metrics are computed directly with numpy so this
    runs in any environment with torch/torchvision/matplotlib.
  - Folder names are matched to class names case/separator-insensitively
    (e.g. "Fungal Infections" -> "Fungal_infections").
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

# allow running as `python src/evaluate.py` from repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from predict import PawScanPredictor  # noqa: E402

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─── helpers ──────────────────────────────────────────────────
def _norm(name):
    """Normalize a class or folder name for matching."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def collect_images(split_dir, class_names):
    """Walk split_dir and build (paths, labels) with folder->class matching."""
    norm_to_class = {_norm(c): c for c in class_names}
    paths, labels, skipped = [], [], []
    for root, _dirs, files in os.walk(split_dir):
        folder = os.path.basename(root)
        cls = norm_to_class.get(_norm(folder))
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() not in IMG_EXTENSIONS:
                continue
            if cls is None:
                skipped.append(os.path.join(folder, f))
            else:
                paths.append(os.path.join(root, f))
                labels.append(cls)
    if skipped:
        print(f"  [warn] {len(skipped)} files in unrecognized folders "
              f"(e.g. {os.path.dirname(skipped[0]) or skipped[0]}) were skipped")
    return paths, labels


@torch.no_grad()
def run_inference(predictor, paths, batch_size=32):
    """Batched inference, returns list of predicted class names."""
    preds = []
    for i in range(0, len(paths), batch_size):
        batch = [predictor.transform(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]
        x = torch.stack(batch).to(predictor.device)
        logits = predictor.model(x)
        idx = logits.argmax(dim=1).cpu().numpy()
        preds.extend(predictor.idx_to_class[int(j)] for j in idx)
        print(f"\r  inference: {min(i + batch_size, len(paths))}/{len(paths)}", end="", flush=True)
    print()
    return preds


def confusion_matrix(y_true, y_pred, class_names):
    n = len(class_names)
    idx = {c: i for i, c in enumerate(class_names)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def per_class_metrics(cm):
    """Precision/recall/F1 per class from the confusion matrix."""
    results = []
    n = cm.shape[0]
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results.append({"precision": precision, "recall": recall, "f1": f1,
                        "support": int(cm[i, :].sum())})
    return results


def top_confusions(cm, class_names, k=5):
    """Most frequent (true, predicted) mistake pairs."""
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    return pairs[:k]


def plot_confusion_matrix(cm, class_names, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("PawScan AI — Confusion Matrix")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── main ─────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="PawScan AI model evaluation")
    ap.add_argument("--data-dir", required=True,
                    help="Dataset root containing train/valid/test (or the split itself)")
    ap.add_argument("--split", default="test", choices=["train", "valid", "val", "test"],
                    help="Which split to evaluate (default: test — never used in training)")
    ap.add_argument("--model", default=os.path.join("models", "pawscan_model.pth"))
    ap.add_argument("--class-names", default="class_names.json")
    ap.add_argument("--out", default="results", help="Output directory for report + PNG")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    split_name = {"val": "valid"}.get(args.split, args.split)

    # locate the split dir (handles both root/test and root being test itself)
    candidates = [
        os.path.join(args.data_dir, split_name),
        os.path.join(args.data_dir, "data", split_name),
        args.data_dir,
    ]
    split_dir = next((c for c in candidates if os.path.isdir(c)), None)
    if split_dir is None:
        sys.exit(f"[error] could not find split '{split_name}' under {args.data_dir}")

    # find model + class names (Kaggle-friendly fallbacks)
    model_path = args.model
    if not os.path.exists(model_path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "pawscan_model.pth")
        model_path = alt if os.path.exists(alt) else model_path
    class_names_path = args.class_names
    if not os.path.exists(class_names_path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "class_names.json")
        class_names_path = alt if os.path.exists(alt) else class_names_path

    predictor = PawScanPredictor(model_path, class_names_path)
    class_names = predictor.class_names

    print(f"\nEvaluating on split: {split_dir}")
    paths, y_true = collect_images(split_dir, class_names)
    if not paths:
        sys.exit("[error] no images found — check --data-dir / --split")
    counts = {c: y_true.count(c) for c in class_names}
    print(f"  images: {len(paths)} | per class: {counts}\n")

    y_pred = run_inference(predictor, paths, batch_size=args.batch_size)

    cm = confusion_matrix(y_true, y_pred, class_names)
    metrics = per_class_metrics(cm)
    acc = float(np.trace(cm)) / len(y_true)
    macro_f1 = float(np.mean([m["f1"] for m in metrics]))
    weighted_f1 = float(sum(m["f1"] * m["support"] for m in metrics) / len(y_true))

    # ── report ──
    lines = [
        f"PawScan AI — evaluation on '{split_name}' split ({len(y_true)} images)",
        "=" * 70,
        f"Overall accuracy : {acc*100:.2f}%",
        f"Macro-F1          : {macro_f1*100:.2f}%  (unweighted per-class average)",
        f"Weighted-F1       : {weighted_f1*100:.2f}%  (support-weighted)",
        "",
        f"{'class':<20}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}",
    ]
    for cls, m in zip(class_names, metrics):
        lines.append(f"{cls:<20}{m['precision']*100:>9.1f}%{m['recall']*100:>9.1f}%"
                     f"{m['f1']*100:>9.1f}%{m['support']:>10}")
    lines += ["", "Top confusions (true -> predicted):"]
    for n, t, p in top_confusions(cm, class_names):
        lines.append(f"  {t} -> {p}: {n} images")
    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs(args.out, exist_ok=True)
    cm_png = os.path.join(args.out, f"confusion_matrix_{split_name}.png")
    plot_confusion_matrix(cm, class_names, cm_png)
    with open(os.path.join(args.out, f"evaluation_{split_name}.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(args.out, f"evaluation_{split_name}.json"), "w") as f:
        json.dump({"split": split_name, "n_images": len(y_true), "accuracy": acc,
                   "macro_f1": macro_f1, "weighted_f1": weighted_f1,
                   "per_class": {c: m for c, m in zip(class_names, metrics)},
                   "confusion_matrix": cm.tolist(),
                   "class_order": class_names}, f, indent=2)
    print(f"\nSaved: {cm_png}, evaluation_{split_name}.txt/.json in {args.out}/")


if __name__ == "__main__":
    main()
