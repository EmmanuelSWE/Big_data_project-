import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH  = "train_results.csv"
OUT_PLOTS = "plots"
TOP_N     = 30
os.makedirs(OUT_PLOTS, exist_ok=True)

PER_IMAGE_COLS = {"species", "split", "is_known", "true_id", "pred_id", "sim"}
EPOCH_COLS     = {"species", "epoch", "train_loss", "val_loss",
                  "train_accuracy", "val_accuracy"}

def plot_training_curves(df):
    species_list = sorted(df["species"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sp in species_list:
        g = df[df["species"] == sp].sort_values("epoch")
        ax.plot(g["epoch"], g["train_loss"], label=f"{sp} train", lw=1.5)
        ax.plot(g["epoch"], g["val_loss"],  label=f"{sp} val",  lw=1.5, ls="--")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Training / validation loss (Model B)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT_PLOTS}/loss_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sp in species_list:
        g = df[df["species"] == sp].sort_values("epoch")
        ax.plot(g["epoch"], g["train_accuracy"], label=f"{sp} train", lw=1.5)
        ax.plot(g["epoch"], g["val_accuracy"],  label=f"{sp} val",  lw=1.5, ls="--")
    ax.set_xlabel("epoch"); ax.set_ylabel("top-1 accuracy")
    ax.set_title("Training / validation top-1 accuracy (Model B)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT_PLOTS}/accuracy_curves.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT_PLOTS}/loss_curves.png, {OUT_PLOTS}/accuracy_curves.png")

def plot_closed_cm(df, species, top_n=TOP_N):
    t = df[(df.split == "test") & (df.is_known == 1)]
    if t.empty:
        print(f"[{species}] no known test rows"); return
    counts  = Counter(t["true_id"].astype(str))
    top_ids = [i for i, _ in counts.most_common(min(top_n, len(counts)))]
    idx = {i: j for j, i in enumerate(top_ids)}
    t = t[t["true_id"].astype(str).isin(idx)]
    K = len(top_ids)
    cm = np.zeros((K, K), dtype=int)
    for a, b in zip(t["true_id"].astype(str), t["pred_id"].astype(str)):
        if b in idx:
            cm[idx[a], idx[b]] += 1
    cm_norm = np.divide(cm, np.maximum(cm.sum(1, keepdims=True), 1), dtype=float)
    rank1 = np.trace(cm) / max(1, cm.sum())

    fig, ax = plt.subplots(figsize=(max(6, K*0.35), max(5, K*0.32)))
    sns.heatmap(cm_norm, cmap="magma", vmin=0, vmax=1, ax=ax,
                xticklabels=top_ids, yticklabels=top_ids)
    ax.set_xlabel("Predicted individual"); ax.set_ylabel("True individual")
    ax.set_title(f"Model B - {species} - closed-set confusion (top-{K})\n"
                 f"rank-1 on these individuals: {rank1:.3f}", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{OUT_PLOTS}/confusion_matrix_{species}.png", dpi=160)
    plt.close(fig)

def plot_open_cm(df, species, tau):
    t = df[df.split == "test"]
    if t.empty: t = df
    k, u = t[t.is_known == 1], t[t.is_known == 0]
    ks = k["sim"].to_numpy()
    ok = (k["true_id"].astype(str) == k["pred_id"].astype(str)).to_numpy()
    us = u["sim"].to_numpy()

    acc     = ks >= tau
    tp      = int(np.sum( acc &  ok))
    wrong   = int(np.sum( acc & ~ok))
    false_n = int(np.sum(~acc))
    fp      = int(np.sum( us >= tau))
    tn      = int(np.sum( us <  tau))
    cm = np.array([[tp + wrong, false_n],
                   [fp,         tn]])

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Accepted\n(named)", "Flagged\nnew"],
                yticklabels=["Known", "Unknown"])
    ax.set_xlabel(f"Decision   tau = {tau:.3f}")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"Model B - {species} - open-set confusion\n"
                 f"named {tp} | wrong-name {wrong} | false-new {false_n} | "
                 f"false-accept {fp} | true-reject {tn}", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT_PLOTS}/confusion_matrix_open_{species}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    cols = set(df.columns)
    if EPOCH_COLS.issubset(cols):
        plot_training_curves(df)
        print("\ntraining curves.")
    else:
        print("Unrecognised CSV. Columns found:", sorted(cols))