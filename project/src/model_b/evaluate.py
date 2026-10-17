"""
Evaluation for Model B, matching the open-set protocol the proposal paper commits
to: BAKS (balanced accuracy on known individuals) and BAUS (balanced accuracy on
unknown/novel individuals), combined via geometric mean, plus AUC and TNR@95 for
the open-set accept/reject decision - the same shape of numbers WildlifeReID-10k
reports (adopted as the external benchmark: 81.8% AUC, 38.4% TNR@95 for
ViT-Base/p16 + ArcFace + NMS).

NOTE: our Model B is ViT-Tiny, trained per-species on a 4-species subset under a
CPU training budget. These numbers are NOT directly comparable to that benchmark -
different model size, far less data, no GPU fine-tuning. Report them side by side,
not as an apples-to-apples result.

Run from project/src/ once model_b_<species>.pt checkpoints exist:
    python -m model_b.evaluate
    python -m model_b.evaluate --species hyena leopard
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from model_b.data_loading import build_loaders
from model_b.dataset import build_species_tables
from model_b.training import embed_everything
from model_b.vit_arcface import ViTTinyEmbedder

MODELS_FOLDER = "models"
METADATA_FILE = "data/metadata.csv"
ROOT_FOLDER = "data"
RESULTS_FILE = "model_b/evaluation_results.csv"

# Must match train_all.py's min_images_per_individual, so build_species_tables
# reproduces the exact same individual_names list (and label numbering) the
# checkpoint's gallery_labels were encoded with.
MIN_IMAGES_PER_INDIVIDUAL = 2

# The threshold is picked at the point where 95% of KNOWN photos are accepted
# (TPR=0.95 on the known set), so TNR@95 and BAKS/BAUS all use one consistent
# operating point instead of an arbitrary fixed number.
KNOWN_TPR_TARGET_PERCENTILE = 5  # 5th percentile of known confidences = TPR 0.95


def load_checkpoint(species_name):
    path = os.path.join(MODELS_FOLDER, f"model_b_{species_name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained checkpoint for '{species_name}' at {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ViTTinyEmbedder(embedding_size=checkpoint["embedding_size"], use_pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def balanced_accuracy(group_labels, correct):
    """Mean of each group's OWN accuracy, so individuals with more test photos
    don't drown out individuals with fewer - the imbalance the paper documents."""
    group_labels = np.asarray(group_labels)
    correct = np.asarray(correct)
    per_group = [correct[group_labels == label].mean() for label in np.unique(group_labels)]
    return float(np.mean(per_group))


@torch.no_grad()
def evaluate_species(species_name):
    model, checkpoint = load_checkpoint(species_name)
    device = torch.device("cpu")

    train_table, test_table, individual_names = build_species_tables(
        METADATA_FILE, ROOT_FOLDER, species_name, MIN_IMAGES_PER_INDIVIDUAL
    )
    image_size = checkpoint["image_size"]
    mean, std = checkpoint["normalise_mean"], checkpoint["normalise_std"]

    _, _, test_loader = build_loaders(
        train_table, test_table, image_size, mean, std, batch_size=64, number_of_workers=0
    )

    gallery_embeddings = checkpoint["gallery_embeddings"]
    gallery_labels = checkpoint["gallery_labels"]
    test_embeddings, test_labels = embed_everything(model, test_loader, device)

    # Confidence score per test image = similarity to its single closest gallery
    # image (the "NMS" scoring method from the paper - this is a cosine-similarity
    # embedder, not a softmax classifier, so there's no softmax probability to use).
    similarity = test_embeddings @ gallery_embeddings.T
    confidence, nearest_index = similarity.max(dim=1)
    predicted_labels = gallery_labels[nearest_index]

    test_identities = test_table["identity"].to_numpy()  # same row order as test_loader
    is_known = (test_labels >= 0).numpy()
    confidence = confidence.numpy()
    predicted_labels = predicted_labels.numpy()
    test_labels = test_labels.numpy()

    results = {
        "species": species_name,
        "test_images": len(test_labels),
        "known_test_images": int(is_known.sum()),
        "novel_test_images": int((~is_known).sum()),
    }

    known_confidence = confidence[is_known]
    novel_confidence = confidence[~is_known]

    # AUC: can the confidence score tell a known individual's photo from a novel one?
    if is_known.any() and (~is_known).any():
        results["auc"] = float(roc_auc_score(is_known.astype(int), confidence))
    else:
        results["auc"] = None

    threshold = float(np.percentile(known_confidence, KNOWN_TPR_TARGET_PERCENTILE)) if len(known_confidence) else None
    results["threshold_used"] = threshold

    # TNR@95: at the threshold where 95% of known photos are accepted, what
    # fraction of novel photos get correctly rejected?
    if threshold is not None and len(novel_confidence) > 0:
        results["tnr_at_95"] = float((novel_confidence < threshold).mean())
    else:
        results["tnr_at_95"] = None

    # BAKS: correct only if accepted (confidence >= threshold) AND matched to the
    # right individual, balanced per known individual.
    if threshold is not None and is_known.any():
        known_correct = (confidence[is_known] >= threshold) & (predicted_labels[is_known] == test_labels[is_known])
        results["baks"] = balanced_accuracy(test_identities[is_known], known_correct)
    else:
        results["baks"] = None

    # BAUS: correct if rejected (confidence < threshold), balanced per novel individual.
    if threshold is not None and (~is_known).any():
        novel_correct = confidence[~is_known] < threshold
        results["baus"] = balanced_accuracy(test_identities[~is_known], novel_correct)
    else:
        results["baus"] = None

    if results["baks"] is not None and results["baus"] is not None:
        results["combined_score"] = math.sqrt(results["baks"] * results["baus"])
    else:
        results["combined_score"] = None

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model B specialists (BAKS/BAUS/AUC/TNR@95).")
    parser.add_argument("--species", nargs="*", default=["hyena", "leopard", "macaque", "dog"])
    args = parser.parse_args()

    all_results = []
    for species_name in args.species:
        print(f"\n===== Evaluating {species_name} =====")
        try:
            results = evaluate_species(species_name)
        except FileNotFoundError as error:
            print(f"  skipped: {error}")
            continue
        for key, value in results.items():
            print(f"  {key}: {value}")
        all_results.append(results)

    if not all_results:
        print("\nNo species evaluated - no trained checkpoints found.")
        return

    results_df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nWrote results to {RESULTS_FILE}")

    numeric_columns = ["baks", "baus", "combined_score", "auc", "tnr_at_95"]
    overall = results_df[numeric_columns].mean(numeric_only=True)
    print("\n===== Overall (mean across species) =====")
    print(overall.to_string())
    print(
        "\nNOTE: WildlifeReID-10k's own ViT-Base/p16 + ArcFace + NMS baseline reports "
        "AUC 81.8% / TNR@95 38.4% on its full benchmark - not directly comparable to "
        "these numbers (different model size, 4-species subset, CPU training budget)."
    )


if __name__ == "__main__":
    main()
