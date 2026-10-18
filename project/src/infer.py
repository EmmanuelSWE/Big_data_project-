import argparse
import os
import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights

from model_b.vit_arcface import ViTTinyEmbedder
from model_b.dataset import read_metadata
from modelA.model_A import ModelA

# settings: same locations you used for training
MODELS_FOLDER = "models"
METADATA_FILE = r"data\metadata.csv"
ROOT_FOLDER = "data"
MODEL_A_CHECKPOINT = os.path.join("modelA", "checkpoints", "modelA_best.pth")

# Keeps each species model in memory after the first load, so repeated calls
_loaded_models = {}
_model_a_bundle = None


def find_available_species(models_folder=MODELS_FOLDER):
    """Looks in the models folder and returns the species that have a trained model."""
    if not os.path.isdir(models_folder):
        return []
    species_names = []
    for file_name in os.listdir(models_folder):
        if file_name.startswith("model_b_") and file_name.endswith(".pt"):
            species_names.append(file_name[len("model_b_"):-len(".pt")])
    return sorted(species_names)


def load_model_a(checkpoint_path=MODEL_A_CHECKPOINT):
    """Loads Model A (the species classifier) once and keeps it in memory."""
    global _model_a_bundle
    if _model_a_bundle is not None:
        return _model_a_bundle

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No trained Model A checkpoint at '{checkpoint_path}'. "
            "Run modelA/train_model_A.py first."
        )

    # weights_only=False because the file also holds the class_to_ids mapping, not just tensors.
    # It is safe here because we made this file ourselves.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelA()
    model.baseModel.fc.load_state_dict(checkpoint["head_state_dict"])
    model.to(device).eval()

    class_to_ids = checkpoint["class_to_ids"]
    id_to_class = {index: species for species, index in class_to_ids.items()}

    _model_a_bundle = {
        "model": model,
        "device": device,
        "transform": ResNet50_Weights.DEFAULT.transforms(),
        "id_to_class": id_to_class,
    }
    return _model_a_bundle


@torch.no_grad()
def predict_species(image_path):
    """Runs Model A on an image. Returns (species_name, confidence)."""
    bundle = load_model_a()
    image = Image.open(image_path).convert("RGB")
    image_tensor = bundle["transform"](image).unsqueeze(0).to(bundle["device"])

    logits = bundle["model"](image_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    label_index = int(torch.argmax(probabilities))

    return bundle["id_to_class"][label_index], float(probabilities[label_index])


@torch.no_grad()
def identify_image_auto(image_path, top_k=3, threshold=None):
    """Same as identify_image, but detects the species with Model A instead of
    requiring the caller to already know it."""
    species_name, species_confidence = predict_species(image_path)
    result = identify_image(species_name, image_path, top_k=top_k, threshold=threshold)
    result["species_confidence"] = species_confidence
    return result


def load_species_model(species_name, models_folder=MODELS_FOLDER):
    """Loads models/model_b_<species>.pt and prepares it for inference."""
    species_name = species_name.lower().strip()
    if species_name in _loaded_models:
        return _loaded_models[species_name]

    checkpoint_path = os.path.join(models_folder, f"model_b_{species_name}.pt")
    if not os.path.exists(checkpoint_path):
        # Help the user by listing which species models actually exist
        available = sorted(
            name[len("model_b_"):-len(".pt")]
            for name in os.listdir(models_folder)
            if name.startswith("model_b_") and name.endswith(".pt")
        )
        raise FileNotFoundError(
            f"No model found for species '{species_name}'. Available species: {available}"
        )

    # weights_only=False because the file also holds lists and tuples, not just tensors.
    # It is safe here because we made this file ourselves.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild the same network shape as in training, but do NOT download pretrained
    # weights, because we load our own trained weights straight after
    model = ViTTinyEmbedder(embedding_size=checkpoint["embedding_size"], use_pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()   # eval mode: BatchNorm uses its learned statistics

    # The image must be prepared exactly as it was for the gallery during training
    image_size = checkpoint["image_size"]
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["normalise_mean"], checkpoint["normalise_std"]),
    ])

    bundle = {
        "model": model,
        "device": device,
        "image_transform": image_transform,
        "individual_names": checkpoint["individual_names"],      # label i -> name
        "gallery_embeddings": checkpoint["gallery_embeddings"],  # (N, embedding_size)
        "gallery_labels": checkpoint["gallery_labels"],          # (N,) label of each gallery image
    }
    _loaded_models[species_name] = bundle
    return bundle


@torch.no_grad()
def identify_image(species_name, image_path, top_k=3, threshold=None):
    """
    Input : species name + path to an image
    Output: dictionary with the species, the best-matching individual, and the top matches
    """
    species_name = species_name.lower().strip()
    bundle = load_species_model(species_name)

    #  Turn the image into an embedding (a unit-length vector)
    image = Image.open(image_path).convert("RGB")
    image_tensor = bundle["image_transform"](image).unsqueeze(0).to(bundle["device"])
    query_embedding = bundle["model"](image_tensor).cpu()[0]       # shape: (embedding_size,)

    # Cosine similarity to every training image (dot product, since all are unit length)
    similarity_to_gallery = bundle["gallery_embeddings"] @ query_embedding   # shape: (N,)

    # An individual has many gallery images, so score each individual by its BEST image
    number_of_individuals = len(bundle["individual_names"])
    best_score_per_individual = torch.full((number_of_individuals,), -1.0)
    best_score_per_individual.scatter_reduce_(
        0, bundle["gallery_labels"], similarity_to_gallery, reduce="amax"
    )

    # Take the top matches
    top_scores, top_labels = torch.topk(best_score_per_individual, k=min(top_k, number_of_individuals))
    top_matches = [
        {"individual": bundle["individual_names"][label], "similarity": round(score, 4)}
        for score, label in zip(top_scores.tolist(), top_labels.tolist())
    ]

    best = top_matches[0]
    
    # PLEASE CHECK what I did - THIS IS THE APPROACH WE DISCUSSED
    # Optional "unknown animal" rule: if even the best match is not similar enough
    is_unknown = threshold is not None and best["similarity"] < threshold

    return {
        "species": species_name,
        "individual": "New animal" if is_unknown else best["individual"],
        "similarity": best["similarity"],     # 1.0 = identical direction, lower = less similar
        "top_matches": top_matches,
    }


def pick_random_test_image(species_name):
    """Picks a random TEST image of a known individual, so we also know the true answer."""
    bundle = load_species_model(species_name)
    table = read_metadata(METADATA_FILE)
    table = table[(table["species"] == species_name) & (table["split"] == "test")]
    table = table[table["identity"].isin(bundle["individual_names"])]   # only animals the model knows
    row = table.sample(1).iloc[0]
    return os.path.join(ROOT_FOLDER, row["path"]), row["identity"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify an individual animal from an image.")
    parser.add_argument("species", help="hyena, leopard, dog or macaque")
    parser.add_argument("image", nargs="?", help="path to the image (not needed with --random_test)")
    parser.add_argument("--random_test", action="store_true",
                        help="pick a random test image of this species and check the answer")
    parser.add_argument("--top_k", type=int, default=3, help="how many top matches to show")
    parser.add_argument("--threshold", type=float, default=None,
                        help="if the best similarity is below this, answer 'unknown'")
    args = parser.parse_args()

    true_identity = None
    if args.random_test:
        image_path, true_identity = pick_random_test_image(args.species)
        print(f"Random test image: {image_path}")
    elif args.image:
        image_path = args.image
    else:
        parser.error("give an image path, or use --random_test")

    result = identify_image(args.species, image_path, top_k=args.top_k, threshold=args.threshold)

    print(f"\nSpecies   : {result['species']}")
    print(f"Individual: {result['individual']}  (similarity {result['similarity']})")
    print("Top matches:")
    for match in result["top_matches"]:
        print(f"   {match['individual']}   {match['similarity']}")

    if true_identity is not None:
        print(f"\nTrue individual: {true_identity}")
        print("RESULT:", "CORRECT" if result["individual"] == true_identity else "WRONG")