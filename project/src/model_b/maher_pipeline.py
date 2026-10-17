import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights

from modelA.model_A import ModelA
from model_b.database_data import TRAIN, VALIDATION_KNOWN, VALIDATION_UNKNOWN, make_species_split
from model_b.train_model_B import CHECKPOINT_FOLDER, checkpoint_path_for
from model_b.database_training import load_embedder_from_checkpoint
from model_b.wildlife_dataset import get_normalisation, make_evaluation_transform

MODEL_A_CHECKPOINT = Path(__file__).resolve().parent.parent / "modelA" / "checkpoints" / "modelA_best.pth"


def list_available_species():
    species_names = [path.stem[len("model_b_"):] for path in CHECKPOINT_FOLDER.glob("model_b_*.pt")]
    return sorted(name for name in species_names if not name.endswith("_gallery"))


def choose_new_individual_threshold(known_best_scores, known_is_correct, unknown_best_scores):
    known_best_scores = np.asarray(known_best_scores, dtype=float)
    known_is_correct = np.asarray(known_is_correct, dtype=bool)
    unknown_best_scores = np.asarray(unknown_best_scores, dtype=float)
    all_scores = np.concatenate([known_best_scores, unknown_best_scores])
    best_threshold, best_value = float(all_scores.min()), -1.0
    for candidate in np.linspace(all_scores.min(), all_scores.max(), 200):
        known_rate = np.mean((known_best_scores >= candidate) & known_is_correct)
        unknown_rate = np.mean(unknown_best_scores < candidate)
        value = (known_rate + unknown_rate) / 2
        if value > best_value:
            best_threshold, best_value = float(candidate), float(value)
    return best_threshold


@torch.no_grad()
def embed_image_paths(embedder, image_paths, image_transform, device, batch_size=32):
    embedder.eval()
    embedding_parts = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch = torch.stack([image_transform(Image.open(path).convert("RGB")) for path in batch_paths])
        embedding_parts.append(embedder(batch.to(device)).cpu().numpy())
    return np.concatenate(embedding_parts).astype(np.float32)


class SpeciesSpecialist:

    def __init__(self, species_name, device):
        self.species_name = species_name
        self.embedder = load_embedder_from_checkpoint(checkpoint_path_for(species_name), device)
        mean, standard_deviation = get_normalisation(self.embedder)
        self.image_transform = make_evaluation_transform(self.embedder.image_size, mean, standard_deviation)
        self.device = device

        records = make_species_split(species_name) 

        def embed(rows):
            return embed_image_paths(self.embedder, rows["path"].tolist(), self.image_transform, device)

        train_rows = records[records["split"] == TRAIN]
        known_query_rows = records[records["split"] == VALIDATION_KNOWN]
        unknown_query_rows = records[records["split"] == VALIDATION_UNKNOWN]
        train_embeddings = embed(train_rows)
        train_identities = train_rows["identity"].to_numpy()

        known_embeddings = embed(known_query_rows)
        known_similarity = known_embeddings @ train_embeddings.T
        known_closest = known_similarity.argmax(axis=1)
        known_best_scores = known_similarity[np.arange(len(known_closest)), known_closest]
        known_is_correct = train_identities[known_closest] == known_query_rows["identity"].to_numpy()

        unknown_embeddings = embed(unknown_query_rows)
        unknown_best_scores = (unknown_embeddings @ train_embeddings.T).max(axis=1)

        self.new_individual_threshold = choose_new_individual_threshold(
            known_best_scores, known_is_correct, unknown_best_scores)

        self.gallery_embeddings = np.concatenate([train_embeddings, known_embeddings])
        self.gallery_identities = np.concatenate([train_identities, known_query_rows["identity"].to_numpy()])

    def known_individuals(self):
        return sorted(set(self.gallery_identities.tolist()))

    def identify(self, pil_image, top_k=5, threshold=None):
        tensor = self.image_transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.embedder(tensor).cpu().numpy()[0]
        similarity = self.gallery_embeddings @ embedding  

        best_per_animal = (pd.Series(similarity).groupby(self.gallery_identities).max()
                           .sort_values(ascending=False))
        top_matches = [{"individual": str(name), "similarity": round(float(score), 4)}
                       for name, score in best_per_animal.head(top_k).items()]

        closest_individual = top_matches[0]["individual"]
        match_score = float(best_per_animal.iloc[0])
        threshold_used = self.new_individual_threshold if threshold is None else float(threshold)
        is_new_individual = match_score < threshold_used
        return {"is_new_individual": is_new_individual,
                "individual": None if is_new_individual else closest_individual,
                "closest_known_individual": closest_individual,
                "match_score": match_score,
                "threshold": threshold_used,
                "top_matches": top_matches,
                "gallery_images_compared": int(len(self.gallery_identities))}


class MaherPipeline:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not MODEL_A_CHECKPOINT.is_file():
            raise FileNotFoundError(f"Model A checkpoint not found: {MODEL_A_CHECKPOINT}")

        # Rebuild Model A exactly as test_model_A.py does.
        checkpoint = torch.load(MODEL_A_CHECKPOINT, map_location=self.device, weights_only=True)
        class_to_ids = checkpoint["class_to_ids"]
        self.species_names = sorted(class_to_ids, key=class_to_ids.get)
        self.model_a = ModelA().to(self.device)
        self.model_a.baseModel.fc.load_state_dict(checkpoint["head_state_dict"])
        self.model_a.eval()
        self.model_a_transform = ResNet50_Weights.DEFAULT.transforms()

        self.specialists = {} 

    def warm_up(self):
        for species_name in list_available_species():
            print(f"[pipeline] loading specialist for {species_name} ...", flush=True)
            self.get_specialist(species_name)

    def predict_species(self, pil_image):
        tensor = self.model_a_transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model_a(tensor), dim=1)[0]
        confidence, species_number = probabilities.max(dim=0)
        return self.species_names[int(species_number)], float(confidence)

    def get_specialist(self, species_name):
        if species_name not in self.specialists:
            if not checkpoint_path_for(species_name).is_file():
                raise FileNotFoundError(f"No Model B specialist for '{species_name}'. "
                                        f"Available species: {list_available_species()}")
            self.specialists[species_name] = SpeciesSpecialist(species_name, self.device)
        return self.specialists[species_name]

    def predict_image(self, pil_image, species_name=None, threshold=None):
        if species_name is None:
            species_name, species_confidence = self.predict_species(pil_image)
        else:
            species_confidence = 1.0   # species given by the caller
        result = self.get_specialist(species_name).identify(pil_image, threshold=threshold)
        return {"species": species_name, "species_confidence": species_confidence, **result}

    def predict(self, image_path, species_name=None, threshold=None):
        return self.predict_image(Image.open(image_path).convert("RGB"), species_name, threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to a photo")
    parser.add_argument("--species", help="skip Model A and use this species")
    arguments = parser.parse_args()
    print(json.dumps(MaherPipeline().predict(arguments.image, arguments.species), indent=2))