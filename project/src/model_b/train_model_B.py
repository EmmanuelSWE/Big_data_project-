"""
Trains one Model B specialist per species, using the images and animals in the database.
"""
import argparse
from pathlib import Path

import torch

from animal_database.wildlife_config import SUB_DATASETS
from model_b.vit_arcface import ArcFaceLoss, IndividualEmbedder
from model_b.database_training import TrainingSettings, train_model_b
from model_b.wildlife_dataset import prepare_species_data

BACKBONE_NAME = "vit_tiny_patch16_224"   # ViT-Tiny. "vit_base_patch16_224" for ViT-Base.
IMAGE_SIZE = 224
EMBEDDING_SIZE = 256
EPOCHS = 30
BATCH_SIZE = 32
NUMBER_OF_WORKERS = 0                   
LEARNING_RATE = 1e-4
CHECKPOINT_FOLDER = Path(__file__).resolve().parent / "checkpoints"


def checkpoint_path_for(species_name):
    return CHECKPOINT_FOLDER / f"model_b_{species_name}.pt"


def train_one_species(species_name, device):
    print(f"\n===== Training Model B for {species_name} =====")
    embedder = IndividualEmbedder(BACKBONE_NAME, EMBEDDING_SIZE, IMAGE_SIZE, use_pretrained_weights=True)
    species_data = prepare_species_data(species_name, embedder, BATCH_SIZE, NUMBER_OF_WORKERS)
    arcface_loss = ArcFaceLoss(EMBEDDING_SIZE, species_data.number_of_identities)  # count comes from the data

    training_settings = TrainingSettings(epochs=EPOCHS, learning_rate=LEARNING_RATE,
                                         checkpoint_path=str(checkpoint_path_for(species_name)))
    train_model_b(embedder, arcface_loss, species_data.train_loader, device, training_settings,
                  species_data.validation_gallery_loader, species_data.validation_query_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", help="train only this species, e.g. dog")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device is {device}")

    species_to_train = [arguments.species] if arguments.species else list(SUB_DATASETS.keys())
    failed_species = []
    for species_name in species_to_train:
        try:
            train_one_species(species_name, device)
        except Exception as error:  # one bad species should not stop the others
            print(f"FAILED for {species_name}: {error}")
            failed_species.append(species_name)
    if failed_species:
        print(f"\nThese species failed: {failed_species}")


if __name__ == "__main__": 
    main()