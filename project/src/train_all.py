import torch

from model_b.training import train_one_species

# settings: THIS IS HOW I HAVE SET EVERYTHING FOR THE DATASET, CHECK THE README for dataset setup
# Waiting for final Data preparation.
METADATA_FILE = "data/metadata.csv"
ROOT_FOLDER = "data"
OUTPUT_FOLDER = "models"
SPECIES_LIST = ["hyena", "leopard", "dog", "macaque"]   # must match the 'species' column exactly


if __name__ == "__main__": 
    torch.manual_seed(42)
    for species_name in SPECIES_LIST:
        train_one_species(
            species_name=species_name,
            metadata_file=METADATA_FILE,
            root_folder=ROOT_FOLDER,
            output_folder=OUTPUT_FOLDER,
        )