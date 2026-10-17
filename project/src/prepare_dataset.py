"""
Builds data/metadata.csv - the single data source both Model A
(modelA/train_model_A.py) and Model B (train_all.py / model_b/dataset.py) train
from, with a train/test split that never splits one individual across both.

Downloads each species in animal_database.wildlife_config.SUB_DATASETS via the
wildlife_datasets package, into data/images/<species>/, and writes
data/metadata.csv with columns: identity, path, date, orientation, species,
split, dataset, extra (same convention model_b/dataset.py already reads).

Run from project/src/:
    python prepare_dataset.py
"""
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "animal_database"))
from wildlife_config import SUB_DATASETS, MIN_IMAGES_PER_INDIVIDUAL, RANDOM_SEED  # noqa: E402

ROOT_FOLDER = os.path.join(os.path.dirname(__file__), "data")
IMAGES_DIRECTORY = os.path.join(ROOT_FOLDER, "images")
METADATA_FILE = os.path.join(ROOT_FOLDER, "metadata.csv")

TEST_FRACTION = 0.15  # fraction of each species' INDIVIDUALS held out for test, not images

COLUMN_NAMES = ["identity", "path", "date", "orientation", "species", "split", "dataset", "extra"]


def download_and_label_species(species_name, dataset_class):
    species_directory = os.path.join(IMAGES_DIRECTORY, species_name)
    os.makedirs(species_directory, exist_ok=True)

    print(f"Downloading and processing: {species_name}...")
    dataset_class.get_data(species_directory)
    loader = dataset_class(species_directory)

    table = loader.df.copy()
    table["species"] = species_name

    # Drop images with no known individual - neither model can learn from those.
    unknown_label = getattr(loader, "unknown_name", "unknown")
    table = table[table["identity"].notna() & (table["identity"] != unknown_label)].copy()
    table["identity"] = table["identity"].astype(str)

    # Store paths relative to ROOT_FOLDER (what model_b/dataset.py and
    # modelA/train_model_A.py both join root_folder with).
    table["path"] = table["path"].apply(
        lambda relative_path: os.path.join("images", species_name, str(relative_path))
    )

    print(f"  {species_name}: {len(table)} labelled images, {table['identity'].nunique()} individuals")
    return table[["identity", "path", "species"]]


def split_by_individual(table, species_name):
    """Train/test split at the individual level, so no individual's images end up
    on both sides - the leakage the proposal paper specifically calls out."""
    images_per_identity = table["identity"].value_counts()
    eligible = images_per_identity[images_per_identity >= MIN_IMAGES_PER_INDIVIDUAL].index.tolist()

    if len(eligible) == 0:
        print(f"  WARNING: no individual of {species_name} has >= {MIN_IMAGES_PER_INDIVIDUAL} images; "
              f"keeping all individuals in train, none in test")
        table = table.copy()
        table["split"] = "train"
        return table

    eligible.sort()  # sorted first so the split is reproducible regardless of row order
    train_identities, test_identities = train_test_split(
        eligible, test_size=TEST_FRACTION, random_state=RANDOM_SEED
    )
    train_identities, test_identities = set(train_identities), set(test_identities)

    table = table[table["identity"].isin(train_identities | test_identities)].copy()
    table["split"] = table["identity"].apply(lambda i: "train" if i in train_identities else "test")
    return table


def main():
    os.makedirs(IMAGES_DIRECTORY, exist_ok=True)

    species_frames = []
    for species_name, dataset_class in SUB_DATASETS.items():
        table = download_and_label_species(species_name, dataset_class)
        table = split_by_individual(table, species_name)
        species_frames.append(table)

    combined = pd.concat(species_frames, ignore_index=True)

    # Columns model_b/dataset.py and modelA/train_model_A.py expect, even though
    # this project doesn't use date/orientation/dataset/extra yet.
    combined["date"] = ""
    combined["orientation"] = ""
    combined["dataset"] = combined["species"]
    combined["extra"] = ""
    combined = combined[COLUMN_NAMES]

    combined.to_csv(METADATA_FILE, index=False)
    print(f"\nWrote {len(combined)} rows to {METADATA_FILE}")
    print(combined.groupby(["species", "split"]).size())


if __name__ == "__main__":
    main()
