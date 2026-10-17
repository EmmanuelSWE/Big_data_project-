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


def split_by_individual(table, species_name, novel_fraction=0.15, known_test_image_fraction=0.2):
    """
    Two-part split, matching the open-set framing the proposal paper commits to:

    - A fraction of individuals are held out entirely ("novel") - every image of
      theirs goes to test, so there's a real open-set signal (animals the model
      has never seen at all).
    - The rest ("known") individuals keep most of their images in train, but a
      fraction go to test too, so there's also something to measure closed-set
      accuracy against (re-identifying an individual the model DID train on).

    A train-XOR-test split by individual alone (no split within a known
    individual's images) would leave zero test images belonging to a known
    individual, which breaks "known individual" accuracy entirely - every
    individual would look novel at evaluation time.
    """
    images_per_identity = table["identity"].value_counts()
    eligible = images_per_identity[images_per_identity >= MIN_IMAGES_PER_INDIVIDUAL].index.tolist()

    if len(eligible) == 0:
        print(f"  WARNING: no individual of {species_name} has >= {MIN_IMAGES_PER_INDIVIDUAL} images; "
              f"keeping all individuals in train, none in test")
        table = table.copy()
        table["split"] = "train"
        return table

    eligible.sort()  # sorted first so the split is reproducible regardless of row order
    known_identities, novel_identities = train_test_split(
        eligible, test_size=novel_fraction, random_state=RANDOM_SEED
    )
    known_identities, novel_identities = set(known_identities), set(novel_identities)

    table = table[table["identity"].isin(known_identities | novel_identities)].copy()

    split_frames = []
    for identity, group in table.groupby("identity"):
        group = group.copy()
        if identity in novel_identities:
            group["split"] = "test"
        else:
            # Guarantee at least one train image even for a known individual right
            # at MIN_IMAGES_PER_INDIVIDUAL (e.g. 6 images * 0.2 rounds down to 1,
            # which train_test_split already respects, but belt-and-braces here).
            test_count = max(1, min(len(group) - 1, round(len(group) * known_test_image_fraction)))
            test_rows = group.sample(n=test_count, random_state=RANDOM_SEED)
            group["split"] = "train"
            group.loc[test_rows.index, "split"] = "test"
        split_frames.append(group)

    return pd.concat(split_frames, ignore_index=True)


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
