import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from animal_database.wildlife_config import DATABASE_PATH

def get_species_table_name(species_name):
    return f"species_{species_name}"

TRAIN = "train"                              # known animals: images the model learns from
VALIDATION_KNOWN = "validation_known"        # known animals: held-back images, for tuning
TEST_KNOWN = "test_known"                    # known animals: held-back images, for the final test
VALIDATION_UNKNOWN = "validation_unknown"    # animals the model never sees: for tuning the threshold
TEST_UNKNOWN = "test_unknown"                # animals the model never sees: for the final test

MINIMUM_IDENTITIES_PER_SPECIES = 4  # fewer than this and the 70/15/15 identity split breaks


def load_species_records(species_name, database_path=DATABASE_PATH):
    """Read the species_<name> table: one row per image with image_id, path, identity."""
    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found at {database_path}. "
                                f"Run the scripts from the project root folder.")
    table_name = get_species_table_name(species_name)
    with closing(sqlite3.connect(str(database_path))) as database:
        records = pd.read_sql_query(f"SELECT image_id, path, identity FROM {table_name} ORDER BY image_id",
                                    database)
    if len(records) == 0:
        raise ValueError(f"Table {table_name} is empty. Did the extraction script fail for '{species_name}'?")
    records["identity"] = records["identity"].astype(str)
    return records


def split_identities_like_model_a(identity_names, random_seed=42):
    """
    70% / 15% / 15% groups.
    """
    if len(identity_names) < MINIMUM_IDENTITIES_PER_SPECIES:
        raise ValueError(f"Only {len(identity_names)} individuals; need at least "
                         f"{MINIMUM_IDENTITIES_PER_SPECIES} to split them 70/15/15.")
    known_identities, remaining = train_test_split(identity_names, test_size=0.30, random_state=random_seed)
    validation_unknown_identities, test_unknown_identities = train_test_split(
        remaining, test_size=0.50, random_state=random_seed)
    return known_identities, validation_unknown_identities, test_unknown_identities


def make_species_split(species_name, random_seed=42, held_out_share=0.15, database_path=DATABASE_PATH):
    """
    Returns the species' records with an extra "split" column.
    """
    records = load_species_records(species_name, database_path)
    known_identities, validation_unknown_identities, test_unknown_identities = split_identities_like_model_a(
        records["identity"].drop_duplicates().tolist(), random_seed)

    records["split"] = TRAIN
    records.loc[records["identity"].isin(validation_unknown_identities), "split"] = VALIDATION_UNKNOWN
    records.loc[records["identity"].isin(test_unknown_identities), "split"] = TEST_UNKNOWN

    random_generator = np.random.default_rng(random_seed)
    for identity_name in sorted(known_identities):  # sorted, so the result is the same every run
        row_labels = records.index[records["identity"] == identity_name].to_numpy()
        number_of_images = len(row_labels)
        if number_of_images < 3:
            continue
        # How many images go to validation and to test each. Always leave at least 1 for training.
        number_held_back = max(1, round(number_of_images * held_out_share))
        number_held_back = min(number_held_back, (number_of_images - 1) // 2)
        shuffled_row_labels = random_generator.permutation(row_labels)
        records.loc[shuffled_row_labels[:number_held_back], "split"] = VALIDATION_KNOWN
        records.loc[shuffled_row_labels[number_held_back:2 * number_held_back], "split"] = TEST_KNOWN
    return records


def describe_split(records):
    """Print how many images and animals ended up in each group."""
    summary = records.groupby("split").agg(images=("image_id", "count"), animals=("identity", "nunique"))
    print(summary.to_string())