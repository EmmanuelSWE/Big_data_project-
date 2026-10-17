import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Column order of the metadata.csv rows (the sample had no header row).
COLUMN_NAMES = [
    "identity", "path", "date", "orientation", "species", "split", "dataset", "extra",
]


def read_metadata(metadata_file):
    """Reads metadata.csv whether or not it has a header row."""
    table = pd.read_csv(metadata_file, dtype=str, keep_default_na=False)
    if "species" in table.columns and "identity" in table.columns:
        return table  # the file has its own header row

    # No header: re-read and name the columns ourselves
    table = pd.read_csv(
        metadata_file, header=None, names=COLUMN_NAMES, usecols=range(len(COLUMN_NAMES)),
        dtype=str, keep_default_na=False,
    )
    return table


def build_species_tables(metadata_file, root_folder, species_name, min_images_per_individual=2):
    """
    Returns everything Model B needs for ONE species:
      train_table, test_table : pandas tables with a full_path and a label column
      individual_names        : list where position i is the name of label i
    """
    table = read_metadata(metadata_file)

    # Keep only this species
    table = table[table["species"] == species_name].copy()
    if len(table) == 0:
        raise ValueError(f"No rows found for species '{species_name}' in {metadata_file}")

    # Build the real path on disk and drop any rows whose image file is missing
    table["full_path"] = table["path"].apply(lambda p: os.path.join(root_folder, p))
    file_exists = table["full_path"].apply(os.path.exists)
    if (~file_exists).sum() > 0:
        print(f"[{species_name}] skipping {(~file_exists).sum()} rows whose image file is missing")
    table = table[file_exists]

    train_table = table[table["split"] == "train"].copy()
    test_table = table[table["split"] == "test"].copy()

    # An animal with only one training image gives ArcFace almost nothing to learn from,
    # so we keep individuals with at least "min_images_per_individual" training images
    images_per_individual = train_table["identity"].value_counts()
    kept_individuals = images_per_individual[images_per_individual >= min_images_per_individual].index
    train_table = train_table[train_table["identity"].isin(kept_individuals)].copy()

    if len(train_table) == 0:
        raise ValueError(
            f"[{species_name}] no individual has {min_images_per_individual}+ training images. "
            "Lower min_images_per_individual."
        )

    # Turn each individual's name into a number (label) and keep the name list for later
    individual_names = sorted(train_table["identity"].unique())
    name_to_label = {name: number for number, name in enumerate(individual_names)}
    train_table["label"] = train_table["identity"].map(name_to_label)

    # Test animals the model never trained on get label -1 (they are "new" animals)
    test_table["label"] = test_table["identity"].map(name_to_label).fillna(-1).astype(int)

    print(
        f"[{species_name}] {len(individual_names)} individuals, "
        f"{len(train_table)} train images, {len(test_table)} test images "
        f"({(test_table['label'] >= 0).sum()} of the test images are of known individuals)"
    )
    return train_table, test_table, individual_names


class IndividualImageDataset(Dataset):
    """Gives (image_tensor, label) for every row in a species table."""

    def __init__(self, table, image_transform):
        self.image_paths = table["full_path"].tolist()
        self.labels = table["label"].tolist()
        self.image_transform = image_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")  # png and jpg both work
        return self.image_transform(image), self.labels[index]