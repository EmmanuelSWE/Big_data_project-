from dataclasses import dataclass
from typing import Optional

import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model_b.database_data import TRAIN, VALIDATION_KNOWN, describe_split, make_species_split


def get_normalisation(embedder):
    config = timm.data.resolve_data_config({}, model=embedder.backbone)
    return config["mean"], config["std"]


def make_training_transform(image_size, mean, standard_deviation):
    return transforms.Compose([
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, standard_deviation),
    ])


def make_evaluation_transform(image_size, mean, standard_deviation):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, standard_deviation),
    ])


class AnimalDatabaseDataset(Dataset):
    def __init__(self, records, image_transform, identity_number_by_name):
        self.image_paths = records["path"].tolist()  # full paths, straight from the database
        self.identity_numbers = [identity_number_by_name.get(name, -1) for name in records["identity"]]
        self.image_transform = image_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.image_transform(image), self.identity_numbers[index]


@dataclass
class SpeciesData:
    species_name: str
    records: object                                   # the full table with its "split" column
    train_loader: DataLoader
    number_of_identities: int
    identity_number_by_name: dict
    validation_gallery_loader: Optional[DataLoader]   # training images, no augmentation
    validation_query_loader: Optional[DataLoader]     # held-back images of known animals


def make_loader(dataset, batch_size, number_of_workers, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=number_of_workers,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=shuffle and len(dataset) > batch_size)


def prepare_species_data(species_name, embedder, batch_size=32, number_of_workers=0, random_seed=42):
    records = make_species_split(species_name, random_seed=random_seed)
    describe_split(records)

    train_records = records[records["split"] == TRAIN]
    validation_records = records[records["split"] == VALIDATION_KNOWN]

    # ArcFace learns exactly the animals in the training images: number them 0 .. N-1.
    identity_names = sorted(train_records["identity"].unique())
    identity_number_by_name = {name: number for number, name in enumerate(identity_names)}

    mean, standard_deviation = get_normalisation(embedder)
    training_transform = make_training_transform(embedder.image_size, mean, standard_deviation)
    evaluation_transform = make_evaluation_transform(embedder.image_size, mean, standard_deviation)

    train_loader = make_loader(AnimalDatabaseDataset(train_records, training_transform, identity_number_by_name),
                               batch_size, number_of_workers, shuffle=True)
    gallery_loader = make_loader(AnimalDatabaseDataset(train_records, evaluation_transform, identity_number_by_name),
                                 batch_size, number_of_workers, shuffle=False)
    query_loader = make_loader(AnimalDatabaseDataset(validation_records, evaluation_transform, identity_number_by_name),
                               batch_size, number_of_workers, shuffle=False)
    return SpeciesData(species_name, records, train_loader, len(identity_names), identity_number_by_name,
                       gallery_loader, query_loader)