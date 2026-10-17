import timm
from torch.utils.data import DataLoader
from torchvision import transforms

from model_b.dataset import IndividualImageDataset


def get_normalise_settings(model):
    """Asks timm which image size / mean / std this pretrained ViT expects."""
    config = timm.data.resolve_data_config({}, model=model.backbone)
    return config["input_size"][1], config["mean"], config["std"]


def build_transforms(image_size, mean, std):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # Used for testing and for building the gallery: no randomness
    plain_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, plain_transform


def build_loaders(train_table, test_table, image_size, mean, std, batch_size, number_of_workers):
    train_transform, plain_transform = build_transforms(image_size, mean, std)

    train_data = IndividualImageDataset(train_table, train_transform)
    gallery_data = IndividualImageDataset(train_table, plain_transform)  # same images, no augmentation
    test_data = IndividualImageDataset(test_table, plain_transform)

    # BatchNorm needs more than one image per batch, so drop a leftover batch of size 1,
    # but only if we have enough images to still make at least one full batch
    drop_last = len(train_data) > batch_size

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=number_of_workers, drop_last=drop_last, pin_memory=True)
    gallery_loader = DataLoader(gallery_data, batch_size=batch_size, shuffle=False,
                                num_workers=number_of_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=number_of_workers, pin_memory=True)
    return train_loader, gallery_loader, test_loader