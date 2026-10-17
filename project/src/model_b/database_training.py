"""
TRAINING LOOP for Model B.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from model_b.vit_arcface import IndividualEmbedder


@dataclass
class TrainingSettings:
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    check_every_epochs: int = 1                      # how often to run the validation check
    checkpoint_path: str = "outputs/model_b.pt"


@torch.no_grad()
def embed_everything(embedder, data_loader, device):
    embedder.eval()
    embedding_parts, identity_number_parts = [], []
    for images, identity_numbers in data_loader:
        embedding_parts.append(embedder(images.to(device)))
        identity_number_parts.append(identity_numbers.to(device))
    return torch.cat(embedding_parts), torch.cat(identity_number_parts)


@torch.no_grad()
def measure_rank_one_accuracy(embedder, gallery_loader, query_loader, device):
    gallery_embeddings, gallery_identity_numbers = embed_everything(embedder, gallery_loader, device)
    query_embeddings, query_identity_numbers = embed_everything(embedder, query_loader, device)
    # Embeddings have length 1, so a dot product is the cosine similarity.
    similarity = query_embeddings @ gallery_embeddings.T          # (queries, gallery images)
    closest_gallery_row = similarity.argmax(dim=1)
    closest_identity_numbers = gallery_identity_numbers[closest_gallery_row]
    return (closest_identity_numbers == query_identity_numbers).float().mean().item()


def save_checkpoint(embedder, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": embedder.state_dict(),
                "backbone_name": embedder.backbone_name,
                "embedding_size": embedder.embedding_size,
                "image_size": embedder.image_size}, file_path)


def load_embedder_from_checkpoint(file_path, device):
    checkpoint = torch.load(file_path, map_location=device)
    embedder = IndividualEmbedder(checkpoint["backbone_name"], checkpoint["embedding_size"],
                                  checkpoint["image_size"], use_pretrained_weights=False)
    embedder.load_state_dict(checkpoint["state_dict"])
    return embedder.to(device).eval()


def train_model_b(embedder, arcface_loss, train_loader, device, training_settings: TrainingSettings,
                  validation_gallery_loader: Optional[object] = None,
                  validation_query_loader: Optional[object] = None):
    """
      The training loop.
      embedder      : IndividualEmbedder from vit_arcface.py
      arcface_loss  : ArcFaceLoss from vit_arcface.py
    """
    embedder.to(device)
    arcface_loss.to(device)

    # Both the ViT and the ArcFace centres are learned, so the optimiser gets both.
    parameters_to_train = list(embedder.parameters()) + list(arcface_loss.parameters())
    optimiser = torch.optim.AdamW(parameters_to_train,
                                  lr=training_settings.learning_rate,
                                  weight_decay=training_settings.weight_decay)
    # Learning rate glides from its start value down towards zero over the whole run.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=training_settings.epochs)

    can_validate = validation_gallery_loader is not None and validation_query_loader is not None
    best_accuracy = -1.0

    for epoch_number in range(1, training_settings.epochs + 1):
        embedder.train()
        total_loss, images_seen = 0.0, 0

        for images, identity_numbers in train_loader:
            images = images.to(device)
            identity_numbers = identity_numbers.to(device)

            optimiser.zero_grad()                              # forget the previous step's gradients
            embeddings = embedder(images)                      # photos -> vectors
            loss = arcface_loss(embeddings, identity_numbers)  # how badly are animals mixed up
            loss.backward()                                    # work out how to improve
            optimiser.step()                                   # take the improvement step

            total_loss += loss.item() * len(images)
            images_seen += len(images)

        scheduler.step()
        message = (f"epoch {epoch_number}/{training_settings.epochs}  "
                   f"loss {total_loss / images_seen:.4f}")

        if can_validate and epoch_number % training_settings.check_every_epochs == 0:
            accuracy = measure_rank_one_accuracy(embedder, validation_gallery_loader,
                                                 validation_query_loader, device)
            message += f"  validation rank-1 accuracy {accuracy:.3f}"
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_checkpoint(embedder, training_settings.checkpoint_path)
                message += "  (saved)"
        print(message)

    if not can_validate:
        save_checkpoint(embedder, training_settings.checkpoint_path)
        print(f"Saved model to {training_settings.checkpoint_path}")
    return embedder