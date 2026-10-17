import os

import torch

from model_b.data_loading import build_loaders, get_normalise_settings
from model_b.dataset import build_species_tables
from model_b.vit_arcface import ArcFaceLoss, ViTTinyEmbedder


@torch.no_grad()
def embed_everything(model, loader, device):
    """Runs every image in the loader through the model. Returns (embeddings, labels)."""
    model.eval()
    all_embeddings, all_labels = [], []
    for images, labels in loader:
        embeddings = model(images.to(device))
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)
    if len(all_embeddings) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(all_embeddings), torch.cat(all_labels)


def test_accuracy(gallery_embeddings, gallery_labels, test_embeddings, test_labels):
    """
    Nearest-neighbour identification: for each test image, find the most similar
    training image (cosine similarity) and copy its individual. Only test images of
    animals the model has seen are scored, because a new animal cannot be named correctly.
    """
    known_mask = test_labels >= 0
    if known_mask.sum() == 0:
        return None
    test_embeddings, test_labels = test_embeddings[known_mask], test_labels[known_mask]

    correct = 0
    chunk_size = 1024  # compare in chunks so big species (dogs) do not use huge memory
    for start in range(0, len(test_embeddings), chunk_size):
        chunk = test_embeddings[start:start + chunk_size]
        similarity = chunk @ gallery_embeddings.T          # unit vectors, so this is cosine
        nearest = similarity.argmax(dim=1)
        predicted = gallery_labels[nearest]
        correct += (predicted == test_labels[start:start + chunk_size]).sum().item()
    return correct / len(test_labels)


def train_one_species(
    species_name, metadata_file, root_folder, output_folder="models",
    epochs=30, batch_size=64, embedding_size=128, min_images_per_individual=2,
    backbone_learning_rate=1e-4, head_learning_rate=1e-3, number_of_workers=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"\n===== Training Model B for '{species_name}' on {device} =====")

    # data
    train_table, test_table, individual_names = build_species_tables(
        metadata_file, root_folder, species_name, min_images_per_individual
    )

    # model and loss
    model = ViTTinyEmbedder(embedding_size=embedding_size, use_pretrained=True).to(device)
    loss_function = ArcFaceLoss(embedding_size, len(individual_names)).to(device)

    image_size, mean, std = get_normalise_settings(model)
    train_loader, gallery_loader, test_loader = build_loaders(
        train_table, test_table, image_size, mean, std, batch_size, number_of_workers
    )

    # The pretrained backbone only needs gentle updates. The new head and the ArcFace
    # centres start from scratch, so they get a bigger learning rate.
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_learning_rate},
            {"params": list(model.head.parameters()) + list(loss_function.parameters()),
             "lr": head_learning_rate},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    gradient_scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)  # mixed precision on GPU

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, batch_count = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_cuda):
                embeddings = model(images)
            loss = loss_function(embeddings, labels)

            gradient_scaler.scale(loss).backward()
            gradient_scaler.step(optimizer)
            gradient_scaler.update()

            total_loss += loss.item()
            batch_count += 1
        scheduler.step()
        print(f"[{species_name}] epoch {epoch}/{epochs}  loss {total_loss / batch_count:.4f}")

        # Quick progress check every 5 epochs (this does not choose which model to save)
        if epoch % 5 == 0 or epoch == epochs:
            gallery_embeddings, gallery_labels = embed_everything(model, gallery_loader, device)
            test_embeddings, test_labels = embed_everything(model, test_loader, device)
            accuracy = test_accuracy(gallery_embeddings, gallery_labels, test_embeddings, test_labels)
            if accuracy is not None:
                print(f"[{species_name}] test top-1 accuracy (known individuals): {accuracy:.4f}")

    # save
    # The gallery holds one embedding for every training image. NOTE the database script
    # can load these directly instead of re-running the model over the training set.
    gallery_embeddings, gallery_labels = embed_everything(model, gallery_loader, device)

    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f"model_b_{species_name}.pt")
    torch.save(
        {
            "species": species_name,
            "embedding_size": embedding_size,
            "image_size": image_size,
            "normalise_mean": mean,
            "normalise_std": std,
            "individual_names": individual_names,      # label i  -  individual_names[i]
            "model_state": model.state_dict(),          # backbone + head (what inference needs)
            "arcface_state": loss_function.state_dict(),  # only needed to continue training
            "gallery_embeddings": gallery_embeddings,
            "gallery_labels": gallery_labels,
        },
        save_path,
    )
    print(f"[{species_name}] saved to {save_path}")