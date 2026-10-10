"""
Model B for MAHER: the individual-animal embedding model.

  1. IndividualEmbedder : a Vision Transformer (ViT-Base/16) that turns a photo into a vector.
  2. ArcFaceLoss        : the loss used to train that vector so photos of the same animal
                          end up close together and different animals end up far apart.

NOTE: This module needs timm:  pip install torch timm.
    : I did not create the training loop code, the dataset loader code.
"""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as functional


class IndividualEmbedder(nn.Module):
    """
    Input : a batch of images, shape (batch_size, 3, image_size, image_size),
            already resized and normalised (a teammate's data code does that).
    Output: a batch of embeddings, shape (batch_size, embedding_size),
            each one scaled to length 1.
    """

    def __init__(self,
                 backbone_name="vit_base_patch16_224",  # ViT-Base, 16x16 patches as in our proposal 
                 embedding_size=512,                    # these are the numbers which will discribe an image
                 image_size=224,                        # width and height the ViT expects
                 use_pretrained_weights=True):          # start from ImageNet weights
        super().__init__()

        # The ViT backbone from timm.
        # num_classes=0 removes timm's own classifier, so the backbone returns raw features
        # (768 numbers per image for ViT-Base) instead of class scores.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=use_pretrained_weights,
            num_classes=0,
            img_size=image_size,
        )

        # A small layer that shrinks the 768 ViT features down to embedding_size numbers.
        self.projection = nn.Linear(self.backbone.num_features, embedding_size)

        # we keep these so other code can read them (for example when saving or loading).
        self.backbone_name = backbone_name
        self.embedding_size = embedding_size
        self.image_size = image_size

    def forward(self, images):
        features = self.backbone(images)          # (batch_size, 768)
        embeddings = self.projection(features)    # (batch_size, embedding_size)
        # Scale every embedding to length 1. After this, the dot product of two embeddings
        # equals their cosine similarity, which is the "how alike are these two photos" score
        # used later when comparing a new photo against the known animals.
        return functional.normalize(embeddings, dim=1) # (THIS IS IMPORTANT -> @ TEAM PLEASE CHECK IT and make sure that you understand it)


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss, used only while training.

    Idea: every known animal gets its own "centre" direction (a row in identity_weights).
    A normal classifier only asks "is this photo closest to the right animal's centre?".
    ArcFace is stricter: it adds an angle penalty (the margin) to the correct animal, so the
    photo must be closer to its own animal by a safe gap. That pushes photos of one animal
    into a tight group and keeps different animals well apart. This is the 

    Input : embeddings (batch_size, embedding_size), and identity_numbers (batch_size,)
            which are whole numbers 0 .. number_of_identities-1, one per photo.
    Output: a single loss value (a tensor you call .backward() on).
    """

    def __init__(self,
                 embedding_size,
                 number_of_identities,      # how many different animals are in the training data
                 scale=30.0,                # makes the scores bigger so the softmax can learn sharply
                 margin=0.5):               # the angle penalty, in radians
        super().__init__()

        # One learnable direction per animal.
        self.identity_weights = nn.Parameter(torch.empty(number_of_identities, embedding_size))
        nn.init.xavier_uniform_(self.identity_weights)

        self.number_of_identities = number_of_identities
        self.scale = scale
        self.margin = margin

        # When angle + margin goes past 180 degrees, cos() starts going back up, which would
        # reward a WORSE match. These two numbers let us switch to a safe straight-line
        # version in that rare case.
        self.cosine_limit = math.cos(math.pi - margin)
        self.fallback_shift = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, identity_numbers):
        # Cosine similarity between every embedding and every animal's centre.
        # Shape: (batch_size, number_of_identities)
        cosine = functional.linear(functional.normalize(embeddings),
                                   functional.normalize(self.identity_weights))
        
        # acos() fails at exactly -1 or 1, so keep the values just inside that range.
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Add the margin to the angle of the correct animal only.
        angle = torch.acos(cosine)
        cosine_with_margin = torch.cos(angle + self.margin)
        cosine_with_margin = torch.where(cosine > self.cosine_limit,
                                         cosine_with_margin,
                                         cosine - self.fallback_shift)

        # Use the margin version for the correct animal, the plain cosine for all the others.
        is_correct_identity = functional.one_hot(identity_numbers, self.number_of_identities).bool()
        scores = torch.where(is_correct_identity, cosine_with_margin, cosine) * self.scale

        # Standard cross-entropy on those scores.
        return functional.cross_entropy(scores, identity_numbers)



# Testing the model: Please note that this is not a training,  just the ArcFaceLoss and embedding.
if __name__ == "__main__":
    batch_size, number_of_identities, image_size, embedding_size = 8, 5, 64, 32

    embedder = IndividualEmbedder(backbone_name="vit_tiny_patch16_224",
                                  embedding_size=embedding_size,
                                  image_size=image_size,
                                  use_pretrained_weights=False)
    arcface_loss = ArcFaceLoss(embedding_size, number_of_identities)

    fake_images = torch.randn(batch_size, 3, image_size, image_size)
    fake_identity_numbers = torch.randint(0, number_of_identities, (batch_size,))

    embeddings = embedder(fake_images)
    loss = arcface_loss(embeddings, fake_identity_numbers)
    loss.backward()

    print("embedding shape :", tuple(embeddings.shape))                       # (8, 32)
    print("embedding length:", embeddings.norm(dim=1)[:3].tolist())           # all about 1.0
    print("loss value      :", round(loss.item(), 4))
    print("gradient reached the ViT:", embedder.projection.weight.grad is not None)