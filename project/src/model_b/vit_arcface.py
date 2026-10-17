"""
Model B for MAHER: the individual-animal embedding model.

  1. IndividualEmbedder : a Vision Transformer that turns a photo into a vector.
  2. ArcFaceLoss        : the loss used to train that vector so photos of the same animal
                          end up close together and different animals end up far apart.

"""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as functional


class IndividualEmbedder(nn.Module):
    """
    Photo in: embedding vector out.

    Input : a batch of images, shape (batch_size, 3, image_size, image_size),
            already resized and normalised.
    Output: a batch of embeddings, shape (batch_size, embedding_size),
            each one scaled to length 1.
    """

    def __init__(self,
                 backbone_name="vit_tiny_patch16_224",  # which ViT to use
                 embedding_size=512,                    # how many numbers describe one photo
                 image_size=224,                        # width and height the ViT expects
                 use_pretrained_weights=True):          # start from pretrained weights
        super().__init__()

        # The ViT backbone from timm.
        # num_classes=0 removes timm's own classifier, so the backbone returns raw features
        # instead of class scores.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=use_pretrained_weights,
            num_classes=0,
            img_size=image_size,
        )

        # A small layer that shrinks the ViT features down to "embedding_size" numbers.
        self.projection = nn.Linear(self.backbone.num_features, embedding_size)

        self.backbone_name = backbone_name
        self.embedding_size = embedding_size
        self.image_size = image_size

    def forward(self, images):
        features = self.backbone(images)
        embeddings = self.projection(features)
        # Scale every embedding to length 1. After this, the dot product of two embeddings
        # equals their cosine similarity, which is the "how alike are these two photos" score.
        return functional.normalize(embeddings, dim=1)


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss, used only while training.

    Every known animal gets its own "centre" direction (a row in "identity_weights").
    ArcFace adds an angle penalty (the margin) to the correct animal, so a photo must be closer
    to its own animal by a safe gap. That pushes photos of one animal into a tight group and
    keeps different animals well apart.

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
        # reward a WORSE match. These two numbers switch to a safe version in that rare case.
        self.cosine_limit = math.cos(math.pi - margin)
        self.fallback_shift = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, identity_numbers):
        # Cosine similarity between every embedding and every animal's centre.
        cosine = functional.linear(functional.normalize(embeddings),
                                   functional.normalize(self.identity_weights))
        # acos() fails at exactly -1 or 1, so keep the values just inside that range.
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Add the margin to the angle of the CORRECT animal only.
        angle = torch.acos(cosine)
        cosine_with_margin = torch.cos(angle + self.margin)
        cosine_with_margin = torch.where(cosine > self.cosine_limit,
                                         cosine_with_margin,
                                         cosine - self.fallback_shift)

        # Use the margin version for the correct animal, the plain cosine for all the others.
        is_correct_identity = functional.one_hot(identity_numbers, self.number_of_identities).bool()
        scores = torch.where(is_correct_identity, cosine_with_margin, cosine) * self.scale

        return functional.cross_entropy(scores, identity_numbers)


# Self-test with random data.
if __name__ == "__main__":
    batch_size, number_of_identities, image_size, embedding_size = 8, 5, 64, 32
    embedder = IndividualEmbedder("vit_tiny_patch16_224", embedding_size, image_size, use_pretrained_weights=False)
    arcface_loss = ArcFaceLoss(embedding_size, number_of_identities)
    embeddings = embedder(torch.randn(batch_size, 3, image_size, image_size))
    loss = arcface_loss(embeddings, torch.randint(0, number_of_identities, (batch_size,))) # loss
    loss.backward() # backpropag
    print("embedding shape:", tuple(embeddings.shape), " loss:", round(loss.item(), 4))