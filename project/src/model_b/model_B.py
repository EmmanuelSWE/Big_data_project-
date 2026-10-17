import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTTinyEmbedder(nn.Module):
    """ViT-Tiny backbone that turns an image into a unit-length embedding vector."""

    def __init__(self, embedding_size=128, use_pretrained=True):
        super().__init__()
        # num_classes=0 removes timm's ImageNet classifier, so the backbone
        # returns its 192-number feature vector.
        self.backbone = timm.create_model(
            "vit_tiny_patch16_224", pretrained=use_pretrained, num_classes=0
        )
        backbone_size = self.backbone.num_features  # 192 for ViT-Tiny

        # Small head: 192 to embedding_size, then BatchNorm to keep values well scaled
        self.head = nn.Sequential(
            nn.Linear(backbone_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, images):
        features = self.backbone(images)
        embedding = self.head(features)
        # Unit length, so a dot product between two embeddings equals their cosine similarity.
        return F.normalize(embedding, dim=1)


class ArcFaceLoss(nn.Module):
    """
    ArcFace: a softmax loss where the correct individual's angle gets an extra margin added.
    This pushes embeddings of the same animal together and different animals apart.
    """

    def __init__(self, embedding_size, number_of_individuals, scale=30.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin

        # One learnable "centre" direction per individual animal
        self.class_centres = nn.Parameter(torch.empty(number_of_individuals, embedding_size))
        nn.init.xavier_uniform_(self.class_centres)

        # Pre-computed numbers for the angle-plus-margin formula
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)
        # If the angle is already so large that adding the margin would wrap past 180 degrees,
        # uses a simple linear penalty instead, which keeps the loss well behaved
        self.angle_limit = math.cos(math.pi - margin)
        self.fallback_penalty = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        embeddings = embeddings.float()  # stay in float32 even when mixed precision is on

        # Cosine between every embedding and every class centre
        cosine = F.linear(embeddings, F.normalize(self.class_centres, dim=1))
        sine = torch.sqrt((1.0 - cosine ** 2).clamp(min=1e-7))

        # cos(angle + margin) = cos(angle)cos(margin) - sin(angle)sin(margin)
        cosine_with_margin = cosine * self.cos_margin - sine * self.sin_margin
        cosine_with_margin = torch.where(
            cosine > self.angle_limit, cosine_with_margin, cosine - self.fallback_penalty
        )

        # Apply the margin only to the correct individual, leave the others unchanged
        correct_mask = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = correct_mask * cosine_with_margin + (1.0 - correct_mask) * cosine
        logits = logits * self.scale

        return F.cross_entropy(logits, labels)