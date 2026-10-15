"""
Flask API for the MAHER reidentification demo.

Wires ModelA (species classifier) and ModelB (individual embedder) to the HTTP
endpoints the frontend needs for the upload -> identify -> remove flow from
Docs/BIg_Data-Docs.drawio.

NOTE: neither model has been trained on real data yet. ModelA's classification
head and ModelB's projection layer are randomly initialised, so predictions are
not meaningful yet -- this wires up the real model code and the full pipeline so
the UI/API can be built and tested now, ahead of training.
"""
import os
import sys
import uuid

import numpy as np
import tensorflow as tf
import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modelA.model_A import MODEL_INPUT, ModelA  # noqa: E402
from modelB.model_b import IndividualEmbedder  # noqa: E402

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

# Placeholder species labels -- replace with the real class names once a
# labelled dataset exists. Order must match ModelA's output layer (MODEL_OUTPUT).
SPECIES_LABELS = ["Species A", "Species B", "Species C", "Species D"]

# Cosine similarity above this counts as "same individual" as an existing entry.
MATCH_THRESHOLD = 0.6

app = Flask(__name__, static_folder=None)

_model_a = None
_model_b = None
_gallery = []  # [{id, species, embedding: np.ndarray}]


def get_model_a():
    global _model_a
    if _model_a is None:
        _model_a = ModelA().model
    return _model_a


def get_model_b():
    global _model_b
    if _model_b is None:
        _model_b = IndividualEmbedder()
        _model_b.eval()
    return _model_b


def load_image(file_storage):
    image = Image.open(file_storage.stream).convert("RGB")
    return image.resize(MODEL_INPUT[:2])


def predict_species(image):
    array = np.array(image, dtype=np.float32)
    array = tf.keras.applications.resnet50.preprocess_input(array)
    batch = np.expand_dims(array, axis=0)
    probabilities = get_model_a().predict(batch, verbose=0)[0]
    label_index = int(np.argmax(probabilities))
    return SPECIES_LABELS[label_index], float(probabilities[label_index])


def embed_individual(image):
    array = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        embedding = get_model_b()(tensor)[0]
    return embedding.numpy()


def match_individual(embedding, species):
    best_id, best_score = None, -1.0
    for animal in _gallery:
        score = float(np.dot(embedding, animal["embedding"]))
        if score > best_score:
            best_id, best_score = animal["id"], score

    if best_id is not None and best_score >= MATCH_THRESHOLD:
        return {"id": best_id, "isNew": False, "confidence": best_score}

    new_id = str(uuid.uuid4())[:8]
    _gallery.append({"id": new_id, "species": species, "embedding": embedding})
    return {"id": new_id, "isNew": True, "confidence": best_score if best_id else None}


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/identify", methods=["POST"])
def identify():
    if "image" not in request.files:
        return jsonify({"error": "no image file provided (expected form field 'image')"}), 400

    image = load_image(request.files["image"])
    species_label, species_confidence = predict_species(image)
    embedding = embed_individual(image)
    individual = match_individual(embedding, species_label)

    return jsonify({
        "species": {"label": species_label, "confidence": species_confidence},
        "individual": individual,
    })


@app.route("/api/animals", methods=["GET"])
def list_animals():
    return jsonify([{"id": a["id"], "species": a["species"]} for a in _gallery])


@app.route("/api/animals/<animal_id>", methods=["DELETE"])
def remove_animal(animal_id):
    global _gallery
    before = len(_gallery)
    _gallery = [a for a in _gallery if a["id"] != animal_id]
    if len(_gallery) == before:
        return jsonify({"error": "animal not found"}), 404
    return jsonify({"removed": animal_id})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
