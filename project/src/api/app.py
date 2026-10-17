import os
import sys
import threading

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC_DIR)

FRONTEND_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "frontend"))

ANIMAL_DATABASE_DIR = os.path.join(SRC_DIR, "animal_database")
if not os.path.isfile(os.path.join(ANIMAL_DATABASE_DIR, "data_environment", "wildlife_animals_database.db")):
    raise FileNotFoundError(f"Database not found inside {ANIMAL_DATABASE_DIR}\\data_environment")
os.chdir(ANIMAL_DATABASE_DIR)

from model_b.maher_pipeline import MaherPipeline, list_available_species   # noqa: E402  (after chdir/path setup)

app = Flask(__name__, static_folder=None)

# Load Model A and every Model B specialist once, when the server starts (not on every request).
pipeline = MaherPipeline()
pipeline.warm_up()
print(f"[api] ready. Species with a specialist: {list_available_species()}", flush=True)

# The GPU model objects are shared, so handle one photo at a time.
inference_lock = threading.Lock()


def species_info(label, confidence):
    return {"label": label, "confidence": round(confidence, 4)}


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
    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
    except Exception as error:
        return jsonify({"error": f"cannot open image: {error}"}), 400

    threshold = request.form.get("threshold", type=float)

    with inference_lock:
        try:
            species_label, species_confidence = pipeline.predict_species(image)
        except Exception as error:
            return jsonify({"error": f"Model A failed: {error}"}), 500

        try:
            specialist = pipeline.get_specialist(species_label)
            result = specialist.identify(image, top_k=5, threshold=threshold)
        except FileNotFoundError as error:      # no specialist trained for this species
            return jsonify({"error": str(error),
                            "species": species_info(species_label, species_confidence)}), 422
        except Exception as error:
            return jsonify({"error": f"Model B failed: {error}",
                            "species": species_info(species_label, species_confidence)}), 500

    return jsonify({
        "species": species_info(species_label, species_confidence),
        "individual": {
            "individual": "unknown" if result["is_new_individual"] else result["individual"],
            "is_new": result["is_new_individual"],
            "similarity": round(result["match_score"], 4),
            "threshold": round(result["threshold"], 4),
            "closest_known_individual": result["closest_known_individual"],
            "top_matches": result["top_matches"],
        },
    })


@app.route("/api/species", methods=["GET"])
def list_species():
    return jsonify(list_available_species())


@app.route("/api/animals", methods=["GET"])
def list_animals():
    animals = [{"id": f"{species}:{name}", "species": species, "individual": name}
               for species, specialist in pipeline.specialists.items()
               for name in specialist.known_individuals()]
    return jsonify({"animals": animals})


@app.route("/api/animals/<animal_id>", methods=["DELETE"])
def remove_animal(animal_id):
    return jsonify({"error": "individuals come from static checkpoints; cannot remove"}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)