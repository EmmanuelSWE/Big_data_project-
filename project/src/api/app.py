"""
Flask web app for the MAHER inference pipeline: upload one image, get back the
species (Model A) and the individual (routed to the matching Model B
specialist) - the same identify_image_auto() flow the Tkinter app uses.

Run from project/src/ (infer.py's checkpoint paths are relative to that):
    python api/app.py
"""
import os
import sys
import tempfile

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infer import find_available_species, identify_image_auto  # noqa: E402

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.after_request
def disable_caching(response):
    # Cloudflare (sitting in front of this app) defaults to caching static
    # assets like .js/.css at its edge for hours regardless of how often this
    # app changes. Explicitly telling it not to avoids serving stale frontend
    # code after a deploy.
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/api/species", methods=["GET"])
def list_species():
    return jsonify(find_available_species())


@app.route("/api/identify", methods=["POST"])
def identify():
    if "image" not in request.files:
        return jsonify({"error": "no image file provided (expected form field 'image')"}), 400

    threshold = request.form.get("threshold", default=None, type=float)
    species_threshold = request.form.get("species_threshold", default=None, type=float)

    uploaded = request.files["image"]
    suffix = os.path.splitext(uploaded.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        uploaded.save(temp_file.name)
        temp_path = temp_file.name

    try:
        result = identify_image_auto(
            temp_path, top_k=5, threshold=threshold, species_threshold=species_threshold
        )
    except Exception as error:
        return jsonify({"error": str(error)}), 500
    finally:
        os.remove(temp_path)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
