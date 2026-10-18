# MAHER — Multi-Agent Hierarchical Expert Re-identification

A two-stage system that identifies wild animals from a photo.

1. **Model A (the generalist)** predicts the species: dog, hyena, leopard or macaque.
2. **Model B (the specialist)** — one model per species — decides which known individual it is, or flags it as a **new individual**.

**Live demo:** <https://maher.lwazi.gay>

```
photo ──► Model A (ResNet50) ──► species ──► Model B specialist (ViT-Tiny + ArcFace) ──► known individual / new individual
```

---

## Contents

- [Repository layout](#repository-layout)
- [Libraries used](#libraries-used)
- [Installation](#installation)
- [Where to run commands from](#where-to-run-commands-from)
- [1. Build the database](#1-build-the-database)
- [2. Train and test Model A](#2-train-and-test-model-a)
- [3. Train and test Model B](#3-train-and-test-model-b)
- [4. Run the full pipeline on one photo](#4-run-the-full-pipeline-on-one-photo)
- [5. Run the web app locally](#5-run-the-web-app-locally)
- [6. Deployed version](#6-deployed-version)
- [Troubleshooting](#troubleshooting)

---

## Repository layout

```
Docs/                         Design documents (ERD, pipeline diagram, plan)
deploy/                       Docker Compose file used for the live deployment
presentation/                 Project slides
project/
├── frontend/                 Web page (index.html, app.js, style.css) served by Flask
└── src/
    ├── animal_database/      Download the datasets, build the SQLite database, split it, statistics
    ├── modelA/               Species classifier: model, training, testing, single-image prediction
    │   ├── checkpoints/      Trained Model A weights (modelA_best.pth)
    │   └── results/          Test metrics, confusion matrix, predictions, training history
    ├── model_b/              Individual re-identification: ViT + ArcFace, training, pipeline
    │   ├── checkpoints/      Trained specialists (model_b_dog.pt, model_b_hyena.pt, ...)
    │   └── results/          Model B result CSVs
    └── api/                  Flask server (app.py) and requirements.txt
```

---

## Libraries used

| Library | Used for | Where |
|---|---|---|
| **PyTorch** (`torch`) | Building, training and running both models | modelA, model_b, api |
| **torchvision** | ResNet50 with ImageNet weights, image transforms | modelA, model_b |
| **timm** | Pretrained ViT-Tiny backbone (`vit_tiny_patch16_224`) | model_b |
| **wildlife-datasets** | Downloading SpottedHyenaID109, LeopardID102, MacaqueFaces, DogFaceNet | animal_database |
| **pandas** | Reading and writing database tables and CSV results | everywhere |
| **NumPy** | Similarity scores, threshold search | model_b |
| **scikit-learn** | Metrics (accuracy, precision, recall, F1, confusion matrix) and identity splits | modelA, model_b |
| **Pillow** (`PIL`) | Opening images | modelA, model_b, api |
| **Flask** | Web server and REST API | api |
| **matplotlib** | Dataset statistics plots, Model B training curves | animal_database, model_b |
| **seaborn** | Model B confusion-matrix heatmaps | model_b |
| **sqlite3** | The image and label database (built into Python) | animal_database, modelA, model_b |
| **tkinter** | File picker in `predict_model_A.py` (built into most Python installs) | modelA |

> `project/src/api/requirements.txt` also lists `tensorflow`, but no code in the project imports it, so you don't need to install it.

---

## Installation

You need **Python 3.10 or newer**. A CUDA GPU makes training much faster but isn't required; the code falls back to the CPU automatically.

```bash
git clone https://github.com/EmmanuelSWE/Big_data_project-.git
cd Big_data_project-

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install torch torchvision timm wildlife-datasets pandas numpy scikit-learn pillow flask matplotlib seaborn
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch first, using the command from <https://pytorch.org/get-started/locally/>, then run the line above.

---

## Where to run commands from

**This matters.** All the scripts find the database through the relative path `data_environment/wildlife_animals_database.db`, so they must run **from inside `project/src/animal_database`**. The model scripts also import packages from `project/src` (`animal_database`, `modelA`, `model_b`), so that folder must be on `PYTHONPATH`.

Set this up once per terminal:

```bash
# macOS / Linux
cd project/src/animal_database
export PYTHONPATH="$(pwd)/.."
```

```bat
:: Windows (Command Prompt)
cd project\src\animal_database
set "PYTHONPATH=%CD%\.."
```

```powershell
# Windows (PowerShell)
cd project\src\animal_database
$env:PYTHONPATH = (Resolve-Path ..).Path
```

Unless a step says otherwise, every command below assumes you are in `project/src/animal_database` with `PYTHONPATH` set.

---

## 1. Build the database

The database is a SQLite file holding the path and labels of every image. It is **not committed** to the repository (it's in `.gitignore`), so every machine has to build it, including if you only want to run the web app.

### 1.1 Download the images and fill the tables

```bash
python wildlife_animals_data_extraction.py
```

This step:

- downloads the four datasets with `wildlife-datasets` into `data_environment/wildlife_data/<species>/`
- removes unlabelled images and individuals with fewer than **6** images
- samples whole individuals, up to **1,000 images per species** (fixed seed 42)
- creates `data_environment/wildlife_animals_database.db` with these tables:

| Table | Columns | Used by |
|---|---|---|
| `Animal_All_Species` | `image_id`, `path`, `species` | Model A |
| `species_dog`, `species_hyena`, `species_leopard`, `species_macaque` | `image_id`, `path`, `identity` | Model B |

`image_id` is the same for a given image in both tables, so they can be joined.

You can change the species, the minimum number of images per individual, the per-species budget and the seed in **`wildlife_config.py`**.

> You can run the script again safely: it empties the tables before filling them. If you only need the empty tables, run `python wildlife_animals_db.py`.

### 1.2 Add the train / validation / test split

```bash
python split_data.py
```

This adds a `split` column (`train` / `val` / `test`, about 70 / 15 / 15 of each individual's images) to every table, and writes `data_environment/split_statistics.csv`. **Model A needs this step.**

### 1.3 Optional: inspect the data

```bash
python db_view.py           # print each table's columns, first rows and counts
python db_statistics.py     # write data_environment/database_statistics.csv
python plot_data_stats.py   # save charts to data_environment/plots/
```

---

## 2. Train and test Model A

Model A is an ImageNet-pretrained **ResNet50**. Its backbone is frozen and it gets a new 4-class output layer.

| Setting | Value |
|---|---|
| Epochs | 5 |
| Optimizer | Adam, learning rate 0.001 (output layer only) |
| Loss | Cross-entropy |
| Batch size | 16 |
| Input | 224×224, standard ResNet50 transforms |

### Train

```bash
python -m modelA.train_model_A
```

Training uses the `train` and `val` rows from the database. Each time the validation loss improves, the script saves the best output layer to `project/src/modelA/checkpoints/modelA_best.pth`. When it finishes, it writes `results/modelA_training_history.csv` (loss, accuracy, precision, recall and F1 for each epoch).

### Test

```bash
python -m modelA.test_model_A
```

This loads the checkpoint, runs it on the `test` split and writes to `project/src/modelA/results/`:

| File | Contents |
|---|---|
| `modelA_test_summary.csv` | Checkpoint epoch, number of test images, accuracy |
| `modelA_test_metrics.csv` | Precision, recall, F1 and support for each species, plus macro and weighted averages |
| `modelA_confusion_matix.csv` | 4×4 confusion matrix (rows = actual, columns = predicted) |
| `modelA_test_predictions.csv` | Actual vs predicted species for every test image |

**Current results:** 99.78% test accuracy (455 of 456 images), macro F1 0.997. The one mistake was a hyena predicted as a leopard.

### Predict one image

```bash
python -m modelA.predict_model_A
```

A file picker opens. Choose an image and the script prints the predicted species and its confidence.

---

## 3. Train and test Model B

Model B is **four separate specialists**, one per species. Each is a **ViT-Tiny** (from `timm`) trained with **ArcFace loss** to turn a photo into a 256-number, unit-length embedding. Photos of the same animal end up close together.

| Setting | Value |
|---|---|
| Backbone | `vit_tiny_patch16_224` (pretrained) |
| Embedding size | 256 |
| Epochs | 30 |
| Batch size | 32 |
| Optimizer | AdamW, learning rate 1e-4, cosine schedule |
| Loss | ArcFace (scale 30, margin 0.5) |

### How the data is split for Model B

Model B splits **by individual**, not by image, so it can practise recognising animals it has never seen (`model_b/database_data.py`):

| Group | Contents | Used for |
|---|---|---|
| `train` | 70% of individuals | Learning |
| `validation_known` / `test_known` | Held-back images of those same individuals | Rank-1 accuracy |
| `validation_unknown` | 15% of individuals never shown in training | Choosing the "new individual" threshold |
| `test_unknown` | The other 15% of unseen individuals | Final open-set test |

Each species needs at least 4 individuals for this split.

### Train

```bash
# all four species, one after another
python -m model_b.train_model_B

# or just one species
python -m model_b.train_model_B --species hyena
```

After every epoch the script measures **validation rank-1 accuracy**: it compares each held-back image with the training images and checks whether the closest match is the right animal. It saves the best model to `project/src/model_b/checkpoints/model_b_<species>.pt`. If one species fails, the others still train.

### Test and evaluate

**Rank-1 accuracy** is printed for every epoch during training.

**Known vs new threshold:** when a specialist loads (in the pipeline or the web app), it:

1. embeds all the training images as its gallery
2. compares the `validation_known` and `validation_unknown` images against that gallery
3. picks the cosine-similarity threshold that best separates correctly matched known animals from unseen ones

If a new photo's best match is below that threshold, it is reported as a **new individual**.

**Plots:** `evaluate_model_b.py` turns a results CSV with the columns `species, epoch, train_loss, val_loss, train_accuracy, val_accuracy` into loss and accuracy curves. Run it from the folder that holds `train_results.csv`:

```bash
cd ../model_b/results
python ../evaluate_model_b.py      # writes plots/loss_curves.png and plots/accuracy_curves.png
```

**Reported validation rank-1 accuracy:** macaque about 98.9%, dog about 90%, leopard about 89%, hyena 76–80%. The macaque score should be read with caution because there are only 5 macaque individuals.

---

## 4. Run the full pipeline on one photo

This runs Model A, then the matching Model B specialist, from the command line:

```bash
python -m model_b.maher_pipeline path/to/photo.jpg

# skip Model A and force a species
python -m model_b.maher_pipeline path/to/photo.jpg --species leopard
```

It prints JSON with the species and its confidence, `is_new_individual`, the closest known individual, the match score, the threshold and the top 5 matches.

---

## 5. Run the web app locally

Trained checkpoints for **both models are already in the repo**, so you don't have to train anything to run the app. You **do** need the database and the downloaded images, because each Model B specialist builds its gallery from them when it starts. So complete [step 1](#1-build-the-database) first.

From the **repository root**:

```bash
python project/src/api/app.py
```

The server loads Model A and all four specialists when it starts, which can take a minute. When the terminal shows `[api] ready`, open:

**<http://127.0.0.1:5000>**

Upload a photo and click **Identify**. The page shows the species with its confidence, then either the individual's name with its similarity score or **new individual**.

`app.py` sets its own working directory and import path, so it doesn't need `PYTHONPATH`.

### API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web page |
| `POST` | `/api/identify` | Form field `image` (file) and optional `threshold` (float). Returns the species and individual result as JSON |
| `GET` | `/api/species` | Species that have a trained specialist |
| `GET` | `/api/animals` | Every known individual, per species |

Example:

```bash
curl -F "image=@photo.jpg" http://127.0.0.1:5000/api/identify
```

---

## 6. Deployed version

The app is live at **<https://maher.lwazi.gay>**.

How it is deployed (`deploy/maher-docker-compose.yaml`):

- the app runs in a **Docker** container (`maher`) that listens on port **5000**
- **Traefik** acts as a reverse proxy in front of it, sending traffic for `maher.<domain>` to the container
- **HTTPS** is automatic: Traefik gets the certificate through the Cloudflare resolver, and HTTP redirects to HTTPS

The compose file builds from a `Dockerfile` in `./repo`. That Dockerfile isn't in this repository.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: train data base path not found` or `Database not found at ...` | You're in the wrong folder, or haven't built the database. `cd project/src/animal_database` and run step 1. |
| `ModuleNotFoundError: No module named 'animal_database'` (or `modelA`, `model_b`) | `PYTHONPATH` isn't set to `project/src`. See [Where to run commands from](#where-to-run-commands-from). |
| `ValueError: some images have no split assigned` | Run `python split_data.py`. |
| A species fails to download | The extraction script carries on with the other species and lists the failed ones. Re-run it, or check the `wildlife-datasets` documentation for that dataset. |
| The web app is slow to start | Normal: every specialist embeds its whole gallery at startup. On a GPU it's much faster. |
| `pre trained model not found` | `modelA/checkpoints/modelA_best.pth` is missing. Train Model A first, or restore it from the repo. |

---

## Authors

T.W. Manhema, J.S. Phala, H.E. Ashimwe, L. Phuma — University of Johannesburg
