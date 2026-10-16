# ViT-Tiny with an ArcFace loss, trained once per species (hyena, leopard, dog, macaque)

big_data_project/
├── data/                      <- dataset
│   ├── metadata.csv           <- metadata_file
│   └── images/
│       ├── HyenaID2022/
│       ├── LeopardID2022/
│       ├── DogFaceNet/
│       └── PrimFace/
├── model_b/
│   ├── __init__.py          (empty file)
│   ├── vit_arcface.py       # ViT-Tiny embedder + ArcFace loss
│   ├── dataset.py           # reads metadata.csv, builds per-species tables + Dataset
│   ├── data_loading.py      # image transforms + DataLoaders
│   └── training.py          # training loop, evaluation, saving
├── train_all.py             # runs all 4 species
└── models/                  # created automatically