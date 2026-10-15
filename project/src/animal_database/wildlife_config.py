'''
SELF EXPLAINATORY :>
'''

from pathlib import Path
from wildlife_datasets import datasets
 
# Isolated environment folders
BASE_DIRECTORY = Path("data_environment")
DATA_DIRECTORY = BASE_DIRECTORY / "wildlife_data"
DATABASE_PATH = BASE_DIRECTORY / "wildlife_animals_database.db"
 
# The single source of truth for which species are used.
# Dictionary with species as the key and the wildlife_datasets class as the value.
# The database script and the extraction script both read this dictionary,
# so adding or removing a species here updates everything.
SUB_DATASETS = {
    #'tiger': datasets.ATRW,
    #'giraffe': datasets.GiraffeZebraID,
    #'panda': datasets.IPanda50,
    'hyena': datasets.SpottedHyenaID109,
    'leopard': datasets.LeopardID102,
    #'cat': datasets.CatIndividualImages,
    #'elephant': datasets.ELPephants,
    #'nyala': datasets.NyalaData,
    'macaque': datasets.MacaqueFaces,
    'dog': datasets.DogFaceNet,
    #'cow': datasets.CowDataset,
    #'chicken': datasets.Chicks4FreeID
}

# An individual needs at least this many images to be used.
# Individuals with fewer images are left out, because they cannot be
# split properly into train / validation / test.
MIN_IMAGES_PER_INDIVIDUAL = 6

# The maximum number of images sampled per species
# a species with fewer images than this keeps all of its images
IMAGES_PER_SPECIES = 1000
 
# Fixed seed so the same sample is drawn every time
RANDOM_SEED = 42
