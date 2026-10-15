'''
THIS SCRIPT IS FOR EXTRACTING THE ANIMAL DATASETS FROM THE WILDLIFE-REID-10K
DATASET THROUGH ITS API AND THEN SAMPLES FROM THOSE EXTRACTED DATASETS.
THE EXTRACTED DATASETS ARE SPECIES OF THE INDIVIDUALS.
THE SAMPLES SUBSETS ARE THEN POPULATED INTO THE RELEVANT TABLES. 
'''

import os
import sys
import sqlite3
import pandas as pd
import random
from wildlife_config import (BASE_DIRECTORY, DATA_DIRECTORY, DATABASE_PATH,
                             SUB_DATASETS, IMAGES_PER_SPECIES, RANDOM_SEED,
                             MIN_IMAGES_PER_INDIVIDUAL)
from wildlife_animals_db import (ALL_SPECIES_TABLE, get_species_table_name,
                                 create_tables, verify_tables)
 
extracted_metadata_frames = []  # One dataframe per species, collected here
failed_species = [] # Species whose download or loading failed
 
# Creating the directories if they do not exist
BASE_DIRECTORY.mkdir(parents=True, exist_ok=True)
DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
print(f"Local environments initialised at: {BASE_DIRECTORY}")
 
# ---------------------------------------------------------------------------
# Downloading each species and collecting its metadata
# ---------------------------------------------------------------------------
print("Starting the targeted sub dataset downloads via their official APIs...")
 
for species_name, dataset_class in SUB_DATASETS.items():
 
    species_working_directory = DATA_DIRECTORY / species_name
    species_working_directory.mkdir(parents=True, exist_ok=True)
 
    try:
        print(f"Downloading and processing: {species_name}...")
 
        # Downloading the files strictly in their assigned folder path
        if species_name == 'chicken':
            dataset_class.get_data(str(species_working_directory), hf_option='chicken-re-id-best-visibility')
        else:
            dataset_class.get_data(str(species_working_directory))
 
        target_dataset_loader = dataset_class(str(species_working_directory))
 
        # Isolating the tracking dataframe
        working_metadata_df = target_dataset_loader.df.copy()
        working_metadata_df['species'] = species_name
 
        # Removing images that have no known individual, because Model B cannot learn from an unlabelled image.
        # Each wildlife_datasets class stores its "unlabelled" label in unknown_name.
        unknown_label = getattr(target_dataset_loader, 'unknown_name', 'unknown')
        has_identity = working_metadata_df['identity'].notna()
        is_labelled = working_metadata_df['identity'] != unknown_label
        working_metadata_df = working_metadata_df[has_identity & is_labelled].copy()
 
        # The identity column is stored as TEXT in the database
        working_metadata_df['identity'] = working_metadata_df['identity'].astype(str)
 
        # The paths given by wildlife_datasets are relative to each dataset's own
        # folder, so two species could have the same relative path.
        # Building the full path so every path in the database is unique and usable.
        full_paths = []
        for relative_path in working_metadata_df['path']:
            full_path = os.path.join(str(species_working_directory), str(relative_path))
            full_paths.append(full_path)
        working_metadata_df['path'] = full_paths
 
        print(f"  {species_name}: {len(working_metadata_df)} labelled images, "
              f"{working_metadata_df['identity'].nunique()} individuals")
 
        extracted_metadata_frames.append(working_metadata_df[['path', 'species', 'identity']])
 
    except Exception as e:
        print(f"Failed to pull dataset partition for {species_name}: {e}")
        failed_species.append(species_name)
 
# ---------------------------------------------------------------------------
# Combining the species into one dataframe
# ---------------------------------------------------------------------------
if len(extracted_metadata_frames) > 0:
    combined_dataframe = pd.concat(extracted_metadata_frames, ignore_index=True)
    print(f"Extracted sub dataset metadata is loaded: {len(combined_dataframe)} total rows.")
else:
    print("Critical Failure: No data frames were successfully processed.")
    sys.exit(1)
 
if len(failed_species) > 0:
    print(f"WARNING: these species failed and will have empty tables: {failed_species}")
 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Selecting whole individuals for each species, up to IMAGES_PER_SPECIES.
#
# for each species, the following code block will do the following:
#  1. Count the images of every individual
#  2. Keep only individuals with at least MIN_IMAGES_PER_INDIVIDUAL images
#  3. Shuffle the kept individuals using a fixed seed so that the results reproduceable
#  4. Go through the individuals one by one:
#        if adding ALL of the individual's images stays within the budget, take it
#        otherwise skip it and try the next individual
#   5. Stops when the budget is full or the individuals run out
# An individual is never cut in half, so none ends up with fewer images than the minimum.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,
sampled_frames = []
downloaded_species = combined_dataframe['species'].unique() # should contain 4 species names
random_generator = random.Random(RANDOM_SEED)
species_below_threshold = []  # Species where no individual reaches the minimum
 
for species_name in downloaded_species:
    species_rows = combined_dataframe[combined_dataframe['species'] == species_name] # stores all the individuals of the current species
 
    # number of images that each individual has
    images_per_identity = species_rows['identity'].value_counts()
 
    # keeping only the individuals that have enough images
    eligible_identities = [] # contains the individuals of the current species that meet the set threshold
    for identity, image_count in images_per_identity.items():
        if image_count >= MIN_IMAGES_PER_INDIVIDUAL:
            eligible_identities.append(identity)
 
    # storing the number of individuals that met the threshold 
    qualifying_count = len(eligible_identities)
 
    # If no individual meets the threshold, the species is NOT skipped.
    # All of its individuals are kept instead, and the species is reported below
    # so the preprocessing for it can be decided separately.
    if qualifying_count == 0:
        species_below_threshold.append(species_name)
        for identity in images_per_identity.index:
            eligible_identities.append(identity)
 
    # sorting first so the shuffle gives the same order every run
    eligible_identities.sort()
    random_generator.shuffle(eligible_identities)
 
    # taking whole individuals while they fit in the budget
    selected_identities = []
    images_selected = 0
    for identity in eligible_identities:
        image_count = int(images_per_identity[identity])

        # Making sure that the eligible individuals selected don't end up exceeding 
        # the images each species should have
        if images_selected + image_count <= IMAGES_PER_SPECIES:
            selected_identities.append(identity)
            images_selected = images_selected + image_count
 
        # stopping once the budget is exactly full
        if images_selected == IMAGES_PER_SPECIES:
            break
 
    print(f"  {species_name}: {qualifying_count} of {len(images_per_identity)} individuals "
          f"have >= {MIN_IMAGES_PER_INDIVIDUAL} images; selected {len(selected_identities)} "
          f"individuals ({images_selected} images)")
 
    # Rare edge case: every individual has more images than the whole budget
    if len(selected_identities) == 0:
        print(f"WARNING: no individual of {species_name} fits inside the budget of "
              f"{IMAGES_PER_SPECIES} images, so nothing was selected for it")
        continue
 
    # Keeping only the rows that belong to the selected individuals
    selected_rows = species_rows[species_rows['identity'].isin(selected_identities)]
    sampled_frames.append(selected_rows)
 
# Telling which species could not meet the threshold
if len(species_below_threshold) > 0:
    print(f"\nWARNING: no individual reached {MIN_IMAGES_PER_INDIVIDUAL} images for these species:")
    for species_name in species_below_threshold:
        print(f"  - {species_name}")
    print("They were still added to the database without the threshold, "
          "so their preprocessing can be decided separately.\n")
 
if len(sampled_frames) == 0:
    print("Critical Failure: no species could be selected.")
    sys.exit(1)
 
samples_df = pd.concat(sampled_frames, ignore_index=True)
sampled_species = samples_df['species'].unique() # The species that actually made it into the sample

# Giving every image one unique id. The same id is used in the species table
# so the two tables can be joined on image_id.
samples_df['image_id'] = samples_df.index
 
#>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<,
# Populating the database tables
#>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print(f"Connecting to the SQLite database at: {DATABASE_PATH}")
connect_database = sqlite3.connect(str(DATABASE_PATH))
 
# Creating the tables from the schema defined in wildlife_animals_db.py
create_tables(connect_database)
 
# Emptying the tables first so running this script twice does not duplicate rows
cursor = connect_database.cursor()
cursor.execute(f"DELETE FROM {ALL_SPECIES_TABLE}")
for species_name in SUB_DATASETS.keys():
    cursor.execute(f"DELETE FROM {get_species_table_name(species_name)}")
connect_database.commit()
 
# Populating Model A's table
print("Populating Model A's table")
model_a_table_data = samples_df[['image_id', 'path', 'species']]
model_a_table_data.to_sql(name=ALL_SPECIES_TABLE,
                          con=connect_database,
                          if_exists='append',  # append keeps the schema defined
                          index=False)
 
# Populating Model B's tables
for species_name in sampled_species:
    table_name = get_species_table_name(species_name)
    print(f"Populating table {table_name}...")
 
    species_subset = samples_df[samples_df['species'] == species_name]
    species_table_data = species_subset[['image_id', 'path', 'identity']]
 
    species_table_data.to_sql(name=table_name,
                              con=connect_database,
                              if_exists='append',
                              index=False)
 
verify_tables(connect_database)
connect_database.close()
 