'''
This script creates an additional column to the existing database tables.
Animals_All table will have a split column with text values "train", "val", and "test"
Species table will have a split column with text values  "train", "val", and "test"
'''

import sqlite3
import random
import pandas as pd
from pathlib import Path

# Same database path as wildlife_config.py
DATABASE_PATH = Path("data_environment") / "wildlife_animals_database.db"
SPLIT_STATISTICS_CSV_PATH = Path("data_environment") / "split_statistics.csv"

# This is the share of each individual's images that goes to validation and to test.
# The rest of the 70% is used for training.
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

# An individual needs at least this many images to be split three ways.
# Individuals below it go entirely to train, and are reported at the end.
MIN_IMAGES_TO_SPLIT = 3

RANDOM_SEED = 42

if not DATABASE_PATH.exists():
    print(f"Database not found at: {DATABASE_PATH}")
    raise SystemExit

connect_database = sqlite3.connect(str(DATABASE_PATH))
cursor = connect_database.cursor()
random_generator = random.Random(RANDOM_SEED)

def add_split_column(table_name):
    # Adds the split column to a table if it is not there yet
    cursor.execute(f"PRAGMA table_info({table_name})")
    column_rows = cursor.fetchall()

    has_split_column = False
    for column_row in column_rows:
        if column_row[1] == "split":
            has_split_column = True

    if not has_split_column:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN split TEXT")

# Extract species from the Animal_All_Species table
species_df = pd.read_sql_query("SELECT DISTINCT species FROM Animal_All_Species ORDER BY species", connect_database)

# Every image gets exactly one split. image_id is unique across all species,
# so this dictionary will be used for the species tables AND Animal_All_Species.
# Both the species table and individual tables share the same image_id and path so whatever
# split happens in All_Animal_Species, applies to the species_* tables 
split_for_image = {} # key-value: image_id-split
individuals_too_small = 0

# The followin for loop splits each species table, one individual at a time
# so each individual with n images:
#   1. Shuffles its images
#   2. test count = round(n * TEST_FRACTION), ensures that at least 1 image is taken
#   3. validation count = round(n * VALIDATION_FRACTION), ensures that at least 1 image is taken
# This makes sure that every individual appears in train, val and test split.
for row_number in range(len(species_df)):
    species_name = species_df.loc[row_number, 'species']
    table_name = f"species_{species_name}"

    add_split_column(table_name)

    table_df = pd.read_sql_query(f"SELECT image_id, identity FROM {table_name}", connect_database)

    # Grouping the image ids by individual
    images_of_identity = {} # key-value: individual-image_id(array of image_id)
    for index in range(len(table_df)):
        image_id = int(table_df.loc[index, 'image_id'])
        identity = table_df.loc[index, 'identity']

        if identity not in images_of_identity:
            images_of_identity[identity] = [] # empty array of image_id
        images_of_identity[identity].append(image_id)

    # Sorted identities, so the result does not depend on the database row order
    for identity in sorted(images_of_identity.keys()): # sorting by individual
        image_ids = images_of_identity[identity]
        image_ids.sort()
        random_generator.shuffle(image_ids) # mixing and matching the image_ids of the current individual

        number_of_images = len(image_ids)

        # Too few images to split three ways, then everything goes to train
        if number_of_images < MIN_IMAGES_TO_SPLIT:
            individuals_too_small = individuals_too_small + 1
            for image_id in image_ids:
                split_for_image[image_id] = "train"
            continue

        test_count = round(number_of_images * TEST_FRACTION) # number of individual images for the test split
        if test_count < 1: # ensuring that at minimum, each individual gets 1 image into a split
            test_count = 1

        validation_count = round(number_of_images * VALIDATION_FRACTION) # number of individual images for the validation split  
        if validation_count < 1:
            validation_count = 1

        # Going through the shuffled individual's images and giving each one a split
        for position in range(number_of_images):
            image_id = image_ids[position]

            if position < test_count:
                split_for_image[image_id] = "test"
            elif position < test_count + validation_count:
                split_for_image[image_id] = "val"
            else:
                split_for_image[image_id] = "train"

    # Writing the split into this species table
    update_rows = [] # array of tuples (split, image_id) 
    for image_id in table_df['image_id']:
        update_rows.append((split_for_image[int(image_id)], int(image_id))) 
    # Updating the table in bulk
    cursor.executemany(f"UPDATE {table_name} SET split = ? WHERE image_id = ?", update_rows) # split = "train" and image_id = somrnef

# Copying the same split into Model A's table, matched by image_id
add_split_column("Animal_All_Species")

all_species_df = pd.read_sql_query("SELECT image_id FROM Animal_All_Species", connect_database)
update_rows = [] # array of tuple (split, image_id) 
for image_id in all_species_df['image_id']:
    update_rows.append((split_for_image[int(image_id)], int(image_id)))

# Updating the table in bulk
cursor.executemany("UPDATE Animal_All_Species SET split = ? WHERE image_id = ?", update_rows) # split = "train" and image_id = somrned

connect_database.commit()

# Checking the result
print("Images per species and split (Animal_All_Species):")
counts_df = pd.read_sql_query(
    "SELECT species, split, COUNT(*) AS images FROM Animal_All_Species GROUP BY species, split ORDER BY species, split",
    connect_database)
print(counts_df.to_string(index=False))

print("\nTotal images per split:")
totals_df = pd.read_sql_query(
    "SELECT split, COUNT(*) AS images FROM Animal_All_Species GROUP BY split ORDER BY split",
    connect_database)
print(totals_df.to_string(index=False))


# Writing the split statistics to a CSV file for plotting
# One row per species, plus an ALL row with the totals
# Reading the counts from counts_df into a dictionary: image_counts[species][split]
image_counts = {}
for index in range(len(counts_df)):
    species_name = counts_df.loc[index, 'species']
    split_name = counts_df.loc[index, 'split']
    images = int(counts_df.loc[index, 'images'])

    if species_name not in image_counts:
        image_counts[species_name] = {}
    image_counts[species_name][split_name] = images

statistics_rows = []
all_train = 0
all_val = 0
all_test = 0

for row_number in range(len(species_df)):
    species_name = species_df.loc[row_number, 'species']

    # .get(key, 0) gives 0 when a species has no images in that split
    train_images = image_counts[species_name].get("train", 0)
    val_images = image_counts[species_name].get("val", 0)
    test_images = image_counts[species_name].get("test", 0)
    total_images = train_images + val_images + test_images

    all_train = all_train + train_images
    all_val = all_val + val_images
    all_test = all_test + test_images

    statistics_rows.append({
        "Species": species_name,
        "train_images": train_images,
        "val_images": val_images,
        "test_images": test_images,
        "total_images": total_images,
        "train_proportion": round(train_images / total_images, 3),
        "val_proportion": round(val_images / total_images, 3),
        "test_proportion": round(test_images / total_images, 3)
    })

all_total = all_train + all_val + all_test
statistics_rows.append({
    "Species": "ALL",
    "train_images": all_train,
    "val_images": all_val,
    "test_images": all_test,
    "total_images": all_total,
    "train_proportion": round(all_train / all_total, 3),
    "val_proportion": round(all_val / all_total, 3),
    "test_proportion": round(all_test / all_total, 3)
})

statistics_df = pd.DataFrame(statistics_rows)
statistics_df.to_csv(str(SPLIT_STATISTICS_CSV_PATH), index=False)
print(f"\nSplit statistics written to: {SPLIT_STATISTICS_CSV_PATH}")

connect_database.close()