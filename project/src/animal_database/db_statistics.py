'''
THIS SCRIPT EXTRACTS THE STATISTICS OF THE COLLECTED ANIMAL DATA.
THE CREATED CSV IS TO BE USED TO VISUALISE THE STATS OF THE DATA
AND MAKE A STORY OUT OF IT.
'''
import sqlite3
import pandas as pd
from pathlib import Path

# Same database path as wildlife_config.py (kept here so this script runs fast)
DATABASE_PATH = Path("data_environment") / "wildlife_animals_database.db"
OUTPUT_CSV_PATH = Path("data_environment") / "database_statistics.csv"

# Column order of the output CSV
CSV_COLUMNS = ["Species", "images", "individuals", "species_count",
               "images_per_individuals", "min_image_per_species", 
                "min_image_per_individual", "max_image_per_species",
                "max_image_per_individual", "balance_ratio_per_species",
                "species_proportion"]

if not DATABASE_PATH.exists():
    print(f"Database not found at: {DATABASE_PATH}")
    raise SystemExit

connect_database = sqlite3.connect(str(DATABASE_PATH))

# >>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,
# Species level statistics from Model A's table
# Only Animal_All_Species and the species_* tables are read
# >>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
species_counts_df = pd.read_sql_query(
    "SELECT species, COUNT(*) AS images FROM Animal_All_Species GROUP BY species ORDER BY species",
    connect_database)

species_count = len(species_counts_df) # should be 4 species since that is how many species we are dealing with
min_image_per_species = int(species_counts_df['images'].min()) # minimum number of images that a species has
max_image_per_species = int(species_counts_df['images'].max()) # maximum number of images that a species has

# Class balamce across species. 1.0 means perfectly balanced.
balance_ratio_across_species = min_image_per_species / max_image_per_species

#>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Individual level statistics from each species_* table of the database
# >>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
statistics_rows = [] # rows of the csv file
all_individual_image_counts = []  # One list of per-individual image counts per species
total_images = 0

for row_number in range(len(species_counts_df)):
    species_name = species_counts_df.loc[row_number, 'species']
    images = int(species_counts_df.loc[row_number, 'images'])
    table_name = f"species_{species_name}"

    identity_df = pd.read_sql_query(f"SELECT identity FROM {table_name}", connect_database) # identity represents the individual animal

    if len(identity_df) == 0:
        print(f"WARNING: {table_name} is empty, skipping this species") # If this happens, then something is wrong, because no table should be empty 
        continue

    # Number of images that each individual has
    images_per_identity = identity_df['identity'].value_counts()
    all_individual_image_counts.append(images_per_identity)

    individuals = len(images_per_identity)
    min_image_per_individual = int(images_per_identity.min())
    max_image_per_individual = int(images_per_identity.max())

    total_images_all_species = int(species_counts_df['images'].sum())

    total_images = total_images + images

    statistics_rows.append({
        "Species": species_name,
        "images": images,
        "individuals": individuals,
        "species_count": species_count,
        "images_per_individuals": round(images / individuals, 2),
        "min_image_per_species": min_image_per_species,
        "min_image_per_individual": min_image_per_individual,
        "max_image_per_species": max_image_per_species,
        "max_image_per_individual": max_image_per_individual,
        "balance_ratio_per_species": round(min_image_per_individual / max_image_per_individual, 3),
        "species_proportion": round(images / total_images_all_species, 3),
    })

# >>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<,,
# summarises every species together.
# Individuals are counted per species, because two species may reuse the same
# identity name for different animals.
# >>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<
if len(all_individual_image_counts) > 0:
    all_counts = pd.concat(all_individual_image_counts, ignore_index=True)
    all_individuals = len(all_counts)
    all_min_individual = int(all_counts.min())
    all_max_individual = int(all_counts.max())

    statistics_rows.append({
        "Species": "ALL",
        "images": total_images,
        "individuals": all_individuals,
        "species_count": species_count,
        "images_per_individuals": round(total_images / all_individuals, 2),
        "min_image_per_species": min_image_per_species,
        "min_image_per_individual": all_min_individual,
        "max_image_per_species": max_image_per_species,
        "max_image_per_individual": all_max_individual,
        "balance_ratio_per_species": round(balance_ratio_across_species, 3),
        "species_proportion": round(total_images / total_images_all_species, 3),
    })

connect_database.close()

# >>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<
# Writing and outputing the CSV
# >>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<
statistics_df = pd.DataFrame(statistics_rows, columns=CSV_COLUMNS)
statistics_df.to_csv(str(OUTPUT_CSV_PATH), index=False)

print(statistics_df.to_string(index=False))
print(f"\nStatistics written to: {OUTPUT_CSV_PATH}")