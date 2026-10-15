'''
THIS SCRIPT IS USED FOR VIEWING THE CONTENTS OF THE DATABASE, NOTHING MUCH ABOUT IT.
'''

import sqlite3
import pandas as pd
from pathlib import Path

# Same database path as wildlife_config.py (kept here so this script runs fast)
DATABASE_PATH = Path("data_environment") / "wildlife_animals_database.db"
ROWS_TO_SHOW = 5  # How many example rows to print per table

if not DATABASE_PATH.exists():
    print(f"Database not found at: {DATABASE_PATH}")
    raise SystemExit

connect_database = sqlite3.connect(str(DATABASE_PATH))
cursor = connect_database.cursor()

# Getting the names of all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
table_rows = cursor.fetchall()

# >>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<
# Overview of every table
#>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>><<<<<<<<<,
for table_row in table_rows:
    table_name = table_row[0]

    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cursor.fetchone()[0]

    print("=" * 70)
    print(f"TABLE: {table_name}  |  rows: {row_count}")
    print("=" * 70)

    # PRAGMA table_info gives one row per column: (id, name, type, notnull, default, pk)
    cursor.execute(f"PRAGMA table_info({table_name})")
    column_rows = cursor.fetchall()
    print("Columns:")
    for column_row in column_rows:
        print(f"  {column_row[1]} ({column_row[2]})")

    print(f"\nFirst {ROWS_TO_SHOW} rows:")
    preview_df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {ROWS_TO_SHOW}", connect_database)
    print(preview_df.to_string(index=False))
    print()

# >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Model A summary: number of images per species
# >>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print("=" * 70)
print("MODEL A: images per species")
print("=" * 70)
species_counts_df = pd.read_sql_query(
    "SELECT species, COUNT(*) AS images FROM Animal_All_Species GROUP BY species",
    connect_database)
print(species_counts_df.to_string(index=False))
print()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Model B summary: individuals and images per individual, per species
# >>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print("=" * 70)
print("MODEL B: individuals and images per individual")
print("=" * 70)
for table_row in table_rows:
    table_name = table_row[0]

    # Only the species_* tables hold individual identities
    if not table_name.startswith("species_"):
        continue

    identity_df = pd.read_sql_query(f"SELECT identity FROM {table_name}", connect_database)

    if len(identity_df) == 0:
        print(f"{table_name}: empty")
        continue

    # Number of images each individual has
    images_per_identity = identity_df['identity'].value_counts()

    print(f"{table_name}: {len(identity_df)} images, "
          f"{len(images_per_identity)} individuals | "
          f"images per individual -> min {images_per_identity.min()}, "
          f"mean {images_per_identity.mean():.1f}, "
          f"max {images_per_identity.max()}")

connect_database.close()