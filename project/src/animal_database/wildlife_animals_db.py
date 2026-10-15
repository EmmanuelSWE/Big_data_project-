'''
ThIS SCRIPT IS FOR CREATING THE DATABASE AND THE TABLES CONTAINING THE METADATA
USEFUL FOR THE MODELS A AND B
'''

import sqlite3
from wildlife_config import BASE_DIRECTORY, DATA_DIRECTORY, DATABASE_PATH, SUB_DATASETS
 
# Name of Model A's table (all species together)
ALL_SPECIES_TABLE = "Animal_All_Species"
 
 
def get_species_table_name(species_name):
    # Model B's table name for one species, e.g. species_dog
    return f"species_{species_name}"
 
 
def create_tables(connect_db):
    """
    Creates every table if it does not exist yet.
    This is the only place where the schema is defined.
 
    Column meaning:
    - image_id: unique id of the image. The same image has the SAME image_id in
                Animal_All_Species and in its species table, so the tables can be joined.
    - path:     full path to the image file on disk
    - species:  species label (Model A's target)
    - identity: individual animal label (Model B's target)
    """
    cursor = connect_db.cursor()
 
    # Model A's training data
    print("Structuring the Animal_All_Species schema")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {ALL_SPECIES_TABLE} (
            image_id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            species TEXT NOT NULL
        );
    """)
 
    # Model B's training data, one table per species
    for species_name in SUB_DATASETS.keys():
        table_name = get_species_table_name(species_name)
        print(f"Structuring the {table_name} schema")
 
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                image_id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                identity TEXT NOT NULL
            );
        """)
 
    connect_db.commit()  # Writing the changes permanently to the database file
    total_tables = 1 + len(SUB_DATASETS)
    print(f"All {total_tables} database tables have been successfully initialised")
 
 
def verify_tables(connect_db):
    # Prints every table in the database with its number of rows
    print("\nVerifying the database:")
    cursor = connect_db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
 
    # Fetching all existing tables
    tables = cursor.fetchall()
 
    for t in tables:
        # Checking the row count for each table
        cursor.execute(f"SELECT COUNT(*) FROM {t[0]}")
        row_count = cursor.fetchone()[0]
        print(f"Table Found: {t[0]: <20} | Current Rows: {row_count}")
 
 
def main():
    # If the directories do not exist, they will be created
    BASE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print(f"Environment directories created successfully at: {BASE_DIRECTORY}")
 
    # Connecting to the SQLite database, if the file does not exist then it
    # will be created as an empty file
    print(f"Connecting to the database file at {DATABASE_PATH}")
    connect_db = sqlite3.connect(str(DATABASE_PATH))
 
    create_tables(connect_db)
    verify_tables(connect_db)
 
    connect_db.close()
 
 
# Only runs when this file is executed directly, not when it is imported
if __name__ == "__main__":
    main()