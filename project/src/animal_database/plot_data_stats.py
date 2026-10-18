import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# The CSV written by database_statistics.py
STATISTICS_CSV_PATH = Path("data_environment") / "database_statistics.csv"
PLOTS_DIRECTORY = Path("data_environment") / "plots"
SPLIT_STATISTICS_CSV_PATH = Path("data_environment") / "split_statistics.csv"

if not STATISTICS_CSV_PATH.exists():
    print(f"Statistics file not found at: {STATISTICS_CSV_PATH}")
    print("Run database_statistics.py first.")
    raise SystemExit

PLOTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

stats_df = pd.read_csv(STATISTICS_CSV_PATH)

# Keeping every row except the "ALL" summary row, so only the species are plotted
species_df = stats_df[stats_df['Species'] != 'ALL']

# The species names as a plain list, used for the legends and the bar chart
species_names = species_df['Species'].tolist()

# One color per species, used in every chart so a species always has the same color
color_options = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
species_colors = []
for position in range(len(species_names)):
    species_colors.append(color_options[position])

# Pie chart: number of images per species
plt.figure(figsize=(6, 6))
wedges, texts = plt.pie(species_df['images'], colors=species_colors)
plt.legend(wedges, species_names, title="Species")
plt.title("Images per species")
plt.savefig(PLOTS_DIRECTORY / "1_images_per_species_pie.png", dpi=200, bbox_inches="tight")

# Bar chart: number of individuals per species
plt.figure(figsize=(6, 4))
plt.bar(species_names, species_df['individuals'], color=species_colors)
plt.xlabel("Species")
plt.ylabel("Number of individuals")
plt.title("Individuals per species")
plt.savefig(PLOTS_DIRECTORY / "2_individuals_per_species_bar.png", dpi=200, bbox_inches="tight")

# Bar chart: species proportion
plt.figure(figsize=(6, 4))
plt.bar(species_names, species_df['species_proportion'], color=species_colors)
plt.xlabel("Species")
plt.ylabel("Species proportion")
plt.title("Species proportion")
plt.savefig(PLOTS_DIRECTORY / "3_species_proportion_bar.png", dpi=200, bbox_inches="tight")

# Bar chart: balance ratio
plt.figure(figsize=(6, 4))
plt.bar(species_names, species_df['balance_ratio_per_species'], color=species_colors)
plt.xlabel("Species")
plt.ylabel("Balance ratio")
plt.title("Balance ratio")
plt.savefig(PLOTS_DIRECTORY / "4_balance_ratio_bar.png", dpi=200, bbox_inches="tight")

# Bar chart: average images per individual
plt.figure(figsize=(6, 4))
plt.bar(species_names, species_df['images_per_individuals'], color=species_colors)
plt.xlabel("Species")
plt.ylabel("Images per individual")
plt.title("Images per individual")
plt.savefig(PLOTS_DIRECTORY / "5_images_per_individual_bar.png", dpi=200, bbox_inches="tight")

# Grouped bar chart: images in the train, val and test splits, one bar per species
if SPLIT_STATISTICS_CSV_PATH.exists():
    split_stats_df = pd.read_csv(SPLIT_STATISTICS_CSV_PATH)
    split_species_df = split_stats_df[split_stats_df['Species'] != 'ALL']

    split_names = ["train", "val", "test"]
    split_columns = ["train_images", "val_images", "test_images"]

    number_of_species = len(species_names)
    bar_width = 0.8 / number_of_species  # The 4 bars of a split share a width of 0.8

    plt.figure(figsize=(8, 5))

    for species_position in range(number_of_species):
        species_name = species_names[species_position]

        # The row of this species in the split CSV
        species_row = split_species_df[split_species_df['Species'] == species_name]

        # Shifting this species' bar left or right inside each split group
        offset = (species_position - (number_of_species - 1) / 2) * bar_width

        x_positions = []
        heights = []
        for split_position in range(len(split_names)):
            x_positions.append(split_position + offset)
            # .iloc[0] takes the first (and only) row of species_row
            heights.append(int(species_row[split_columns[split_position]].iloc[0]))

        plt.bar(x_positions, heights, width=bar_width,
                color=species_colors[species_position], label=species_name)

    plt.xticks([0, 1, 2], split_names)  # One label in the middle of each group
    plt.xlabel("Split")
    plt.ylabel("Number of images")
    plt.title("Images per split")
    plt.legend(title="Species")
    plt.savefig(PLOTS_DIRECTORY / "6_images_per_split_bar.png", dpi=200, bbox_inches="tight")
else:
    print(f"Split statistics not found at: {SPLIT_STATISTICS_CSV_PATH}")
    print("Run split_data.py to create it, then run this script again for plot 6.")

print(f"Plots saved to: {PLOTS_DIRECTORY}")
plt.show()