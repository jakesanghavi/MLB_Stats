import pandas as pd
import numpy as np

# Get just the year you care about
year=2025

# Read from your data source
df = pd.read_csv(f'{year}_full_pbp.csv')

# Baseball diameter in feet
# (the units from the API are in feet)
BASEBALL_DIAM = 2.9/12

# Center of strike zone for all players
# in the X direction is the same: 0
center_x = 0

# Standard Y center of strike zone is 2.5 feet 
# above the ground
center_y = 2.5

# KEEP IN MIND THIS BIN CALCULATION
# IS IF EVERY HITTER HAD THE SAME STANDARD STRIKE ZONE
def calculate_bin(x, y):
    # Round to nearest integer to get number of cells in x and y directions
    cells_in_x = round((x - center_x) / (BASEBALL_DIAM/2))
    cells_in_y = round((y - center_y) / (BASEBALL_DIAM/2))

    # Calculate cell coordinates
    cell_coordinates = (center_x + cells_in_x, cells_in_y)

    return pd.Series({'bin_x': cell_coordinates[0], 'bin_z': cell_coordinates[1]})


df[['bin_x', 'bin_z']] = df.apply(lambda row: calculate_bin(row['pitch_coordinate_X'], row['pitch_coordinate_Z']), axis=1)

df.to_csv(f'{year}_binned_balls.csv', index=False)
