#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:02:58 2025

@author: kkeramati
"""

# Detect file encoding
import chardet
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
file_path = './data/motorParameters/log-20250220T174405.545.csv'
with open(file_path, "rb") as f:
    raw_data = f.read(10000)  # Read a portion of the file for detection

encoding_info = chardet.detect(raw_data)
encoding_info

# Read the CSV again, skipping metadata lines
df = pd.read_csv(file_path, skiprows=11)

# Display the first few rows to check if the data is correctly loaded
# Plot all numerical columns using Matplotlib

# Convert the Timestamp column to string and extract the milliseconds part
df['tt'] = df['Timestamp'].astype(str).str.split(":").str[-1].astype(float)

# Plot all parameters against the modified Timestamp
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout(pad=6.0)
colors_dict = {
    "gray": "#525252ff",
    "teal": "#009494ff",
    "turquoise": "#00d0b8",
    "sky_blue": "#0FAAF0",
    "red": "#f74242ff",
    "orange": "#f7941df",
    "navy_blue": "#0F3878",
    "dark_red": "#9e0012ff",
    "black": "black"}
colors = list(colors_dict.values())

ax = np.ravel([axes])
columns = ['Ls[H]', 'Rs[Ohm]'] 
for i, column in enumerate(columns):
    ax[i].plot(df['Frequency[Hz]'], df[column], label=column, marker = 'o', color = colors[i])

    ax[i].set_xlabel("Frequency[Hz]")
    ax[i].set_ylabel(f"{column}")
    ax[i].set_title(f"{column}")
    ax[i].legend(loc="upper right")
    ax[i].grid(True)

# Show the plot
plt.show()

