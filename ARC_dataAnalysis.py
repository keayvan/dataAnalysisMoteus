#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:29:16 2025

@author: kkeramati
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

def find_extrema(signal, order_factor=0.01):
    order = max(1, int(len(signal) * order_factor))
    local_max_indices = argrelextrema(signal, np.greater, order=order)[0]
    local_min_indices = argrelextrema(signal, np.less, order=order)[0]
    return local_min_indices, local_max_indices

def fit_curve(x, y, degree=5, extrapolate_x=None):
    if len(x) > degree:
        coeffs = np.polyfit(x, y, degree)
        poly_func = np.poly1d(coeffs)
        if extrapolate_x is not None:
            return poly_func(extrapolate_x)
        return poly_func(x)
    return np.full_like(extrapolate_x if extrapolate_x is not None else x, np.mean(y))


# Load the CSV file
file_path = "Telem ESC 01_2025.csv"
df = pd.read_csv(file_path, skiprows=9)

# Rename columns based on row 7 (found earlier)
df.columns = [
    "Unknown0","Time_sms", "Time","VOLTAGE", "CURRENT", "SPEED",
    "Controller temperature", "BEC voltage", "BEC current",
    "BEC temperature", "Input request","Incoming power",
    "Unknown1", "Unknown2", "Unknown3", "Unknown4",
    "Unknown5", "Unknown6", "Unknown7", "Unknown8",
    "Unknown9", "Unknown10", "Unknown11", "Unknown12",
    "Unknown13", "Pulse current"
]

colors = ['#009494ff', '#00d0b8', '#0FAAF0', '#f74242ff', '#525252ff', '#f7941dff', '#0F3878', '#9e0012ff', 'black', 'red']

# Convert numeric columns to proper data types
numeric_cols = ["VOLTAGE", "CURRENT", "SPEED", "Incoming power"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
# Multiply "Motor revolutions" column by 100
df_cleaned.loc[:, "SPEED"] *= 100

# Plot data
fig0, axes0 = plt.subplots(2, 2, figsize=(14, 10))
fig0.tight_layout(pad=4.0)
axes0 = axes0.flatten()
fig0.suptitle('Raw Data',fontsize = 20)
for i,(ax0, param) in enumerate(zip(axes0, numeric_cols)):
    ax0.scatter(df['Time'], df[param], label=f'{param}',s =6, color=colors[i] )
    ax0.set_xlabel('Time')
    ax0.set_ylabel(param)
    ax0.set_ylim(0,df[param].max()*1.3)
    ax0.set_title(f'{param} vs Time')
    ax0.legend()
    ax0.grid(True)

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.tight_layout(pad=4.0)
axes1 = axes1.flatten()
fig1.suptitle('Cleaned Data',fontsize = 20)
for i,(ax1, param) in enumerate(zip(axes1, numeric_cols)):
    ax1.scatter(df_cleaned['Time'], df_cleaned[param], label=f'{param}',s =6, color=colors[i] )
    ax1.set_xlabel('Time')
    ax1.set_ylabel(param)
    ax1.set_ylim(0, df_cleaned[param].max()*1.3)
    ax1.set_title(f'{param} vs Time')
    ax1.legend()
    ax1.grid(True)


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.tight_layout(pad=4.0)
axes = axes.flatten()
for i,(ax, param) in enumerate(zip(axes, numeric_cols)):
    ax.scatter(df['SPEED'], df[param], label=f'{param}',s =6, color=colors[i] )
    ax.set_xlabel('Time')
    ax.set_ylabel(param)
    ax.set_ylim(0,df_cleaned[param].max()*1.3)
    ax.set_title(f'{param} vs Time')
    ax.legend()
    ax.grid(True)
    
df_sorted = df_cleaned.sort_values(by="SPEED", ascending=True)
time_new = list(range(len(df_sorted['Time'])))
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.tight_layout(pad=4.0)
axes1 = axes1.flatten()
for i,(ax, param) in enumerate(zip(axes1, numeric_cols)):
    ax.scatter(time_new, df_sorted[param], label=f'{param}',s =6, color=colors[i] )
    ax.set_xlabel('Time')
    ax.set_ylabel(param)
    ax.set_ylim(0/1.3,df_cleaned[param].max()*1.3)
    ax.set_title(f'{param} vs Time')
    ax.legend()
    ax.grid(True)
    

def bound_func(current):
    local_maxima_indices = argrelextrema(current, np.greater, order=5)[0]
    local_minima_indices = argrelextrema(current, np.less, order=5)[0]
    
    if len(local_maxima_indices) > 0:
        local_maxima = current[local_maxima_indices]
        max_time = [time_new[i] for i in local_maxima_indices]
        upper_bound = fit_curve(max_time, local_maxima, extrapolate_x=time_new)
    else:
        upper_bound = np.full_like(time_new, np.max(current))
    
    if len(local_minima_indices) > 0:
        local_minima = current[local_minima_indices]
        min_time = [time_new[i] for i in local_minima_indices]
        lower_bound = fit_curve(min_time, local_minima, extrapolate_x=time_new)
    else:
        lower_bound = np.full_like(time_new, np.min(current))
        
    bound_range = np.max(current) - np.min(current)
    expansion_factor = 0.01 * bound_range 
    upper_bound_expanded = upper_bound + expansion_factor
    lower_bound_expanded = lower_bound - expansion_factor
    return time_new, time_new, lower_bound_expanded, upper_bound_expanded

fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))
fig2.tight_layout(pad=4.0)
axes2 = axes2.flatten()
for i,(ax, param) in enumerate(zip(axes2, ['CURRENT','Incoming power'])):
    sig = df_sorted[param].dropna().to_numpy()
    min_time, max_time, lower_bound_expanded, upper_bound_expanded = bound_func(sig)

    ax.scatter(time_new, sig, label=f'{param}',s =6, color=colors[1] )
    ax.plot(max_time, upper_bound_expanded, label="Upper Bound", color=colors[3])
    ax.plot(min_time, lower_bound_expanded, label="Lower Bound", color=colors[2])

    ax.set_xlabel('Time')
    ax.set_ylabel(param)
    ax.set_ylim(0/1.3,df_cleaned[param].max()*1.3)
    ax.set_title(f'{param} vs Time')
    ax.legend()
    ax.grid(True)

