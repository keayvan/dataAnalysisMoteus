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
motor = 'motor01/'
tstNo = 'test04/'
fldr =  'L_measurment_LCR_28Feb2025/'
# fldr =  ''

file_path = './data/motorParameters/'
file_path = file_path+fldr+motor+tstNo
files = ['RY','YB', 'BR'] 

df_all = []
for i in range(len(files)):
    with open(f'{file_path+files[i]}.csv', "rb") as f:
        raw_data = f.read(10000)  # Read a portion of the file for detection
    
    encoding_info = chardet.detect(raw_data)
    encoding_info
    df = pd.read_csv(f'{file_path+files[i]}.csv', skiprows=11)
    df['thetaRad']=df['Θd[°]']*np.pi/180
    df['resistance_theta'] =  df['Z[Ohm]']*np.cos(df['thetaRad'])
    df['resistance_sqrt'] =  np.sqrt(df['Z[Ohm]']**2 - 2*np.pi*df['Frequency[Hz]']*df['Ls[H]']**2)
    df['calc_reactance'] = np.sqrt(df['Z[Ohm]']**2-0.0086**2)
    df['calc_inductance'] = df['calc_reactance']/2/np.pi/df['Frequency[Hz]']


    df_all.append(df)

colors_dict = {
    "dark_red": "#9e0012ff",
    "orange": "#f7941d",
    "gray": "#525252ff",
    "teal": "#009494ff",
    "turquoise": "#00d0b8",
    "navy_blue": "#0F3878",
    "sky_blue": "#0FAAF0",
    "red": "#f74242ff",
    "black": "black"}
colors = list(colors_dict.values())

columns = ['Ls[H]', 'Rs[Ohm]'] 
max_var = [2e-5, 0.03]
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
fig.tight_layout(pad=6.0)

for j in range(len(files)):
    print('*************************')
    for i, column in enumerate(columns):
        print(f'{column}_{files[j]}: {df_all[j][column][100:].mean()}')
        ax[j,i].plot(df_all[i]['Frequency[Hz]'], df_all[i][column], label=f'{column}_{files[j]}', marker = 'o', color = colors[i+2])
    
        ax[j, i].set_xlabel("Frequency[Hz]")
        ax[j, i].set_ylabel(f"{column}")
        ax[j, i].set_title(f"{column}")
        ax[j, i].set_ylim(0,max_var[i])
        ax[j, i].legend(loc="upper right")
        ax[j, i].grid(True)
    
    plt.show()

fig, ax = plt.subplots(1,2 , figsize = (10, 5))
ax = np.ravel([ax])
for k in range(len(files)):
    for i in range(len(columns)):
        ax[i].plot(df_all[k]['Frequency[Hz]'], df_all[k][columns[i]], color= colors[k], lw = 2, label = f'{files[k]}')
        ax[i].grid()
        ax[i].legend()
        ax[i].set_ylim(0,max_var[i])
        ax[i].set_title(columns[i])
plt.show()

# columns = ['Rs[Ohm]','resistance_theta', 'resistance_sqrt'] 

# fig, ax = plt.subplots(1,3 , figsize = (10, 5))
# ax = np.ravel([ax])
# for k in range(len(files)):
#     for i in range(len(columns)):
#         ax[i].plot(df_all[k]['Frequency[Hz]'], df_all[k][columns[i]], color= colors[k], lw = 2, label = f'{files[k]}')
#         ax[i].grid()
#         ax[i].legend()
#         ax[i].set_title(columns[i])
       
#     plt.show()
    
# columns = ['Ls[H]','calc_inductance'] 
# max_var = [2e-5, 2e-5]

# fig, ax = plt.subplots(1,2 , figsize = (10, 5))
# ax = np.ravel([ax])
# for k in range(len(files)):
#     for i in range(len(columns)):
#         ax[i].plot(df_all[k]['Frequency[Hz]'], df_all[k][columns[i]], color= colors[k], lw = 2, label = f'{files[k]}')
#         ax[i].grid()
#         ax[i].legend()
#         ax[i].set_ylim(0, max_var[i])
#         ax[i].set_title(columns[i])
       
#     plt.show()