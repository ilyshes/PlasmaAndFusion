# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:25:26 2018

@author: ilys
"""

import numpy as np
import pickle
import os 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import myutulities as mu
from os import listdir
from os.path import isfile, join
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d

import scipy.constants as C
from matplotlib import rc
plt.close("all")

pi = np.pi
c = C.speed_of_light

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    



Nstixplus = []
Nstixminus  = []


Aa = []
Ba  = []
Fa = []



Nsq_x_fast_arr = []
Nsq_x_slow_arr = []

Nsq_x_plus_arr = []
Nsq_x_minus_arr = []






# Frequency and plasma parameters
f_array = np.logspace(6, 12, 20000)
omega_array = 2 * pi * f_array
n_e = 1e20
Mi, Bfield = 2, 2  # Ion mass and magnetic field

# Loop over frequencies to calculate dispersion relations
for omega in omega_array:
    Nsq_plus, Nsq_minus, A, B, F = mu.Nsq_Stix_Uncorrected(Mi, Bfield, n_e, omega, 75)
    Nstixplus.append(Nsq_plus)
    Nstixminus.append(Nsq_minus)
    Aa.append(A)
    Ba.append(B)
    Fa.append(F)

   

    


# Plot setup
plt.figure(figsize=(20, 10))
plt.xscale('log')
plt.yscale('symlog')
plt.rcParams['text.usetex'] = True

# Plot F and find maximum points
plt.plot(f_array, Fa, '-x', color='green', markersize=5, label='F')
f_truncated = f_array[:-1]
F_diff = np.diff(Fa)
F_diff[np.isnan(F_diff)] = 0

# Find first and last significant peaks
first_max_index = np.argmax(F_diff)
first_f_max = f_truncated[first_max_index]
plt.plot([first_f_max, first_f_max], [-1e10, 1e16], color='black')

f_idx = find_nearest(f_truncated, 8e9)
last_max_index_sub = np.argmax(F_diff[f_idx:])
last_max_index = np.where(F_diff == F_diff[f_idx:][last_max_index_sub])[0][0]
last_f_max = f_truncated[last_max_index]
plt.plot([last_f_max, last_f_max], [-1e10, 1e16], color='black')

# Swap values for visualization purposes
first_max_index += 2
last_max_index += 2
Nstixplus[:first_max_index], Nstixminus[:first_max_index] = Nstixminus[:first_max_index], Nstixplus[:first_max_index]
Nstixplus[last_max_index:], Nstixminus[last_max_index:] = Nstixminus[last_max_index:], Nstixplus[last_max_index:]

# Plot N^2 values
plt.plot(f_array, Nstixplus, '-x', color='r', markersize=5, label='Fast wave')
plt.plot(f_array, Nstixminus, '-x', color='b', markersize=5, label='Slow wave')

# Labeling and display settings
plt.legend(loc='upper right')
plt.title('Stix Dispersion vs Frequency', fontsize=16)
plt.xlabel(r'$f~[Hz]$', fontsize=18)
plt.ylabel(r'$N^2$', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.axhline(0, color='black', linestyle='--')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.grid(True, which='major', linestyle='-', linewidth=0.8)

# Save figure
filename = "wavesTokamakStix.png"
plt.savefig(filename, bbox_inches='tight')
plt.show()

















