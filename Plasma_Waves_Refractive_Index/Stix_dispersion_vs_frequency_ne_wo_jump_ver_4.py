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

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import LogLocator

plt.close("all")

# Constants
pi = np.pi
c = C.speed_of_light

# Define grid resolution for frequency and density
N_freq = 1000
N_dens = 500

# Initialize arrays for results
Nsq_x_fast_arr = np.zeros((N_freq, N_dens))
Nsq_x_slow_arr = np.zeros((N_freq, N_dens))

# Define frequency and density arrays (logarithmic scales)
ne = np.logspace(15, 21, N_dens)
f_array = np.logspace(2, 12, N_freq)
omega_array = 2 * pi * f_array

# Define a helper function to find the nearest index for a given value
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

# Main loop to compute N^2 for each density and frequency pair




for idx, n_e in enumerate(ne):
    
    Aa = []
    Ba  = []
    Fa = []
    
    
    Nstixplus = []
    Nstixminus  = []


   
    for omega in omega_array:
        
        
             
        
        Mi = 2
        Bfield = 2
        #mi = Mi*amu
        
        
        
        (Nsq_plus,Nsq_minus, A, B, F) = mu.Nsq_Stix_Uncorrected(Mi,Bfield,n_e,omega,75)
        
        Nstixplus.append(Nsq_plus)
        Nstixminus.append(Nsq_minus)
        
        Aa.append(A)
        Ba.append(B)
        Fa.append(F)
    
   
    # Swap plus/minus refractive indices for defined frequency ranges    
    # Swap plus/minus refractive indices for defined frequency ranges   
    # Swap plus/minus refractive indices for defined frequency ranges   
    # Swap plus/minus refractive indices for defined frequency ranges   
    # Swap plus/minus refractive indices for defined frequency ranges   
    #------------------------ first max index -------------------------------------
    
    f_truncated = f_array[:-1]
    F_diff = np.diff(Fa)
     
     
    F_diff[np.isnan(F_diff)] = 0
     
    first_max_index = np.argmax(F_diff)
     
     
    first_f_max = f_truncated[first_max_index]
     
    #--------------------------------- last max index -----------------------------------------
     
    f_idx = find_nearest(f_truncated, 8e9)
    f_sub = f_truncated[f_idx:]
    F_diff_sub = F_diff[f_idx:]
     
     
     
     
    last_max_index_sub = np.argmax(F_diff_sub)
     
     
    last_max_index = np.where(F_diff == F_diff_sub[last_max_index_sub])[0]
    last_f_max = f_truncated[last_max_index]
     
     
    #-------------------------------------------------------------------------
     
    first_max_index = first_max_index + 2
    tmp = Nstixplus[:first_max_index]
    Nstixplus[:first_max_index] = Nstixminus[:first_max_index]
    Nstixminus[:first_max_index] = tmp
     
     
     
    last_max_index = last_max_index[0] + 1
    tmp = Nstixplus[last_max_index:]
    Nstixplus[last_max_index:] = Nstixminus[last_max_index:]
    Nstixminus[last_max_index:] = tmp


    Nsq_x_fast_arr[:,idx] = Nstixplus 
    Nsq_x_slow_arr[:,idx] = Nstixminus 
   
    


#--------------------------------------------------------------------------------------------------------------------------------   
#--------------------------------------------------------------------------------------------------------------------------------   
#--------------------------------------------------------------------------------------------------------------------------------   
#--------------------------------------------------------------------------------------------------------------------------------   
#--------------------------------------------------------------------------------------------------------------------------------   
  
    
   
    


   
# Set up plot
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
x, y = np.meshgrid(ne, f_array)
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 16}

# Define colormap and normalization
cmap = LinearSegmentedColormap.from_list('blue_white_red', ['blue', 'white', 'red'], N=256)
norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=8)

# Plotting function
def plot_Nsq(ax, N_sq_data, title):
    N_sq =   N_sq_data
    Nthresh = 1
    N_sq[(N_sq < Nthresh) & (N_sq > -Nthresh)] = Nthresh
    N_pos = np.log10(N_sq)
    N_neg = np.log10(-N_sq)
    N_neg[np.isnan(N_neg)] = 0
    N_pos[np.isnan(N_pos)] = 0
    
    Nsq= N_pos -  N_neg

    surf = ax.contourf(x, y, Nsq, 100, cmap=cmap, norm=norm)
    fig.colorbar(surf, ax=ax, orientation='vertical')
    
    ax.set_title(title, fontdict=font)
    ax.set_xlabel(r'$n_e [m^{-3}]$', fontdict=font)
    ax.set_ylabel(r'$f [Hz]$', fontdict=font)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Set y-axis tick spacing
    
    # Set custom log y-ticks spacing
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))  # Base 10 with up to 10 ticks
    
    # Set minor ticks in between major ticks on a log scale
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    
    # Enable grid for both major and minor ticks
    ax.grid(visible=True, which='major', linestyle='--', linewidth=0.7)
    

    # Enable grid for both major and minor ticks
    ax.grid(visible=True, which='minor', linestyle='--', linewidth=0.3)

# Plot N^2 for slow and fast modes
plot_Nsq(axes[0], Nsq_x_slow_arr, r'$N^2$ Slow Mode')
plot_Nsq(axes[1], Nsq_x_fast_arr, r'$N^2$ Fast Mode')

# Display plots
plt.show()


















