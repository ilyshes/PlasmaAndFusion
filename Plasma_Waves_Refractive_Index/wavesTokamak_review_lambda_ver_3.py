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
epsilon_0 = C.epsilon_0
e = C.elementary_charge
me = C.electron_mass


mu0 = C.mu_0
# Define grid resolution for frequency and density
N_freq = 1000
N_dens = 100

# Initialize arrays for results
Nsq_x_fast_arr = np.zeros((N_freq, N_dens))
Nsq_x_slow_arr = np.zeros((N_freq, N_dens))

Ksq_x_fast_arr = np.zeros((N_freq, N_dens))
Ksq_x_slow_arr = np.zeros((N_freq, N_dens))

K_fast_arr = np.zeros((N_freq, N_dens))
K_slow_arr = np.zeros((N_freq, N_dens))



#-------------   magnetic field matrix-------------------------------
B_fast_arr = np.ones((N_freq, N_dens))
B_slow_arr = np.ones((N_freq, N_dens))


#-------------   magnetic energy of the wave -------------------------------
P_B_fast_arr = 0.5*B_fast_arr**2/mu0
P_B_slow_arr = 0.5*B_slow_arr**2/mu0




# Define frequency and density arrays (logarithmic scales)
ne = np.logspace(15, 21, N_dens)
f_array = np.logspace(2, 12, N_freq)
omega_array = 2 * pi * f_array

# Define a helper function to find the nearest index for a given value
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

# Main loop to compute N^2 for each density and frequency pair


Mi = 2
Bfield = 2

# angle between k and B
theta = 89.8



for idx, n_e in enumerate(ne):
    
    Aa = []
    Ba  = []
    Fa = []
    
    
    Nstixplus = []
    Nstixminus  = []


   
    for omega in omega_array:
        
        (Nsq_plus,Nsq_minus, A, B, F) = mu.Nsq_Stix_Uncorrected(Mi,Bfield,n_e,omega,theta)
        
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



#------------------------ Now our N^2 is ready to be stoed in 2D array  -----------------------------------



    Nsq_x_fast_arr[:,idx] = Nstixplus 
    Nsq_x_slow_arr[:,idx] = Nstixminus 
   
#------------- Evaluation of k^2 -----------------------------------------------------------------------   
    Ksq_x_fast_arr[:,idx] = Nstixplus*(omega_array/c)**2
    Ksq_x_slow_arr[:,idx] = Nstixminus*(omega_array/c)**2
    
    
#------------- Evaluation of k -----------------------------------------------------------------------   
    K_fast_arr[:,idx] = np.sqrt(    Ksq_x_fast_arr[:,idx]  )
    K_slow_arr[:,idx] = np.sqrt(    Ksq_x_slow_arr[:,idx]  )
   

     

#------------------------  copy a 1D frequency array into a 2D frequency vs density array N_dens times -------------------------------------------
omega_array_2d = np.repeat(omega_array[:, np.newaxis], N_dens, axis=1)

ne_2d = np.repeat(ne[np.newaxis,:], N_freq, axis=0)



#------------- Evaluation of E -----------------------------------------------------------------------   
E_fast_arr =  omega_array_2d/K_fast_arr
E_slow_arr =  omega_array_2d/K_slow_arr
 


# Evaluate wavelength
lambd_fast_arr =   2*pi/K_fast_arr
lambd_slow_arr =   2*pi/K_slow_arr


lambd_fast_arr = lambd_fast_arr*100 # convert m -> cm
lambd_slow_arr = lambd_slow_arr*100 # convert m -> cm


#--------------- electrical energy density of an electromagnetic (EM) wave --------------------------------------



P_E_fast = 0.5*epsilon_0*E_fast_arr**2
P_E_slow = 0.5*epsilon_0*E_slow_arr**2

#------------------------------------------------------------------------------------------------------------------



#------------------------------------------  Larmour frequency -------------------------------------------

omega_ce = e*Bfield/me 

amu = C.atomic_mass
mi       = Mi*amu
omega_ci = e*Bfield/mi  
 
 #------------ From Swanson Plasma Waves, Page 37, Eq. (2.5) 
 #--------------- kinetik energy density of an electromagnetic (EM) wave in plasma --------------------------------------


v_fast_e  = (-e/me)*E_fast_arr/(omega_array_2d - (-1)*omega_ce)
v_fast_i  = (e/mi)*E_fast_arr/(omega_array_2d - omega_ci)      # --------------- resonating wave 


v_slow_e  = (-e/me)*E_slow_arr/(omega_array_2d + (-1)*omega_ce)   # --------------- resonating wave 
v_slow_i  = (e/mi)*E_slow_arr/(omega_array_2d + omega_ci)   




P_K_fast  =  ne_2d*( me*v_fast_e**2/2 + mi*v_fast_i**2/2  ) # kinetik energy
P_K_slow =  ne_2d*( me*v_slow_e**2/2 + mi*v_slow_i**2/2 )  # kinetik energy



#--------------------------------------------------------------------------------------------------------------------------------   
#--------------------------------------------------------------------------------------------------------------------------------   
#--------------------------------------------------------------------------------------------------------------------------------   
#--------------------------------------------------------------------------------------------------------------------------------   
#--------------------------------------------------------------------------------------------------------------------------------   
  
    
   
    


   
# Set up plot
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
fig.canvas.manager.window.setGeometry(0, 0, 1500, 800)
fig.suptitle('Refractive index', fontsize=16)

fig_2, axes_2 = plt.subplots(1, 2, figsize=(12, 8))
fig_2.canvas.manager.window.setGeometry(0, 0, 1500, 800)
fig_2.suptitle('Wavelength', fontsize=16)


fig_3, axes_3 = plt.subplots(1, 2, figsize=(12, 8))
fig_3.canvas.manager.window.setGeometry(0, 0, 1500, 800)
fig_3.suptitle('Electric energy density of the wave', fontsize=16)

fig_4, axes_4 = plt.subplots(1, 2, figsize=(12, 8))
fig_4.canvas.manager.window.setGeometry(0, 0, 1500, 800)
fig_4.suptitle('Kinetik energy density of the wave', fontsize=16)


x, y = np.meshgrid(ne, f_array)
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 16}

# Define colormap and normalization
cmap = LinearSegmentedColormap.from_list('blue_white_red', ['blue', 'white', 'red'], N=256)


# Plotting function
def plot_Nsq(ax, N_sq_data, figure, title, color_norm):
    N_sq =   N_sq_data
    Nthresh = 1
    N_sq[(N_sq < Nthresh) & (N_sq > -Nthresh)] = Nthresh
    N_pos = np.log10(N_sq)
    N_neg = np.log10(-N_sq)
    N_neg[np.isnan(N_neg)] = 0
    N_pos[np.isnan(N_pos)] = 0
    
    Nsq= N_pos -  N_neg

    surf = ax.contourf(x, y, Nsq, 100, cmap=cmap, norm=color_norm)
    figure.colorbar(surf, ax=ax, orientation='vertical')
    
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
    figure.tight_layout()
    
    
    
    
# Plotting function
def plot_Lambda(ax, lambda_matrix, figure, title, color_norm):
   
    lambd_log = np.log10(lambda_matrix)

    surf = ax.contourf(x, y, lambd_log, 100, cmap=cmap, norm=color_norm)
    figure.colorbar(surf, ax=ax, orientation='vertical')
    
    levels = np.linspace(-0.5, 0.5, num=5 )


    CS = ax.contour(surf, levels = levels, colors='black', linestyle='dotted')
    #ax.clabel(CS, fmt='%2.1f', colors='green', fontsize=14, inline=True)

    
    
    
    
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
    figure.tight_layout()
    
    


# Plotting function
def plot_P_E(ax, matrix, figure, title, color_norm):
   
    matrix_log = np.log10(matrix)

    surf = ax.contourf(x, y, matrix_log, 100, cmap=cmap, norm=color_norm)
    figure.colorbar(surf, ax=ax, orientation='vertical')
    
    
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
    figure.tight_layout()


# Plotting function
def plot_P_K(ax, matrix, figure, title, color_norm):
   
    matrix_log = np.log10(matrix)

    surf = ax.contourf(x, y, matrix_log, 100, cmap=cmap, norm=color_norm)
    figure.colorbar(surf, ax=ax, orientation='vertical')
    
    
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
    figure.tight_layout()



# Plot N^2 for slow and fast modes
norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=8)
plot_Nsq(axes[0], Nsq_x_slow_arr, fig, r'$N^2$ Slow Mode', norm)
plot_Nsq(axes[1], Nsq_x_fast_arr, fig, r'$N^2$ Fast Mode', norm)


# Display plots
plt.show()


# Plot lambda for slow and fast modes
norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
plot_Lambda(axes_2[0], lambd_slow_arr, fig_2, r'$log_{10}\lambda~~[cm]$, Slow wave, $\theta=%.2f ~^o$' % theta, norm)
plot_Lambda(axes_2[1], lambd_fast_arr, fig_2, r'$log_{10}\lambda~~[cm]$, Fast wave, $\theta=%.2f ~^o$' % theta, norm)


# Plot electric energy for slow and fast modes
norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=8)
plot_P_E(axes_3[0], P_E_slow, fig_3, r'$log_{10}P_E~~[J/m^2]$, Slow wave, $\theta=%.2f ~^o$' % theta, norm)
norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=10)
plot_P_E(axes_3[1], P_E_fast, fig_3, r'$log_{10}P_E~~[J/m^2]$, Fast wave, $\theta=%.2f ~^o$' % theta, norm)


# Plot electric energy for slow and fast modes
norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=8)
plot_P_K(axes_4[0], P_K_slow, fig_4, r'$log_{10}P_K~~[J/m^2]$, Slow wave, $\theta=%.2f ~^o$' % theta, norm)
norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=10)
plot_P_K(axes_4[1], P_K_fast, fig_4, r'$log_{10}P_K~~[J/m^2]$, Fast wave, $\theta=%.2f ~^o$' % theta, norm)














