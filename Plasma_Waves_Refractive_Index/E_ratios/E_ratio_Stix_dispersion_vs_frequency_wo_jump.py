# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:25:26 2018

@author: ilys
"""



# This file is created to compare the angular dependence of the resonance frequencies, according to the Ginsburg Bogdankevich Ruckadze side 116
# This file is created to compare the angular dependence of the resonance frequencies, according to the Ginsburg Bogdankevich Ruckadze side 116
# This file is created to compare the angular dependence of the resonance frequencies, according to the Ginsburg Bogdankevich Ruckadze side 116
# This file is created to compare the angular dependence of the resonance frequencies, according to the Ginsburg Bogdankevich Ruckadze side 116



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

Sa = []
Da  = []
Pa = []


Nsq_x_fast_arr = []
Nsq_x_slow_arr = []

Nsq_x_plus_arr = []
Nsq_x_minus_arr = []



font = {'family': 'serif',
        'color':  'black',
        'weight': 'heavy',
        'size': 16,
        }



# Frequency and plasma parameters
f_array = np.logspace(6, 12, 20000)
omega_array = 2 * pi * f_array
n_e = 1e19
Mi, Bfield = 2, 2  # Ion mass and magnetic field


#----- LH resonance -----------------------------------------

f_LH = mu.f_LH(Mi,Bfield,n_e) #  low hybrid
f_UH = mu.f_UH(Bfield,n_e)   # upper hybrid

f_ci = mu.f_ci(Mi,Bfield)   # ion cyclotron wave
f_ce = mu.f_ce(Bfield)      # electron cyclotron wave

angle = 88.9
# Loop over frequencies to calculate dispersion relations
for omega in omega_array:
    Nsq_plus, Nsq_minus, A, B, F, S, D, P = mu.Nsq_Stix_Uncorrected_SDP(Mi, Bfield, n_e, omega, angle)
    Nstixplus.append(Nsq_plus)
    Nstixminus.append(Nsq_minus)
    Aa.append(A)
    Ba.append(B)
    Fa.append(F)

   
    Sa.append(S)
    Da.append(D)
    Pa.append(P)
    









# Plot F and find maximum points

f_truncated = f_array[:-1]
F_diff = np.diff(Fa)
F_diff[np.isnan(F_diff)] = 0

# Find first and last significant peaks
first_max_index = np.argmax(F_diff)
first_f_max = f_truncated[first_max_index]
#plt.plot([first_f_max, first_f_max], [-1e10, 1e16], color='black')

f_idx = find_nearest(f_truncated, 8e9)
last_max_index_sub = np.argmax(F_diff[f_idx:])
last_max_index = np.where(F_diff == F_diff[f_idx:][last_max_index_sub])[0][0]
last_f_max = f_truncated[last_max_index]
#plt.plot([last_f_max, last_f_max], [-1e10, 1e16], color='black')



# Swap values for visualization purposes
first_max_index += 2
last_max_index += 2
Nstixplus[:first_max_index], Nstixminus[:first_max_index] = Nstixminus[:first_max_index], Nstixplus[:first_max_index]
Nstixplus[last_max_index:], Nstixminus[last_max_index:] = Nstixminus[last_max_index:], Nstixplus[last_max_index:]



 

###########################################################################################################
# Convert lists to numpy arrays
Sa = np.array(Sa)
Da = np.array(Da)
Pa = np.array(Pa)
Nstixplus = np.array(Nstixplus)
Nstixminus = np.array(Nstixminus)


###############################################   Evaluate ratios  ##############################################

#------------- Evaluation of k^2 -----------------------------------------------------------------------   
Ksq_plus  = Nstixplus*(omega_array/c)**2
Ksq_minus = Nstixminus*(omega_array/c)**2

#------------- Evaluation of k -----------------------------------------------------------------------   
K_plus = np.sqrt(   Ksq_plus  )
K_minus = np.sqrt(   Ksq_minus )



# Evaluate wavelength
lambd_plus =   2*pi/K_plus
lambd_minus =   2*pi/K_minus


lambd_plus_cm = lambd_plus*100 # convert m -> cm
lambd_minus_cm = lambd_minus*100 # convert m -> cm




Ex_over_Ey_plus = (Nstixplus-Sa)/Da
Ex_over_Ey_minus = (Nstixminus-Sa)/Da


Ey_over_Ex_plus = 1/Ex_over_Ey_plus
Ey_over_Ex_minus = 1/Ex_over_Ey_minus


pi = np.pi
theta_rad = angle*pi/180            # theta ist in radian


Ex_over_Ez_plus  = (  Nstixplus*np.sin(theta_rad)**2 - Pa  )/(   Nstixplus*np.sin(theta_rad)*np.cos(theta_rad) )
Ex_over_Ez_minus = (  Nstixminus*np.sin(theta_rad)**2 - Pa  )/(   Nstixminus*np.sin(theta_rad)*np.cos(theta_rad) )

Ez_over_Ex_plus = 1/Ex_over_Ez_plus
Ez_over_Ex_minus = 1/Ex_over_Ez_minus







#----------------------------- PLOT PLOT PLOT ----------------------------------------------------
#----------------------------- PLOT PLOT PLOT ----------------------------------------------------
#----------------------------- PLOT PLOT PLOT ----------------------------------------------------
#----------------------------- PLOT PLOT PLOT ----------------------------------------------------
#----------------------------- PLOT PLOT PLOT ----------------------------------------------------

Lx = 12
Ly = 6


# Plot setup
plt.figure(1, figsize=(Lx, Ly))
plt.xscale('log')
plt.yscale('symlog')
plt.rcParams['text.usetex'] = True
plt.plot(f_array, Fa, '-x', color='green', markersize=5, label='F')

# Plot N^2 values
plt.plot(f_array, Nstixplus, '-x', color='r', markersize=5, label='plus')
plt.plot(f_array, Nstixminus, '-x', color='b', markersize=5, label='minus')

# Labeling and display settings
plt.legend(loc='upper right')
plt.title(r'Stix Dispersion vs Frequency, angle  = $%3.2f^o$' % angle, fontsize=16)
plt.xlabel(r'$f~[Hz]$', fontsize=18)
plt.ylabel(r'$N^2$', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.axhline(0, color='black', linestyle='--')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.grid(True, which='major', linestyle='-', linewidth=0.8)


#---------------------- plot LH resonance -------------------------
plt.plot([f_LH, f_LH], [-1e10, 1e16], color='black')
plt.text(f_LH , 0.5, r'$f_{LH}=%.2e Hz$' % f_LH, fontdict=font, rotation=90)


#---------------------- plot UH resonance -------------------------
plt.plot([f_UH, f_UH], [-1e10, 1e16], color='black')
plt.text(f_UH , 1e6, r'$f_{UH}=%.2e Hz$' % f_UH, fontdict=font, rotation=90)


#---------------------- plot UH resonance -------------------------
plt.plot([f_ci, f_ci], [-1e10, 1e16], color='black')
plt.text(f_ci , 0.5, r'$f_{ci}=%.2e Hz$' % f_ci, fontdict=font, rotation=90)

#---------------------- plot UH resonance -------------------------
plt.plot([f_ce, f_ce], [-1e10, 1e16], color='blue')
plt.text(f_ce , -1e6, r'$f_{ce}=%.2e Hz$' % f_ce, fontdict=font, rotation=90)


# Save figure
filename = "Stix_dispersion_vs_frequency_wo_jump_Nsq_ver3.png"
plt.savefig(filename, bbox_inches='tight')
plt.show()




#####################################################################################################################





# Plot setup
plt.figure(2, figsize=(Lx, Ly))
plt.xscale('log')
plt.yscale('symlog')
plt.rcParams['text.usetex'] = True
plt.plot(f_array, Fa, '-x', color='green', markersize=5, label='F')

# Plot N^2 values
plt.plot(f_array, Ex_over_Ey_plus, '-x', color='r', markersize=5, label='Ex/Ey plus')
plt.plot(f_array, Ex_over_Ey_minus, '-x', color='b', markersize=5, label='Ex/Ey minus')

# Labeling and display settings
plt.legend(loc='upper right')
plt.title(r'Stix Dispersion vs Frequency, angle  = $%3.2f^o$' % angle, fontsize=16)
plt.xlabel(r'$f~[Hz]$', fontsize=18)
plt.ylabel(r'$Ex/Ey$', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.axhline(0, color='black', linestyle='--')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.grid(True, which='major', linestyle='-', linewidth=0.8)


#---------------------- plot LH resonance -------------------------
plt.plot([f_LH, f_LH], [-1e10, 1e16], color='black')
plt.text(f_LH , 0.5, r'$f_{LH}=%.2e Hz$' % f_LH, fontdict=font, rotation=90)


#---------------------- plot UH resonance -------------------------
plt.plot([f_UH, f_UH], [-1e10, 1e16], color='black')
plt.text(f_UH , 1e6, r'$f_{UH}=%.2e Hz$' % f_UH, fontdict=font, rotation=90)


#---------------------- plot UH resonance -------------------------
plt.plot([f_ci, f_ci], [-1e10, 1e16], color='black')
plt.text(f_ci , 0.5, r'$f_{ci}=%.2e Hz$' % f_ci, fontdict=font, rotation=90)

#---------------------- plot UH resonance -------------------------
plt.plot([f_ce, f_ce], [-1e10, 1e16], color='blue')
plt.text(f_ce , -1e6, r'$f_{ce}=%.2e Hz$' % f_ce, fontdict=font, rotation=90)


# Save figure
filename = "E_ratio_Stix_dispersion_vs_frequency_wo_jump_Ex_over_Ey.png"
plt.savefig(filename, bbox_inches='tight')
plt.show()



#####################################################################################################################





# Plot setup
plt.figure(3, figsize=(Lx, Ly))
plt.xscale('log')
plt.yscale('symlog')
plt.rcParams['text.usetex'] = True
plt.plot(f_array, Fa, '-x', color='green', markersize=5, label='F')

# Plot N^2 values
plt.plot(f_array, Ex_over_Ez_plus, '-x', color='r', markersize=5, label='Ex/Ez plus')
plt.plot(f_array, Ex_over_Ez_minus, '-x', color='b', markersize=5, label='Ex/Ez minus')

# Labeling and display settings
plt.legend(loc='upper right')
plt.title(r'Stix Dispersion vs Frequency, angle  = $%3.2f^o$' % angle, fontsize=16)
plt.xlabel(r'$f~[Hz]$', fontsize=18)
plt.ylabel(r'$Ex/Ez$', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.axhline(0, color='black', linestyle='--')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.grid(True, which='major', linestyle='-', linewidth=0.8)


#---------------------- plot LH resonance -------------------------
plt.plot([f_LH, f_LH], [-1e10, 1e16], color='black')
plt.text(f_LH , 0.5, r'$f_{LH}=%.2e Hz$' % f_LH, fontdict=font, rotation=90)


#---------------------- plot UH resonance -------------------------
plt.plot([f_UH, f_UH], [-1e10, 1e16], color='black')
plt.text(f_UH , 1e6, r'$f_{UH}=%.2e Hz$' % f_UH, fontdict=font, rotation=90)


#---------------------- plot UH resonance -------------------------
plt.plot([f_ci, f_ci], [-1e10, 1e16], color='black')
plt.text(f_ci , 0.5, r'$f_{ci}=%.2e Hz$' % f_ci, fontdict=font, rotation=90)

#---------------------- plot UH resonance -------------------------
plt.plot([f_ce, f_ce], [-1e10, 1e16], color='blue')
plt.text(f_ce , -1e6, r'$f_{ce}=%.2e Hz$' % f_ce, fontdict=font, rotation=90)


# Save figure
filename = "E_ratio_Stix_dispersion_vs_frequency_wo_jump_Ex_over_Ez.png"
plt.savefig(filename, bbox_inches='tight')
plt.show()




#####################################################################################################################





# Plot setup
plt.figure(4, figsize=(Lx, Ly))
plt.xscale('log')
plt.yscale('symlog')
plt.rcParams['text.usetex'] = True
plt.plot(f_array, Fa, '-x', color='green', markersize=5, label='F')

# Plot N^2 values
plt.plot(f_array, 1/Ex_over_Ey_plus, '-x', color='r', markersize=5, label='Ey/Ex plus')
plt.plot(f_array, 1/Ex_over_Ey_minus, '-x', color='b', markersize=5, label='Ey/Ex minus')

# Labeling and display settings
plt.legend(loc='upper right')
plt.title(r'Stix Dispersion vs Frequency, angle  = $%3.2f^o$' % angle, fontsize=16)
plt.xlabel(r'$f~[Hz]$', fontsize=18)
plt.ylabel(r'$Ey/Ex$', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.axhline(0, color='black', linestyle='--')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.grid(True, which='major', linestyle='-', linewidth=0.8)


#---------------------- plot LH resonance -------------------------
plt.plot([f_LH, f_LH], [-1e10, 1e16], color='black')
plt.text(f_LH , 0.5, r'$f_{LH}=%.2e Hz$' % f_LH, fontdict=font, rotation=90)


#---------------------- plot UH resonance -------------------------
plt.plot([f_UH, f_UH], [-1e10, 1e16], color='black')
plt.text(f_UH , 1e6, r'$f_{UH}=%.2e Hz$' % f_UH, fontdict=font, rotation=90)


#---------------------- plot UH resonance -------------------------
plt.plot([f_ci, f_ci], [-1e10, 1e16], color='black')
plt.text(f_ci , 0.5, r'$f_{ci}=%.2e Hz$' % f_ci, fontdict=font, rotation=90)

#---------------------- plot UH resonance -------------------------
plt.plot([f_ce, f_ce], [-1e10, 1e16], color='blue')
plt.text(f_ce , -1e6, r'$f_{ce}=%.2e Hz$' % f_ce, fontdict=font, rotation=90)


# Save figure
filename = "E_ratio_Stix_dispersion_vs_frequency_wo_jump_Ey_over_Ex.png"
plt.savefig(filename, bbox_inches='tight')
plt.show()





#####################################################################################################################





# Plot setup
plt.figure(5, figsize=(Lx, Ly))
plt.xscale('log')
plt.yscale('symlog')
plt.rcParams['text.usetex'] = True
plt.plot(f_array, Fa, '-x', color='green', markersize=5, label='F')

# Plot N^2 values
plt.plot(f_array, 1/Ex_over_Ez_plus, '-x', color='r', markersize=5, label='Ez/Ex plus')
plt.plot(f_array, 1/Ex_over_Ez_minus, '-x', color='b', markersize=5, label='Ez/Ex minus')

# Labeling and display settings
plt.legend(loc='upper right')
plt.title(r'Stix Dispersion vs Frequency, angle  = $%3.2f^o$' % angle, fontsize=16)
plt.xlabel(r'$f~[Hz]$', fontsize=18)
plt.ylabel(r'$Ez/Ex$', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.axhline(0, color='black', linestyle='--')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.grid(True, which='major', linestyle='-', linewidth=0.8)


#---------------------- plot LH resonance -------------------------
plt.plot([f_LH, f_LH], [-1e10, 1e16], color='black')
plt.text(f_LH , 0.5, r'$f_{LH}=%.2e Hz$' % f_LH, fontdict=font, rotation=90)


#---------------------- plot UH resonance -------------------------
plt.plot([f_UH, f_UH], [-1e10, 1e16], color='black')
plt.text(f_UH , 1e6, r'$f_{UH}=%.2e Hz$' % f_UH, fontdict=font, rotation=90)


#---------------------- plot UH resonance -------------------------
plt.plot([f_ci, f_ci], [-1e10, 1e16], color='black')
plt.text(f_ci , 0.5, r'$f_{ci}=%.2e Hz$' % f_ci, fontdict=font, rotation=90)

#---------------------- plot UH resonance -------------------------
plt.plot([f_ce, f_ce], [-1e10, 1e16], color='blue')
plt.text(f_ce , -1e6, r'$f_{ce}=%.2e Hz$' % f_ce, fontdict=font, rotation=90)


# Save figure
filename = "E_ratio_Stix_dispersion_vs_frequency_wo_jump_Ez_over_Ex.png"
plt.savefig(filename, bbox_inches='tight')
plt.show()


#####################################################################################################################





# Plot setup
plt.figure(6, figsize=(Lx, Ly))
plt.xscale('log')
plt.yscale('symlog')
plt.rcParams['text.usetex'] = True
plt.plot(f_array, Fa, '-x', color='green', markersize=5, label='F')

# Plot N^2 values
plt.plot(f_array, lambd_plus_cm, '-x', color='r', markersize=5, label='lambd_plus_cm')
plt.plot(f_array, lambd_minus_cm, '-x', color='b', markersize=5, label='lambd_minus_cm')

# Labeling and display settings
plt.legend(loc='upper right')
plt.title(r'Stix Dispersion vs Frequency, angle  = $%3.2f^o$' % angle, fontsize=16)
plt.xlabel(r'$f~[Hz]$', fontsize=18)
plt.ylabel(r'$Ez/Ex$', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.axhline(0, color='black', linestyle='--')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.grid(True, which='major', linestyle='-', linewidth=0.8)


#---------------------- plot LH resonance -------------------------
plt.plot([f_LH, f_LH], [-1e10, 1e16], color='black')
plt.text(f_LH , 0.5, r'$f_{LH}=%.2e Hz$' % f_LH, fontdict=font, rotation=90)


#---------------------- plot UH resonance -------------------------
plt.plot([f_UH, f_UH], [-1e10, 1e16], color='black')
plt.text(f_UH , 1e6, r'$f_{UH}=%.2e Hz$' % f_UH, fontdict=font, rotation=90)


#---------------------- plot UH resonance -------------------------
plt.plot([f_ci, f_ci], [-1e10, 1e16], color='black')
plt.text(f_ci , 0.5, r'$f_{ci}=%.2e Hz$' % f_ci, fontdict=font, rotation=90)

#---------------------- plot UH resonance -------------------------
plt.plot([f_ce, f_ce], [-1e10, 1e16], color='blue')
plt.text(f_ce , -1e6, r'$f_{ce}=%.2e Hz$' % f_ce, fontdict=font, rotation=90)


# Save figure
filename = "E_ratio_Stix_dispersion_vs_frequency_wo_jump_lambda.png"
plt.savefig(filename, bbox_inches='tight')
plt.show()


