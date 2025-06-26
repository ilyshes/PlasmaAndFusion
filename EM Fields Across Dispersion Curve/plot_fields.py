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
#from PyPDF2 import PdfWriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import myutulities as mu
from os import listdir
from os.path import isfile, join
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import locale





import scipy.constants as C
from matplotlib import rc
plt.close("all")

pi = np.pi
c = C.speed_of_light
mu0 = C.mu_0

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


N_freq = 20000
# Frequency and plasma parameters
f_array = np.logspace(6, 12, N_freq)
omega_array = 2 * pi * f_array
n_e = 1e19
Mi, Bfield = 2, 2  # Ion mass and magnetic field


#----- LH resonance -----------------------------------------

f_LH = mu.f_LH(Mi,Bfield,n_e) #  low hybrid
f_UH = mu.f_UH(Bfield,n_e)   # upper hybrid

f_ci = mu.f_ci(Mi,Bfield)   # ion cyclotron wave
f_ce = mu.f_ce(Bfield)      # electron cyclotron wave

angle = 88
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



Ey_over_Ez_plus = Ex_over_Ez_plus*Ey_over_Ex_plus
Ey_over_Ez_minus = Ex_over_Ez_minus*Ey_over_Ex_minus

#------------   normalised E  assuming Ex = 1   --------------------------------------------------------------------------
#                      (plus wave)           

Ex = np.ones(np.size(Ey_over_Ex_plus))


E_norm = np.sqrt( Ex**2 + Ey_over_Ex_plus**2 + Ez_over_Ex_plus**2) # E norm

                                               # NORMILIZED E , |E|=1

E_plus = np.stack((Ex/E_norm, Ey_over_Ex_plus/E_norm, Ez_over_Ex_plus/E_norm))           # NORMILIZED E representing only E direction     Ex, Ey, Ez   |E|=1


#------------   normalised E  assuming Ex = 1   --------------------------------------------------------------------------
#                      (minus wave)           


Ex = np.ones(np.size(Ey_over_Ex_minus))


E_norm = np.sqrt( Ex**2 + Ey_over_Ex_minus**2 + Ez_over_Ex_minus**2) # E norm

                                               # NORMILIZED E , |E|=1

E_minus = np.stack((Ex/E_norm, Ey_over_Ex_minus/E_norm, Ez_over_Ex_minus/E_norm))           # NORMILIZED E representing only E direction     Ex, Ey, Ez   |E|=1


#------------- Evaluation of k^2 -----------------------------------------------------------------------   
Ksq_plus  = Nstixplus*(omega_array/c)**2
Ksq_minus = Nstixminus*(omega_array/c)**2

#------------- Evaluation of k -----------------------------------------------------------------------   
K_plus = np.sqrt(   Ksq_plus  )
K_minus = np.sqrt(   Ksq_minus )



K_plus_x = K_plus * np.sin(theta_rad)
K_plus_z = K_plus * np.cos(theta_rad)
K_plus_y = np.zeros(np.size(K_plus))
K_plus_vector = np.stack((K_plus_x, K_plus_y, K_plus_z))  

K_norm = np.sqrt( K_plus_x**2 + K_plus_y**2 + K_plus_z**2) # E norm
K_plus_vector_norm = np.stack((K_plus_x/K_norm, K_plus_y/K_norm, K_plus_z/K_norm))            # NORMILIZED K representing only K direction     Ex, Ey, Ez   |E|=1





K_minus_x = K_minus * np.sin(theta_rad)
K_minus_z = K_minus * np.cos(theta_rad)
K_minus_y = np.zeros(np.size(K_minus))
K_minus_vector = np.stack((K_minus_x, K_minus_y, K_minus_z))  

K_norm = np.sqrt( K_minus_x**2 + K_minus_y**2 + K_minus_z**2) # E norm
K_minus_vector_norm = np.stack((K_minus_x/K_norm, K_minus_y/K_norm, K_minus_z/K_norm))            # NORMILIZED K representing only K direction     Ex, Ey, Ez   |E|=1



# Evaluate wavelength
lambd_plus =   2*pi/K_plus
lambd_minus =   2*pi/K_minus


lambd_plus_cm = lambd_plus*100 # convert m -> cm
lambd_minus_cm = lambd_minus*100 # convert m -> cm



##------------------------------    evaluation of B         -----------------------------------------------------------------------
#------------                 assuming NORMILIZED E , |E|=1   --------------------------------------------------------------------------
#                                    (plus wave)           

B_plus = np.zeros([3,N_freq])
# Example 2: Create a list of tuples (row index, column index, value)

for i in range(N_freq):
    B_plus[:,i] = -np.cross( E_plus[:,i],  K_plus_vector[:,i]  ) / omega_array[i]

B_plus_ampl = np.sqrt( B_plus[0,:]**2 + B_plus[1,:]**2 + B_plus[2,:]**2) # B norm
B_plus_norm = np.stack((B_plus[0,:]/B_plus_ampl, B_plus[1,:]/B_plus_ampl, B_plus[2,:]/B_plus_ampl))            # NORMILIZED B representing only B direction     |B|=1


#------------                 assuming NORMILIZED E , |E|=1   --------------------------------------------------------------------------
#                                    (minus wave)           

B_minus = np.zeros([3,N_freq])
# Example 2: Create a list of tuples (row index, column index, value)

for i in range(N_freq):
    B_minus[:,i] = -np.cross( E_minus[:,i],  K_minus_vector[:,i]  ) / omega_array[i]

B_minus_ampl = np.sqrt( B_minus[0,:]**2 + B_minus[1,:]**2 + B_minus[2,:]**2) # B norm
B_minus_norm = np.stack((B_minus[0,:]/B_minus_ampl, B_minus[1,:]/B_minus_ampl, B_minus[2,:]/B_minus_ampl))            # NORMILIZED B representing only B direction     |B|=1


###------------------------ Pointing vector ------------------------------------------------------------
#------------                 assuming NORMILIZED E , |E|=1   --------------------------------------------------------------------------
#                                    (plus wave)           

P_plus = np.zeros([3,N_freq])

for i in range(N_freq):
    P_plus[:,i] = np.cross( E_plus[:,i],  B_plus[:,i]  ) /mu0


P_plus_ampl = np.sqrt( P_plus[0,:]**2 + P_plus[1,:]**2 + P_plus[2,:]**2) # B norm
P_plus_norm = np.stack((P_plus[0,:]/P_plus_ampl, P_plus[1,:]/P_plus_ampl, P_plus[2,:]/P_plus_ampl))            # NORMILIZED B representing only B direction     |B|=1



#------------                 assuming NORMILIZED E , |E|=1   --------------------------------------------------------------------------
#                                    (minus wave)           

P_minus = np.zeros([3,N_freq])

for i in range(N_freq):
    P_minus[:,i] = np.cross( E_minus[:,i],  B_minus[:,i]  ) /mu0


P_minus_ampl = np.sqrt( P_minus[0,:]**2 + P_minus[1,:]**2 + P_minus[2,:]**2) # B norm
P_minus_norm = np.stack((P_minus[0,:]/P_minus_ampl, P_minus[1,:]/P_minus_ampl, P_minus[2,:]/P_minus_ampl))            # NORMILIZED B representing only B direction     |B|=1


#----------------------------- PLOT PLOT PLOT ----------------------------------------------------
#----------------------------- PLOT PLOT PLOT ----------------------------------------------------
#----------------------------- PLOT PLOT PLOT ----------------------------------------------------
#----------------------------- PLOT PLOT PLOT ----------------------------------------------------
#----------------------------- PLOT PLOT PLOT ----------------------------------------------------

Lx = 28
Ly = 7




#####################################################################################################################


#---------- Y and Z are exchanged to get consistent with the Stix and Swanson 


freq_cut = 1.3e7
freq_idx, _ = mu.find_nearest(f_array,freq_cut)



# Plot setup
fig9 = plt.figure(9, figsize=(Lx, Ly))
plt.rcParams['text.usetex'] = True
plt.tight_layout(pad=-10.0, w_pad=0.0, h_pad=-14.0)

plt.ion()


ax = fig9.add_subplot(132, projection='3d')
ax0 = fig9.add_subplot(133, projection='3d')
ax2 = fig9.add_subplot(131)




ax2.set_title(r"$N^2_{\pm}$",  fontsize=22)


plt.subplots_adjust(left=0.04, right=0.99, top=0.9, bottom=0.1, wspace=0.0)

# Set axes scale
ax2.set_xscale('log')
ax2.set_yscale('symlog')

# camera position
ax.azim = 22
ax.elev = 41
ax.dist = 1

ax0.azim = 22
ax0.elev = 41
ax0.dist = 1




# Plot N^2 values
ax2.plot(f_array, Nstixplus, '-x', color='r', markersize=5, label='plus')
ax2.plot(f_array, Nstixminus, '-x', color='b', markersize=5, label='minus')

ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.grid(True, which='major', linestyle='-', linewidth=0.8)




# Set legend location for this subplot
ax2.legend(loc='upper right')

#---------------------- plot LH resonance -------------------------
ax2.plot([f_LH, f_LH], [-1e10, 1e16], color='black')
ax2.text(f_LH , 0.5, r'$f_{LH}=%.2e Hz$' % f_LH, fontdict=font, rotation=90)


#---------------------- plot UH resonance -------------------------
ax2.plot([f_UH, f_UH], [-1e10, 1e16], color='black')
ax2.text(f_UH , 1e6, r'$f_{UH}=%.2e Hz$' % f_UH, fontdict=font, rotation=90)


#---------------------- plot UH resonance -------------------------
ax2.plot([f_ci, f_ci], [-1e10, 1e16], color='black')
ax2.text(f_ci , 0.5, r'$f_{ci}=%.2e Hz$' % f_ci, fontdict=font, rotation=90)

#---------------------- plot UH resonance -------------------------
ax2.plot([f_ce, f_ce], [-1e10, 1e16], color='blue')
ax2.text(f_ce , -1e6, r'$f_{ce}=%.2e Hz$' % f_ce, fontdict=font, rotation=90)

#------------------ marker line -------------------------------------
marker_line, = ax2.plot([freq_cut, freq_cut], [-1e10, 1e16], label='Marker', color='magenta', linestyle='-', linewidth=2)


####################################################################################################################



# Save the original locale
original_locale = locale.getlocale(locale.LC_NUMERIC)

# Set locale to use a period as the decimal separator
locale.setlocale(locale.LC_NUMERIC, 'C')


# Function to update the vector and redraw the plot
def update_vector(val):
    global vector
    # Get the new vector components from the sliders
    
   
    f_slcted = 10**(f_slider.get())
    freq_idx, _ = mu.find_nearest(f_array,f_slcted)

    # New marker
    marker_line.set_data([f_slcted, f_slcted], [-1e10, 1e16])

    
  
    
    # Replot the vector
    
    # Define the origin and the vector
    origin = np.array([0, 0, 0])
    
    # Clear the current plot
    ax.cla()
    
    
    if   not np.isnan(P_plus_norm[0,freq_idx])  :
        
        ################################## E vactor ##################################
        vector = np.array([E_plus[0,freq_idx], E_plus[2,freq_idx], E_plus[1,freq_idx]])
        ax.quiver(origin[0], origin[1], origin[2], vector[0], vector[1], vector[2], color='blue', linewidth=3,  label='E')
    
        
    
    
        ################################## B vactor ##################################
        vector = np.array([B_plus_norm[0,freq_idx], B_plus_norm[2,freq_idx], B_plus_norm[1,freq_idx]])
        ax.quiver(origin[0], origin[1], origin[2], vector[0], vector[1], vector[2], color='magenta', linewidth=3,  label='B')
    
       
        ################################## K vactor ##################################
        vector = np.array([K_plus_vector_norm[0,freq_idx], K_plus_vector_norm[2,freq_idx], K_plus_vector_norm[1,freq_idx]])
        ax.quiver(origin[0], origin[1], origin[2], vector[0], vector[1], vector[2], color='red', linewidth=3,  label='K')
    
    
        ################################## P vactor ##################################
        vector = np.array([P_plus_norm[0,freq_idx], P_plus_norm[2,freq_idx], P_plus_norm[1,freq_idx]])
        ax.quiver(origin[0], origin[1], origin[2], vector[0], vector[1], vector[2], color='green', linewidth=3,  label='P')
    


   



  
    def plot_coordinates_and_planes(ax_ref):
    ######################################### PLOT x,y,z coordinates and planes ##########################################################################################
    ###################################################################################################################################
    
       # copied from  https://knifelees3.github.io/2019/08/04/A_En_Python_PlotStudy2_ThreeOrthogonalVectors/index.html
        # To plot the projection of nar_1
        num = 20
        x_mat = np.linspace(-1, 1, num)
        z_mat = np.linspace(0, 1, num)
       
        
        # To plot the xy plane
        xx, yy = np.meshgrid(x_mat, x_mat)
        zz_xy = np.zeros((num, num))
       
        
        # To  plot the x,y,z coordinate
        line_x = np.array([x_mat, x_mat*0, x_mat*0])
        line_y = np.array([x_mat*0, x_mat, x_mat*0])
        line_z = np.array([x_mat*0, x_mat*0, x_mat])
    
    
        # The x,y,z coordinate
        ax_x = ax_ref.plot(line_x[0, :], line_x[1, :], line_x[2, :], 'k--')
        ax_y = ax_ref.plot(line_y[0, :], line_y[1, :], line_y[2, :], 'k--')
        ax_z = ax_ref.plot(line_z[0, :], line_z[1, :], line_z[2, :], 'k--')
    
        # The xy plane
        ax_ref.plot_surface(xx, yy, zz_xy, alpha=0.15, color=(0, 0, 1))
        ax_ref.plot_surface(xx, zz_xy, yy, alpha=0.15, color=(0, 0, 1))
        ax_ref.plot_surface(zz_xy, xx, yy, alpha=0.15, color=(0, 0, 1))
        
        # Set labels and limits
        ax_ref.text(0, 0, 1, r'$Y$',  fontsize=18)
        ax_ref.text(1, 0, 0, r'X',  fontsize=18)
        ax_ref.text(0, 1, 0, r'Z',  fontsize=18)
     
        ax_ref.set_xlim([1, -1])
        ax_ref.set_ylim([-1, 1])
        ax_ref.set_zlim([-1, 1])
        
   
    
   
    
   
    plot_coordinates_and_planes(ax)
   
     

    
    ax.legend()
    
    
    #------------------------------------------------------------------------------------------------------------------
  
    
    # Clear the current plot
    ax0.cla()
    
    if   not np.isnan(P_minus_norm[0,freq_idx])  : 
        
        ################################## E vactor ##################################
        vector = np.array([E_minus[0,freq_idx], E_minus[2,freq_idx], E_minus[1,freq_idx]])
        ax0.quiver(origin[0], origin[1], origin[2], vector[0], vector[1], vector[2], color='blue', linewidth=3,  label='E')
      
      
        ################################## B vactor ##################################
        vector = np.array([B_minus_norm[0,freq_idx], B_minus_norm[2,freq_idx], B_minus_norm[1,freq_idx]])
        ax0.quiver(origin[0], origin[1], origin[2], vector[0], vector[1], vector[2], color='magenta', linewidth=3,  label='B')
      
       
        ################################## K vactor ##################################
        vector = np.array([K_minus_vector_norm[0,freq_idx], K_minus_vector_norm[2,freq_idx], K_minus_vector_norm[1,freq_idx]])
        ax0.quiver(origin[0], origin[1], origin[2], vector[0], vector[1], vector[2], color='red', linewidth=3,  label='K')
      
      
        ################################## P vactor ##################################
        vector = np.array([P_minus_norm[0,freq_idx], P_minus_norm[2,freq_idx], P_minus_norm[1,freq_idx]])
        ax0.quiver(origin[0], origin[1], origin[2], vector[0], vector[1], vector[2], color='green', linewidth=3,  label='P')
  
        
  
    plot_coordinates_and_planes(ax0)

    ax.set_title(r"Plus solution (red line of $N^2$)",  fontsize=18)
    ax0.set_title(r"Minus solution (blue line of $N^2$)",   fontsize=18)
    # camera position
    # ax0.azim = 22
    # ax0.elev = 41
    # ax0.dist = 1
    
    ax0.legend()
    
    f_slider.focus_set()
    
    
    # Redraw the canvas
    canvas.draw()


# Create the main GUI window
root = tk.Tk()
root.title("3D Vector Plot with Controls")

# Embed the plot into the Tkinter window
canvas = FigureCanvasTkAgg(fig9, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
canvas.flush_events()

f_arr_log = np.log10(f_array)

# Create sliders for X, Y, and Z components
f_slider = tk.Scale(root, from_=6.0, to=12.0, tickinterval=0.5, label = "log₁₀(f) - decimal logarithm of frequency", font=("Helvetica", 16), resolution=0.001, orient=tk.HORIZONTAL,  command=update_vector)

f_slider.focus_set()

#f_slider.set(np.log10(1e7))
f_slider.set(7)
f_slider.pack(side=tk.LEFT, fill=tk.X, expand=1)




# Run the Tkinter event loop
root.mainloop()









# # Save figure
# filename = "9.pdf"
# plt.savefig(filename, bbox_inches='tight')
# plt.show()


# #####################################################################################################################
# pdfs = ['1.pdf', '2.pdf', '3.pdf', '4.pdf', '5.pdf', '6.pdf', '7.pdf', '8.pdf']




# writer = PdfWriter()

# for pdf in pdfs:
#     writer.append(pdf)

# writer.write("E_ratio_Stix_dispersion_vs_frequency_wo_jump.pdf")
# #merger.close()



























