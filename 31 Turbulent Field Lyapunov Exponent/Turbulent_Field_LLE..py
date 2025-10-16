#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  This rescales to 1920 px width, keeping aspect ratio
#  ffmpeg -r 14 -f image2 -s 1920x1080  -i "img/u_%05d.png" -vf "scale=1920:-2"   -vcodec libx264 -crf 25  -pix_fmt yuv420p  output.mp4

import pickle
import os
import io
import time as t
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import shutil
import re
from scipy.interpolate import RegularGridInterpolator
from src.field import DecayingTurbulence
from src.fluid import Fluid
from PIL import Image

def generate_initial_conditions_circle(num_points, x0, y0, R):
    """
    Generates a list of initial conditions on a circle in the phase space.
    """
    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_values = x0 + R * np.cos(thetas)
    y_values = y0 + R * np.sin(thetas)
    initial_conditions = [[x, y] for x, y in zip(x_values, y_values)]
    initial_conditions.append(initial_conditions[0])
    return initial_conditions
###############################################################################################

###############################################################################################

plt.close('all')

delay = 150  # ms per frame in GIF
points_coords = []

delta = 0.03
# Center the phase volume near a known point of interest (e.g., around x=1, y=0)
x0_val = 4.0 #5.0
y0_val = 2.0 # 0.8
Radius_initial_region = 0.03 # Keep the radius small for local linear approximation
num_points = 10 # Number of points defining the circular volume

initial_point_for_LLE = [x0_val, y0_val]

points_coords = generate_initial_conditions_circle(
    num_points=num_points, 
    x0 = x0_val, 
    y0 = y0_val, 
    R = Radius_initial_region # Keep the radius small for local linear approximation
)

# --- Benettin Setup ---
# Initialize a second point for the perturbation vector delta.
# This point will be X_0 + delta_0. We choose delta_0 to be along the x-axis initially.
# Initial distance is the Radius_initial_region.
initial_point_perturbed = [x0_val + Radius_initial_region, y0_val]

# Initial perturbation vector
delta_0 = np.array([initial_point_perturbed[0] - x0_val, initial_point_perturbed[1] - y0_val])
initial_delta_norm = np.linalg.norm(delta_0) # Store the initial norm for renormalization

trajectory_perturbed = []
trajectory_perturbed.append(initial_point_perturbed)
# --- End Benettin Setup ---


Npoints = len(points_coords)
            
trajectory_for_LLE = []
trajectory_for_LLE.append(initial_point_for_LLE)


# --- LLE Traces ---
lle_time = [0.0]
local_lyapunov_exponent = []
local_LE = []
current_sum_of_log_stretching = 0.0
# --- End LLE Traces ---

##################################################################################

# Trace u and v for LLE

# Array for central point 
u_traj_0 = []
v_traj_0 = []
  
# Array for point x0 + dx
u_traj_1 = []
v_traj_1 = []

    
# Array for point y0 + dy
u_traj_2 = []
v_traj_2 = []

# J_numeric
J_numeric = []



output_dir = 'dat'

# Get a list of all files in the directory and sort them
file_list = sorted([f for f in os.listdir(output_dir) if f.endswith('.pkl')])

# Filter files that are too large (if necessary, but keeping the original for now)
# file_list = file_list[0:200]


fig = plt.figure(figsize=(15.0, 9.6))
ax =  plt.subplot2grid((10, 10), (0, 0), colspan=5, rowspan = 7) # Top-left subplot
ax_zoom =  plt.subplot2grid((10, 10), (0, 5), colspan=5, rowspan = 7) # Top-right subplot
ax_lle = plt.subplot2grid((10, 10), (7, 1), colspan=8,  rowspan = 3) # Bottom subplot spanning both columns
 


ax.set_title("Evolution of turbulent field and tracing a set of two\n initially close fluid particles", fontsize=18)

ax.set_xlabel("X", fontsize=18)
ax.set_ylabel("Y", fontsize=18)

ax_zoom.set_xlabel("X", fontsize=18)
ax_zoom.set_ylabel("Y", fontsize=18)

xmin, xmax = 0, 2*np.pi
ymin, ymax = 0, 2*np.pi
extent = [xmin, xmax, ymin, ymax]


with open('dat/u_00002.pkl', 'rb') as f:
    # Load the data from the current file
    data = pickle.load(f)
    wh = data['wh']
    x_arr = data['x']
    y_arr = data['y']

im = ax.imshow(np.fft.irfft2(wh, axes=(-2,-1)), extent=extent, norm=None, cmap="bwr")

ax.set_title("Evolution of turbulent field and tracing a set of two\n initially close fluid particles", fontsize=18)

ax.set_xlabel("X", fontsize=18)
ax.set_ylabel("Y", fontsize=18)

ax.set_aspect('equal', adjustable='box') # Keep the aspect ratio fixed for better shape representation

ax3 = fig.add_axes(ax.get_position(), frameon=False)  # Same position
ax3.set_xticks(ax.get_xticks())
ax3.set_xticks(ax.get_xticks())

ax3.set_xlim(ax.get_xlim())
ax3.set_ylim(ax.get_ylim())

ax3.set_position(ax.get_position())


frames = []  



plt.tight_layout()                  

#file_list = file_list[0:5000] #  Temporary for test


   
# Loop through each file in the directory
for k, filename in enumerate(file_list): # k is the time step index
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'rb') as f:
            # Load the data from the current file
            data = pickle.load(f)
            
            # Add the loaded data to our list
            dt = data['time'] # Time increment
            u = data['u']
            v = data['v']
            x = data['x']
            y = data['y']
            wh = data['wh']
            it = data['it']
            
            
            # Set up interpolators for this step
            u_interp = RegularGridInterpolator((y_arr, x_arr), u, bounds_error=False, fill_value=None)
            v_interp = RegularGridInterpolator((y_arr, x_arr), v, bounds_error=False, fill_value=None)



            
            # --- MODIFIED: LLE calculation is now here, using two points ---
            def traj_interp_Lyapunov(traj_center, traj_pert, dt, u_interp, v_interp):
                """
                Advances the central trajectory and the perturbed trajectory by one time step,
                and returns the new perturbed point.
                """
                # 1. Advance the central point X_t -> X_{t+dt}
                last_point_center = np.array(traj_center[-1])
                u_val_center = u_interp(np.flip(last_point_center))[0] # u(y, x)
                v_val_center = v_interp(np.flip(last_point_center))[0] # v(y, x)
                
                x_center_new = last_point_center[0] + u_val_center * dt
                y_center_new = last_point_center[1] + v_val_center * dt
                traj_center.append([x_center_new, y_center_new])
                
                # 2. Advance the perturbed point X_t + delta_t -> X_{t+dt} + delta'_{t+dt}
                last_point_pert = np.array(traj_pert[-1])
                u_val_pert = u_interp(np.flip(last_point_pert))[0] # u(y, x)
                v_val_pert = v_interp(np.flip(last_point_pert))[0] # v(y, x)

                x_pert_new = last_point_pert[0] + u_val_pert * dt
                y_pert_new = last_point_pert[1] + v_val_pert * dt
                traj_pert.append([x_pert_new, y_pert_new])
                
                # 3. Calculate the new perturbation vector delta'_{t+dt}
                delta_prime_new = np.array([x_pert_new - x_center_new, y_pert_new - y_center_new])
                return delta_prime_new, x_center_new, y_center_new
           
            
            
            delta_prime, x_c_new, y_c_new = traj_interp_Lyapunov(trajectory_for_LLE, trajectory_perturbed, dt, u_interp, v_interp)    
           
            # --- Benettin Algorithm Step (Rescaling) ---
            # Benettin's idea: track the growth of the tangent vector
            
            # 1. Calculate the length of the stretched perturbation vector
            delta_prime_norm = np.linalg.norm(delta_prime)
            
            # 2. Update the accumulated sum of log stretching factors
            # log(stretching) = log( ||delta'_{t+dt}|| / ||delta_t|| )
            if delta_prime_norm > 1e-12: # Avoid log(0)
                 local_deform = np.log(delta_prime_norm / initial_delta_norm)
                 current_sum_of_log_stretching += local_deform
                 local_LE.append(local_deform/dt)
            
            # 3. Renormalize the perturbation vector (the key of Benettin)
            # This sets the new perturbed point to X_{t+dt} + delta_{t+dt}, where ||delta_{t+dt}|| = ||delta_0||
            # delta_{t+dt} = ( ||delta_0|| / ||delta'_{t+dt}|| ) * delta'_{t+dt}
            if delta_prime_norm > 1e-12: # Avoid division by zero
                delta_new = (initial_delta_norm / delta_prime_norm) * delta_prime
            else:
                 # If it collapsed, re-initialize delta_new with the initial norm
                delta_new = delta_0 
            
            # 4. Update the perturbed trajectory for the next step X_{t+dt} + delta_{t+dt}
            # Remove the point X_{t+dt} + delta'_{t+dt} and add X_{t+dt} + delta_{t+dt}
            trajectory_perturbed.pop() 
            trajectory_perturbed.append([x_c_new + delta_new[0], y_c_new + delta_new[1]])
            
            # 5. Update LLE array (largest Lyapunov Exponent)
            time_now = k * dt
            lle_time.append(time_now)
            # LLE = (1/t) * Sum_{i=1}^{k} log( ||delta'_i|| / ||delta_{i-1}|| )
            if time_now > 1e-12:
                current_LLE = current_sum_of_log_stretching / time_now
                local_lyapunov_exponent.append(current_LLE)
            else:
                 local_lyapunov_exponent.append(0.0)
            
            
            # --- End Benettin Algorithm Step ---
            
        
            if(it % 100 == 0): 
                plt.pause(0.1)
                im.set_data(np.fft.irfft2(wh, axes=(-2,-1)))
                fig.canvas.draw()
                fig.canvas.flush_events()
                
                print(f"  - Successfully loaded '{filename}', LLE: {local_lyapunov_exponent[-1]:.4f}")
               
           
                ax3.clear()
                ax_zoom.clear()
                ax_lle.clear()
                
                # --- Plot Trajectories and Current Shape ---
                
                # --- Plot Trajectories and Current Shape ---
                x_region_initial = []
                y_region_initial = []
                
                x_region = []
                y_region = []
                        
         
                
                
                trace_x_LLE = [p[0] for p in trajectory_for_LLE]
                trace_y_LLE = [p[1] for p in trajectory_for_LLE]
                
                # Plot central trajectory
                ax3.plot(trace_x_LLE, trace_y_LLE, '-', color='magenta', alpha=0.7,  zorder=1 )         
                
                ax3.set_xticks(ax.get_xticks())
                ax3.set_xticks(ax.get_xticks())
                ax3.set_xlim(ax.get_xlim())
                ax3.set_ylim(ax.get_ylim())
                ax3.tick_params(axis='x', labelsize=14)
                ax3.tick_params(axis='y', labelsize=14)
                ax3.set_position(ax.get_position())   
                
                
                
                # --- Plot Current Position of Points in Zoom ---
                last_point_x = trace_x_LLE[-1]
                last_point_y = trace_y_LLE[-1]
                
                last_point_pert_x = trajectory_perturbed[-1][0]
                last_point_pert_y = trajectory_perturbed[-1][1]
                
                ax_zoom.plot(last_point_x, last_point_y, 'o', color='magenta', markersize=8, label='Center Point $\mathbf{x}(t)$', zorder=4)
                ax_zoom.plot(last_point_pert_x, last_point_pert_y, 'x', color='blue', markersize=8, label='Perturbed Point $\mathbf{x}(t) + \mathbf{\delta}(t)$', zorder=4)
                # Plot the perturbation vector
                ax_zoom.plot([last_point_x, last_point_pert_x], [last_point_y, last_point_pert_y], ':', color='gray', zorder=3)
           
                ax_zoom.set_aspect('equal', adjustable='box') # Keep the aspect ratio fixed for better shape representation
    
                ax_zoom.set_xlabel("X", fontsize=18)
                ax_zoom.set_ylabel("Y", fontsize=18)
                ax_zoom.legend(loc='lower left', fontsize=12)
    
                    # --- Plot LLE Eigenvectors at the Central Point ---
                
                
                radius = 0.3
                x_min = min(last_point_x, last_point_pert_x) - radius
                x_max = max(last_point_x, last_point_pert_x) + radius
                y_min = min(last_point_y, last_point_pert_y) - radius
                y_max = max(last_point_y, last_point_pert_y) + radius     
                
                
                
            
    
    
                  # Add a small buffer/padding
                padding_factor = 0.5 # 50% padding
                
                # Calculate range
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                # Add padding to the limits, ensuring a minimum size
                min_range = 0.1
                x_range = max(x_range, min_range)
                y_range = max(y_range, min_range)
                
                x_pad = x_range * padding_factor 
                y_pad = y_range * padding_factor 
                
                x_mid = (x_min + x_max) / 2.0
                y_mid = (y_min + y_max) / 2.0
                
                ax_zoom.set_xlim(x_mid - x_range/2.0 - x_pad, x_mid + x_range/2.0 + x_pad)
                ax_zoom.set_ylim(y_mid - y_range/2.0 - y_pad, y_mid + y_range/2.0 + y_pad)
                
                ax_zoom.set_aspect('equal', adjustable='box')
    
    
                ax_zoom.grid(True)
                ax_zoom.tick_params(axis='both', which='major', labelsize=14)
    
               
                # --- Plot LLE ---
                ax_lle.plot(lle_time[1:], local_LE, '-', color='red', linewidth=2, label=r'LLE $\lambda$')
                ax_lle.tick_params(axis='both', which='major', labelsize=14)
                ax_lle.text(0.3, 0.95, r'Local Lyapunov Exponent $\lambda$' , transform=ax_lle.transAxes, 
                                    fontsize=15, color='black', 
                                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8, ec="gray"), 
                                    verticalalignment='top')
                ax_lle.axhline(0, color='black', linestyle='--', linewidth=1) # Zero line
    
                ax_lle.set_xlabel(r'Time $t$', fontsize=18)
                ax_lle.set_ylabel(r'$\lambda$', fontsize=20)
                ax_lle.grid(True)
        
                plt.tight_layout()   
      
             #   Save frame (uncomment if you need the GIF generation)
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                frames.append(Image.open(buf))
            





  



    
            # print(f"  - Successfully loaded '{filename}'")
    except (pickle.UnpicklingError, IOError) as e:
        print(f"  - Error loading file '{filename}': {e}")                     
                     
                     
                     
# The duration for the first n-1 frames is the original 'delay'
durations = [delay] * (len(frames) - 1)
# The last frame's duration is 3000 ms (3 seconds)
durations.append(1000)

  # # Save as GIF
if len(frames) > 0:
    frames[0].save('LLE evaluation.gif',
                save_all=True, append_images=frames[1:],
                duration=durations, loop=0)
