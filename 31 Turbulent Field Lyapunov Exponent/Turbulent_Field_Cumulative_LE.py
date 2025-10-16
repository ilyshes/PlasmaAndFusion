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

# Removed: generate_initial_conditions_circle is no longer needed

###############################################################################################

# --- NEW: Structure to hold data for multiple trajectories ---
class TrajectoryData:
    def __init__(self, x0, y0, initial_pert_norm):
        self.initial_point_for_LLE = [x0, y0]
        self.trajectory_for_LLE = [[x0, y0]] # Central trajectory
        self.initial_point_perturbed = [x0 + initial_pert_norm, y0]
        self.trajectory_perturbed = [[x0 + initial_pert_norm, y0]] # Perturbed trajectory
        self.initial_delta_norm = initial_pert_norm
        
        # LLE Traces
        self.lle_time = [0.0]
        self.cumulative_lyapunov_exponent = [0.0] # LLE(t) = (1/t) * Sum(...)  cumulative running average of Lyapunov exponents
        self.local_LE = [] # log( ||delta'_i|| / ||delta_{i-1}|| ) / dt
        self.current_sum_of_log_stretching = 0.0

    def update_LLE(self, delta_prime, x_c_new, y_c_new, dt, k):
        """Updates the LLE calculation and performs Benettin renormalization."""
        
        # 1. Calculate the length of the stretched perturbation vector
        delta_prime_norm = np.linalg.norm(delta_prime)
        
        # 2. Update the accumulated sum of log stretching factors
        if delta_prime_norm > 1e-12: # Avoid log(0)
             local_deform = np.log(delta_prime_norm / self.initial_delta_norm)
             self.current_sum_of_log_stretching += local_deform
             self.local_LE.append(local_deform/dt) # Local deformation rate
        else:
            # If perturbation collapses, record a zero deformation rate
             self.local_LE.append(0.0) 
        
        # 3. Renormalize the perturbation vector (the key of Benettin)
        # delta_{t+dt} = ( ||delta_0|| / ||delta'_{t+dt}|| ) * delta'_{t+dt}
        if delta_prime_norm > 1e-12: # Avoid division by zero
            delta_new = (self.initial_delta_norm / delta_prime_norm) * delta_prime
        else:
             # If it collapsed, re-initialize delta_new with the initial norm along x-axis
             delta_new = np.array([self.initial_delta_norm, 0.0]) 
        
        # 4. Update the perturbed trajectory for the next step X_{t+dt} + delta_{t+dt}
        # Remove the point X_{t+dt} + delta'_{t+dt} (which is the last element in trajectory_perturbed) 
        # and add X_{t+dt} + delta_{t+dt}
        self.trajectory_perturbed.pop() 
        self.trajectory_perturbed.append([x_c_new + delta_new[0], y_c_new + delta_new[1]])
        
        # 5. Update LLE array (Largest Lyapunov Exponent)
        time_now = (k + 1) * dt # k is the index, time is accumulated
        self.lle_time.append(time_now)
        # LLE = (1/t) * Sum_{i=1}^{k} log( ||delta'_i|| / ||delta_{i-1}|| )
        if time_now > 1e-12:
            current_LLE = self.current_sum_of_log_stretching / time_now
            self.cumulative_lyapunov_exponent.append(current_LLE)
        else:
             self.cumulative_lyapunov_exponent.append(0.0)
        
        return delta_new # Return the renormalized delta
        
###############################################################################################

plt.close('all')

delay = 100  # ms per frame in GIF
points_coords = []

# --- NEW: Setup for multiple random initial conditions ---
N_TRAJECTORIES = 50
Radius_initial_region = 0.03 # Initial perturbation norm (delta_0)

# Randomly distribute 20 points in x = [1.5, 4.5], y = [1.5, 4.5]
np.random.seed(42) # For reproducibility
x0_vals = np.random.uniform(1.5, 4.5 , N_TRAJECTORIES)
y0_vals = np.random.uniform(1.5, 4.5, N_TRAJECTORIES)

# List to hold the data structure for each trajectory
trajectories_data = []
for x0, y0 in zip(x0_vals, y0_vals):
    trajectories_data.append(TrajectoryData(x0, y0, Radius_initial_region))
# --- End NEW Setup ---


output_dir = 'dat'

# Get a list of all files in the directory and sort them
file_list = sorted([f for f in os.listdir(output_dir) if f.endswith('.pkl')])

fig = plt.figure(figsize=(15.0, 9.6))
# Only keep the visualization of the field and the LLE plot
ax =  plt.subplot2grid((10, 10), (0, 0), colspan=5, rowspan = 7) # Top-left subplot (Field)
# Removed: ax_zoom
ax_lle = plt.subplot2grid((10, 10), (7, 1), colspan=8,  rowspan = 3) # Bottom subplot spanning both columns
ax_zoom =  plt.subplot2grid((10, 10), (0, 5), colspan=5, rowspan = 7) # Top-right subplot
# Adjusted Title
ax.set_title(f"Evolution of turbulent field and LLE for {N_TRAJECTORIES} trajectories", fontsize=18)

ax.set_xlabel("X", fontsize=18)
ax.set_ylabel("Y", fontsize=18)

xmin, xmax = 0, 2*np.pi
ymin, ymax = 0, 2*np.pi
extent = [xmin, xmax, ymin, ymax]


with open('dat/u_00002.pkl', 'rb') as f:
    # Load the data from the current file
    data = pickle.load(f)
    wh = data['wh']
    x_arr = data['x']
    y_arr = data['y']

# Setup field visualization
im = ax.imshow(np.fft.irfft2(wh, axes=(-2,-1)), extent=extent, norm=None, cmap="bwr")

ax.set_aspect('equal', adjustable='box') # Keep the aspect ratio fixed for better shape representation

# Setup overlay axis for trajectory tracing (will only show the starting points now)
ax3 = fig.add_axes(ax.get_position(), frameon=False)  # Same position
ax3.set_xticks(ax.get_xticks())
ax3.set_xticks(ax.get_xticks())

ax3.set_xlim(ax.get_xlim())
ax3.set_ylim(ax.get_ylim())

ax3.set_position(ax.get_position())






# Plot initial points on the field visualization
for data_obj in trajectories_data:
    ax3.plot(data_obj.initial_point_for_LLE[0], data_obj.initial_point_for_LLE[1], 'o', color='yellow', markersize=5, alpha=0.8, zorder=5)


frames = []  

plt.tight_layout()                  


# Function to advance a single trajectory
def traj_interp_Lyapunov(traj_center, traj_pert, dt, u_interp, v_interp):
    """
    Advances the central trajectory and the perturbed trajectory by one time step,
    and returns the new perturbation vector delta_prime.
    """
    # 1. Advance the central point X_t -> X_{t+dt}
    last_point_center = np.array(traj_center[-1])
    # The interpolator expects (y, x) order, so we flip
    u_val_center = u_interp(np.flip(last_point_center))[0] 
    v_val_center = v_interp(np.flip(last_point_center))[0] 
    
    x_center_new = last_point_center[0] + u_val_center * dt
    y_center_new = last_point_center[1] + v_val_center * dt
    traj_center.append([x_center_new, y_center_new])
    
    # 2. Advance the perturbed point X_t + delta_t -> X_{t+dt} + delta'_{t+dt}
    last_point_pert = np.array(traj_pert[-1])
    u_val_pert = u_interp(np.flip(last_point_pert))[0] 
    v_val_pert = v_interp(np.flip(last_point_pert))[0] 

    x_pert_new = last_point_pert[0] + u_val_pert * dt
    y_pert_new = last_point_pert[1] + v_val_pert * dt
    traj_pert.append([x_pert_new, y_pert_new])
    
    # 3. Calculate the new perturbation vector delta'_{t+dt}
    delta_prime_new = np.array([x_pert_new - x_center_new, y_pert_new - y_center_new])
    return delta_prime_new, x_center_new, y_center_new


#file_list = file_list[0:19000]

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
            x_arr = data['x']
            y_arr = data['y']
            wh = data['wh']
            it = data['it']
            
            
            # Set up interpolators for this step
            u_interp = RegularGridInterpolator((y_arr, x_arr), u, bounds_error=False, fill_value=None)
            v_interp = RegularGridInterpolator((y_arr, x_arr), v, bounds_error=False, fill_value=None)


            # --- Loop over all trajectories to update them and calculate LLE ---
            for data_obj in trajectories_data:
                
                delta_prime, x_c_new, y_c_new = traj_interp_Lyapunov(
                    data_obj.trajectory_for_LLE, 
                    data_obj.trajectory_perturbed, 
                    dt, u_interp, v_interp
                )    
               
                # Update LLE and re-normalize the perturbation vector
                data_obj.update_LLE(delta_prime, x_c_new, y_c_new, dt, k)
            
            # --- End Loop over all trajectories ---
            
        
            if(it % 300 == 0):
                plt.pause(0.1)
                im.set_data(np.fft.irfft2(wh, axes=(-2,-1)))
                fig.canvas.draw()
                fig.canvas.flush_events()
                
                # Report LLE for the first trajectory only for console output
                print(f"  - Successfully loaded '{filename}', LLE (Traj 0): {trajectories_data[0].cumulative_lyapunov_exponent[-1]:.4f}")
               
           
                ax3.clear()
                ax_zoom.clear()
                ax_lle.clear()
                
                # --- Plot Trajectories on Field (Ax) ---
                for data_obj in trajectories_data:
                    # Plot the full trace for all
                    trace_x_LLE = [p[0] for p in data_obj.trajectory_for_LLE]
                    trace_y_LLE = [p[1] for p in data_obj.trajectory_for_LLE]
                    
                    # Plot the whole trace
                    ax3.plot(trace_x_LLE, trace_y_LLE, '-', alpha=0.8, zorder=1) # Reduced alpha for many traces
                    # Plot the current point
                    ax3.plot(trace_x_LLE[-1], trace_y_LLE[-1], 'o', color='magenta', markersize=3, alpha=0.8, zorder=2)
                    # Plot the initial point (re-adding what was cleared)
                    ax3.plot(data_obj.initial_point_for_LLE[0], data_obj.initial_point_for_LLE[1], 'o', color='yellow', markersize=5, alpha=0.8, zorder=5)

                
                ax3.set_xticks(ax.get_xticks())
                ax3.set_xticks(ax.get_xticks())
                ax3.set_xlim(ax.get_xlim())
                ax3.set_ylim(ax.get_ylim())
                ax3.tick_params(axis='x', labelsize=14)
                ax3.tick_params(axis='y', labelsize=14)
                ax3.set_position(ax.get_position())   
                
                
                # Removed: ax_zoom plotting

               
                # --- Plot LLE ---
                # Plot local deformation (local_LE) for all trajectories
                for i, data_obj in enumerate(trajectories_data):
                    # Local deformation uses time from the second element onward, corresponding to len(local_LE)
                    time_data = data_obj.lle_time[1:] 
                    #ax_lle.plot(time_data, data_obj.local_LE, '-', alpha=0.8, linewidth=1, label=f'Traj {i}' if i == 0 else "")
                    ax_lle.plot(time_data, data_obj.cumulative_lyapunov_exponent[1:], '-', alpha=0.8, linewidth=1, label=f'Traj {i}' if i == 0 else "")
                ax_lle.tick_params(axis='both', which='major', labelsize=14)
                # Adjusted text to reflect the plot contents
                ax_lle.text(0.5, 0.95, r'Cumulative Lyapunov Exponent Estimate $\lambda$ for 50 Trajectories' , 
                            transform=ax_lle.transAxes, 
                            fontsize=15, color='black', 
                            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8, ec="gray"), 
                            verticalalignment='top', horizontalalignment='center')
                ax_lle.axhline(0, color='black', linestyle='--', linewidth=1) # Zero line
    
                ax_lle.set_xlabel(r'Time $t$', fontsize=18)
                ax_lle.set_ylabel(r'$\lambda$', fontsize=20)
                ax_lle.grid(True)
        
        
        
                
        
                # --- Plot Distribution of Cumulative LLE (Ax_zoom) ---
                last_cumulative_le_values = [   traj.cumulative_lyapunov_exponent[-1]    for traj in trajectories_data ]

                # Plot the histogram (distribution function) of the last cumulative LLE values
                ax_zoom.hist(last_cumulative_le_values, bins=10, density=True, color='skyblue', edgecolor='black', alpha=0.7)
                
                # Add mean and standard deviation lines
                mean_le = np.mean(last_cumulative_le_values)
                std_le = np.std(last_cumulative_le_values)
                ax_zoom.axvline(mean_le, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_le:.3f}')
                ax_zoom.axvline(mean_le + std_le, color='gray', linestyle='--', linewidth=1)
                ax_zoom.axvline(mean_le - std_le, color='gray', linestyle='--', linewidth=1)
                
                # Title and labels for the distribution plot
                ax_zoom.set_title("Distribution of Cumulative LLEs", fontsize=15)
                ax_zoom.set_xlabel(r'Cumulative LLE $\lambda$', fontsize=18)
                ax_zoom.set_ylabel('Probability Density', fontsize=18)
                ax_zoom.legend()
                ax_zoom.grid(True, linestyle=':', alpha=0.6)
                # Removed: ax_zoom plotting (original content)
        
        
        
        
                plt.tight_layout()   
      
             #   Save frame 
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                frames.append(Image.open(buf))
            
    except (pickle.UnpicklingError, IOError) as e:
        print(f"  - Error loading file '{filename}': {e}")                     
                     
  
 
  
                     
# The duration for the first n-1 frames is the original 'delay'
durations = [delay] * (len(frames) - 1)
# The last frame's duration is 1000 ms (1 second)
if len(durations) > 0:
    durations.append(1000)

  # # Save as GIF
if len(frames) > 0:
    frames[0].save('Turbulent_Field_Cumulative_LE.gif', # Changed filename
                save_all=True, append_images=frames[1:],
                duration=durations, loop=0)
