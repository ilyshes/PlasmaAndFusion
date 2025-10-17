#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This rescales to 1920 px width, keeping aspect ratio
# ffmpeg -r 14 -f image2 -s 1920x1080 -i "img/u_%05d.png" -vf "scale=1920:-2" -vcodec libx264 -crf 25 -pix_fmt yuv420p output.mp4

import pickle
import os
import time as t
import numpy as np
import matplotlib.pyplot as plt
import shutil
import re
import io
from PIL import Image
# Removed: from scipy.interpolate import RegularGridInterpolator
# We replace this with a JIT-compiled custom function

from numba import jit # Keep Numba

# Assuming src.field and src.fluid are available in the execution environment
# from src.field import DecayingTurbulence
# from src.fluid import Fluid


###############################################################################################

# --- Topological Entropy Analysis Parameters ---
N_TRAJECTORIES = 1000  # Number of initial trajectories
delta_dist = 0.15     # Distance threshold for trajectories to be "distinguishable"
# -----------------------------------------------

# Initialize multiple trajectories and their initial coordinates
np.random.seed(42) # for reproducibility

# Initial points localized in the specified range
x0_vals = np.random.uniform(2.0, 4.0, N_TRAJECTORIES)
y0_vals = np.random.uniform(2.0, 4.0, N_TRAJECTORIES)

# Convert initial points to a NumPy array for JIT compatibility
# Shape: (N_TRAJECTORIES, 2)
initial_points = np.stack([x0_vals, y0_vals], axis=1)

# Main array to store all trajectory steps. 
# Shape: (N_TRAJECTORIES, T, 2) where T starts at 1
trajectories_array = initial_points[:, np.newaxis, :] 

# List to store the number of distinguishable trajectories at each time step
distinguishable_counts = []
times = []

# Placeholder lists for LLE plot (removed in this modification but kept structure)
lle_data = [] # Not used in this version but keeps structure


frames = []  
delay = 100  # ms per frame in GIF
##################################################################################

output_dir = 'dat'

# Get a list of all files in the directory and sort them
file_list = sorted([f for f in os.listdir(output_dir) if f.endswith('.pkl')])

# --- Figure and Axes Setup ---
fig = plt.figure(figsize=(15.0, 9.6))
# Only keep the visualization of the field, the trajectories, and the new plot
ax = plt.subplot2grid((10, 10), (0, 0), colspan=5, rowspan=7) # Top-left subplot (Field + Trajectories)
# ax_lle = plt.subplot2grid((10, 10), (7, 1), colspan=8, rowspan=3) # Bottom subplot (LLE placeholder - not used for LLE here)
ax_zoom = plt.subplot2grid((10, 10), (0, 5), colspan=5, rowspan=7) # Top-right subplot (Distinguishable Trajectories)

ax.set_title("Evolution of turbulent field and tracing \n multiple fluid particles", fontsize=18)
ax.set_xlabel("X", fontsize=18)
ax.set_ylabel("Y", fontsize=18)

ax_zoom.set_title(f"Number of Distinguishable Trajectories ($\delta > {delta_dist}$)", fontsize=18)
ax_zoom.set_xlabel("Time", fontsize=18)
ax_zoom.set_ylabel("Distinguishable Trajectories (M)", fontsize=18)
# -----------------------------


xmin, xmax = 0, 2 * np.pi
ymin, ymax = 0, 2 * np.pi
domain_L = 2 * np.pi # Domain size
extent = [xmin, xmax, ymin, ymax]

# Initial field visualization setup (using an arbitrary file)
try:
    with open('dat/u_00002.pkl', 'rb') as f:
        data = pickle.load(f)
        wh = data['wh']
    x_arr = data['x']
    y_arr = data['y']
    im = ax.imshow(np.fft.irfft2(wh, axes=(-2, -1)), extent=extent, norm=None, cmap="bwr")
except FileNotFoundError:
    print("Warning: Initial field file 'dat/u_00002.pkl' not found.")
    im = None # Handle case where file doesn't exist

# Setup a secondary axes for particle tracing overlaid on the field
ax3 = fig.add_axes(ax.get_position(), frameon=False) # Same position
ax3.set_xlim(ax.get_xlim())
ax3.set_ylim(ax.get_ylim())
ax3.set_position(ax.get_position())


##################################################################################
# --------------------- JIT-Optimized Functions --------------------------
##################################################################################

@jit(nopython=True)
def get_bilinear_interp_val_jit(point, x_arr, y_arr, field, nx, ny):
    """
    Performs fast bilinear interpolation for a single point (x, y) 
    on a regular grid defined by x_arr, y_arr and field (u or v).
    Assumes periodic boundaries for index lookups.
    Field shape must be (Y, X).
    """
    x, y = point[0], point[1]
    
    # Calculate uniform spacing (assuming x_arr[0]=0, y_arr[0]=0)
    # This assumes the first element is 0 and the grid is uniform.
    dx = x_arr[1] - x_arr[0]
    dy = y_arr[1] - y_arr[0]

    # Calculate fractional index, then floor for i and j
    # Field has shape (ny, nx), so y is row index (j), x is column index (i)
    i_float = x / dx
    j_float = y / dy
    
    # Indices i and j are the lower-left corner
    i = int(i_float) % nx
    j = int(j_float) % ny

    # Fractional part
    fx = i_float - i
    fy = j_float - j

    # Indices for the four corners (with periodic wrap-around)
    i1 = (i + 1) % nx
    j1 = (j + 1) % ny
    
    # Get the four corner values. Field shape is (Y, X) -> (row, col)
    q00 = field[j, i]
    q10 = field[j, i1]
    q01 = field[j1, i]
    q11 = field[j1, i1]

    # Bilinear Interpolation Formula:
    r1 = q00 * (1.0 - fx) + q10 * fx
    r2 = q01 * (1.0 - fx) + q11 * fx
    
    val = r1 * (1.0 - fy) + r2 * fy
    
    return val


@jit(nopython=True)
def update_trajectories_jit(current_points, dt, u, v, x_arr, y_arr, domain_L):
    """
    Updates all trajectory points in parallel using Euler step and JIT-compiled
    interpolation with periodic boundary conditions.
    """
    N = current_points.shape[0]
    ny, nx = u.shape # Grid dimensions (Y, X)
    new_points = np.empty_like(current_points)

    for i in range(N):
        x0 = current_points[i, 0]
        y0 = current_points[i, 1]
        
        # Pass the point, grid arrays, field and dimensions to the interpolator
        u_val = get_bilinear_interp_val_jit(current_points[i], x_arr, y_arr, u, nx, ny)
        v_val = get_bilinear_interp_val_jit(current_points[i], x_arr, y_arr, v, nx, ny)

        # Euler update
        x0 += u_val * dt
        y0 += v_val * dt

        # Apply periodic boundary conditions for the domain [0, domain_L]
        x0 = x0 % domain_L
        y0 = y0 % domain_L
        
        new_points[i, 0] = x0
        new_points[i, 1] = y0

    return new_points
    
# Keep the distinguishable counting function as it was already JIT-optimized
@jit(nopython=True)
def count_distinguishable_2d_euclidean_jit(segments, eps):
    """
    Counts distinguishable 2D trajectory segments using the maximum Euclidean distance
    over the trajectory length (time), optimized with Numba JIT.
    
    Segments shape: (N, T, 2) -> (N, number of time steps, coordinates)
    """
    n = segments.shape[0]
    T = segments.shape[1]
    eps_sq = eps**2 # Pre-calculate squared threshold

    # Use explicit numpy boolean type, preferred by Numba
    used = np.zeros(n, dtype=np.bool_) 
    count = 0

    # i is the reference segment
    for i in range(n):
        if used[i]:
            continue
            
        count += 1
        
        # Mark segment i as used (it covers itself)
        used[i] = True 
        
        # j is the segment being compared to i
        for j in range(n):
            if used[j]:
                continue
            
            # --- Distance Calculation (Inner Loops) ---
            max_dist_sq = 0.0
            
            # t is the time step
            for t in range(T):
                # Calculate squared difference for x (index 0) and y (index 1)
                dx = segments[i, t, 0] - segments[j, t, 0]
                dy = segments[i, t, 1] - segments[j, t, 1]
                
                # Squared Euclidean distance at time t
                dist_sq = dx**2 + dy**2
                
                # Update the maximum squared distance over time
                if dist_sq > max_dist_sq:
                    max_dist_sq = dist_sq 
                
                # Early Exit Optimization
                if max_dist_sq > eps_sq:
                    break
            
            # Check if the overall maximum squared distance is within eps
            if max_dist_sq <= eps_sq:
                # If within distance, mark segment j as covered/used
                used[j] = True
                
    return count

##################################################################################
# --------------------- Main Loop (Using JIT) --------------------------
##################################################################################


file_list = file_list[0:5000]

# Loop through each file in the directory
for filename in file_list:
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, 'rb') as f:
            # Load the data from the current file
            data = pickle.load(f)

            # Add the loaded data to our lists
            dt = data['time']
            u = data['u'] # 2D array (Y, X)
            v = data['v'] # 2D array (Y, X)
            it = data['it']
            current_time = it * dt
            times.append(current_time)

            # --- JIT Trajectory Update ---
            # Get the last point array (N_TRAJECTORIES, 2)
            last_points = trajectories_array[:, -1, :] 
            
            # Call the optimized function
            new_points_array = update_trajectories_jit(
                last_points, dt, u, v, x_arr, y_arr, domain_L
            )
            
            # Append the new points to the main array (N, T+1, 2)
            trajectories_array = np.concatenate(
                (trajectories_array, new_points_array[:, np.newaxis, :]), axis=1
            )
            
            # --- Calculate and store distinguishable count ---
            # Use the JIT-optimized function with the new array
            dist_count = count_distinguishable_2d_euclidean_jit(trajectories_array, delta_dist)
            distinguishable_counts.append(dist_count)


            if (it % 100 == 0):
                plt.pause(0.1)
                
                # Update Field Visualization
                if im:
                    wh = data['wh']
                    im.set_data(np.fft.irfft2(wh, axes=(-2, -1)))
                
                # Update Trajectory Plot (ax3) - **REWRITTEN FOR NUMPY ARRAY**
                ax3.clear()
                
                # Get the points for plotting: (T_steps, N_TRAJECTORIES, 2)
                # Note: We plot trajectory segment by segment to avoid large
                # memory allocations, but NumPy array access is fast.
                
                # Plot all trajectories up to the current time step
                # Transpose array to make plotting easier: (N_TRAJECTORIES, T, 2) -> (T, N_TRAJECTORIES, 2)
                traces = trajectories_array.transpose(1, 0, 2)
                
                # Plot the full trace for all N trajectories
                for i in range(N_TRAJECTORIES):
                    trace_x = traces[:, i, 0]
                    trace_y = traces[:, i, 1]
                    # Plot the full trace
                    ax3.plot(trace_x, trace_y, lw=1, color='gray', alpha=0.5, zorder=0)
                
                # Plot the current point (last step)
                current_x = new_points_array[:, 0]
                current_y = new_points_array[:, 1]
                ax3.plot(current_x, current_y, 'o', ms=5, color='red', zorder=2)
                
                ax3.set_xlim(ax.get_xlim())
                ax3.set_ylim(ax.get_ylim())
                ax3.set_position(ax.get_position())
                ax3.tick_params(axis='both', which='both', length=0) # Hide ticks
                ax3.set_xticks([])
                ax3.set_yticks([])
                
                # Update Distinguishable Trajectories Plot (ax_zoom)
                ax_zoom.clear()
                ax_zoom.plot(times, distinguishable_counts, 'b-', lw=2)
                
                ax_zoom.set_title(f"Number of Distinguishable Trajectories ($\delta > {delta_dist}$)", fontsize=18)
                ax_zoom.set_xlabel("Time", fontsize=18)
                ax_zoom.set_ylabel("Distinguishable Trajectories (M)", fontsize=18)
                ax_zoom.grid(True)
                ax_zoom.tick_params(axis='x', labelsize=14)
                ax_zoom.tick_params(axis='y', labelsize=14)
                
                ax_zoom.set_ylim(100,300)
                ax_zoom.set_xlim(-0.001,0.36)
                
                plt.tight_layout()
                
                # Redraw figure

                print(f"  - Successfully loaded and processed '{filename}'. Distinguishable count: {dist_count}")


             #   Save frame 
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                frames.append(Image.open(buf))



    except (pickle.UnpicklingError, IOError) as e:
        print(f"  - Error loading file '{filename}': {e}")
        
plt.show() # Keep the window open after the loop finishes


# The duration for the first n-1 frames is the original 'delay'
durations = [delay] * (len(frames) - 1)
# The last frame's duration is 1000 ms (1 second)
if len(durations) > 0:
    durations.append(1000)

  # # Save as GIF
if len(frames) > 0:
    frames[0].save('Distinguishable_trajectories.gif', # Changed filename
                save_all=True, append_images=frames[1:],
                duration=durations, loop=0)



