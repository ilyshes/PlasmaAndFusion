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
from scipy.interpolate import RegularGridInterpolator

from numba import jit


# Assuming src.field and src.fluid are available in the execution environment
# from src.field import DecayingTurbulence
# from src.fluid import Fluid


###############################################################################################

# --- Topological Entropy Analysis Parameters ---
N_TRAJECTORIES = 1000  # Number of initial trajectories
delta_dist = 0.1     # Distance threshold for trajectories to be "distinguishable"
# -----------------------------------------------

# Initialize multiple trajectories and their initial coordinates
np.random.seed(42) # for reproducibility

# Initial points localized in the specified range
x0_vals = np.random.uniform(2.0, 4.0, N_TRAJECTORIES)
y0_vals = np.random.uniform(2.0, 4.0, N_TRAJECTORIES)

# List of lists to store the trajectory for each point
trajectories = [[[x, y]] for x, y in zip(x0_vals, y0_vals)] # trajectories[i][-1] is the last point of traj i

# List to store the number of distinguishable trajectories at each time step
distinguishable_counts = []
times = []

# Placeholder lists for LLE plot (removed in this modification but kept structure)
lle_data = [] # Not used in this version but keeps structure

##################################################################################

output_dir = 'dat'

# Get a list of all files in the directory and sort them
file_list = sorted([f for f in os.listdir(output_dir) if f.endswith('.pkl')])

# --- Figure and Axes Setup ---
fig = plt.figure(figsize=(15.0, 9.6))
# Only keep the visualization of the field, the trajectories, and the new plot
ax = plt.subplot2grid((10, 10), (0, 0), colspan=5, rowspan=7) # Top-left subplot (Field + Trajectories)
ax_lle = plt.subplot2grid((10, 10), (7, 1), colspan=8, rowspan=3) # Bottom subplot (LLE placeholder - not used for LLE here)
ax_zoom = plt.subplot2grid((10, 10), (0, 5), colspan=5, rowspan=7) # Top-right subplot (Distinguishable Trajectories)

ax.set_title("Evolution of turbulent field and tracing multiple fluid particles", fontsize=18)
ax.set_xlabel("X", fontsize=18)
ax.set_ylabel("Y", fontsize=18)

ax_zoom.set_title(f"Number of Distinguishable Trajectories ($\delta > {delta_dist}$)", fontsize=18)
ax_zoom.set_xlabel("Time", fontsize=18)
ax_zoom.set_ylabel("Distinguishable Trajectories (M)", fontsize=18)
# -----------------------------


xmin, xmax = 0, 2 * np.pi
ymin, ymax = 0, 2 * np.pi
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


def traj_interp(last_point, dt, u_interp, v_interp):
    """Calculates the next point using Euler step and periodic boundary conditions."""
    point = np.array(last_point) # x, y order
    # Interpolator expects (y, x) order, so we flip the point
    u_val = u_interp(np.flip(point))
    v_val = v_interp(np.flip(point))

    x0 = last_point[0]
    y0 = last_point[1]
    # Euler update
    x0 += u_val[0] * dt
    y0 += v_val[0] * dt

    # Apply periodic boundary conditions for the domain [0, 2*pi]
    x0 = np.mod(x0, 2 * np.pi)
    y0 = np.mod(y0, 2 * np.pi)

    return [x0, y0]

# Count distinguishable orbit segments at resolution eps
def count_distinguishable(segments, eps):
    n = len(segments)
    used = np.zeros(n, dtype=bool)
    count = 0
    trajector = np.array(segments) # (N, 2) array of (x, y)
    
    for i in range(n):
        if used[i]:
            continue
        count += 1
        # sup norm distance between segment i and all others
        # segments - segments[i] uses broadcasting
        difference = trajector - trajector[i]
        distances_each_point = [   np.sqrt( trj[:,0]**2 +  trj[:,1]**2 )  for trj in difference ]
        dists_max            =   np.max(distances_each_point, axis=1)
        within = dists_max <= eps
        used = used | within
    return count

def count_distinguishable_2d_euclidean(segments, eps):
    """
    Counts distinguishable 2D trajectory segments using the maximum Euclidean distance
    over the trajectory length (time).
    
    Segments shape: (N, T, 2)
    """
    n = len(segments)
    used = np.zeros(n, dtype=bool)
    count = 0
    trajector = np.array(segments) # (N, 2) array of (x, y)
    
    for i in range(n):
        if used[i]:
            continue
        count += 1
        
        # Calculate the difference vector: shape (N, T, 2)
        diff = trajector - trajector[i]
        
        # Calculate squared Euclidean distance at each time step: shape (N, T)
        sq_euc_dist = np.sum(diff**2, axis=2)
        
        # Calculate Euclidean distance: shape (N, T)
        euc_dist = np.sqrt(sq_euc_dist)
        
        # MODIFICATION: Take the maximum Euclidean distance over time (axis=1).
        # The resulting dists array has shape (N,)
        dists = np.max(euc_dist, axis=1)
        
        # Identify all segments within the eps distance
        within = dists <= eps
        
        # Mark the current segment and all segments within eps as 'used'
        used = used | within
        
    return count


@jit(nopython=True)
def count_distinguishable_2d_euclidean_jit(segments, eps):
    """
    Counts distinguishable 2D trajectory segments using the maximum Euclidean distance
    over the trajectory length (time), optimized with Numba JIT.
    
    Segments shape: (N, T, 2) -> (N, number of time steps, coordinates)
    
    Args:
        segments (np.ndarray): Array of 2D trajectory segments (N, T, 2).
        eps (float): The distance threshold for distinguishability.
        
    Returns:
        int: The number of distinguishable trajectories.
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
            # If j is already covered by a previous segment k, skip
            if used[j]:
                continue
            
            # Since i is already marked used, we skip i == j
            # The check `if used[j]: continue` handles the case i == j if we set used[i]=True before the inner loop
            
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
                
                # Early Exit Optimization: If the distance is already greater than eps, 
                # this pair is distinguishable, so we can stop checking further time steps (t)
                if max_dist_sq > eps_sq:
                    break
            
            # Check if the overall maximum squared distance is within eps
            if max_dist_sq <= eps_sq:
                # If within distance, mark segment j as covered/used
                used[j] = True
                
    return count




def count_distinguishable_trajectories(trajectories, delta_dist):
    """
    Counts the number of trajectory pairs (i, j) where the distance at the current
    time step is greater than delta_dist.
    """
    N = len(trajectories)
    count = 0
    current_points = np.array([t[-1] for t in trajectories]) # (N, 2) array of (x, y)

    # Check all unique pairs (i, j) where i < j
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate the distance, considering periodicity (using min distance in a torus)
            dx = np.abs(current_points[i, 0] - current_points[j, 0])
            dy = np.abs(current_points[i, 1] - current_points[j, 1])
            # Minimum distance on a torus of size 2*pi
            dx_p = min(dx, 2 * np.pi - dx)
            dy_p = min(dy, 2 * np.pi - dy)

            distance = np.sqrt(dx_p**2 + dy_p**2)

            if distance > delta_dist:
                count += 1
    return count

# Loop through each file in the directory
for filename in file_list:
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, 'rb') as f:
            # Load the data from the current file
            data = pickle.load(f)

            # Add the loaded data to our lists
            dt = data['time']
            u = data['u']
            v = data['v']
            it = data['it']
            current_time = it * dt
            times.append(current_time)

            # Set up interpolators for this step
            u_interp = RegularGridInterpolator((y_arr, x_arr), u, bounds_error=False, fill_value=None)
            v_interp = RegularGridInterpolator((y_arr, x_arr), v, bounds_error=False, fill_value=None)

            # --- Update all trajectories ---
            new_points = []
            for i, traj in enumerate(trajectories):
                new_point = traj_interp(traj[-1], dt, u_interp, v_interp)
                trajectories[i].append(new_point)

            # --- Calculate and store distinguishable count ---
            dist_count = 5 # count_distinguishable_2d_euclidean_jit(np.array(trajectories), delta_dist)
            distinguishable_counts.append(dist_count)


            if (it % 100 == 0):
                plt.pause(0.1)
                
                # Update Field Visualization
                if im:
                    wh = data['wh']
                    im.set_data(np.fft.irfft2(wh, axes=(-2, -1)))
                
                # Update Trajectory Plot (ax3)
                ax3.clear()
                
                # Plot all trajectories up to the current time step
                for traj in trajectories:
                    trace_x = [p[0] for p in traj]
                    trace_y = [p[1] for p in traj]
                    # Plot the full trace
                    ax3.plot(trace_x, trace_y, lw=1, color='gray', alpha=0.5, zorder=0)
                    # Plot the current point as a distinct marker
                    ax3.plot(trace_x[-1], trace_y[-1], 'o', ms=5, color='red', zorder=2)
                
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
                
                # Redraw figure
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(1e-9)
                print(f"  - Successfully loaded and processed '{filename}'. Distinguishable count: {dist_count}")


    except (pickle.UnpicklingError, IOError) as e:
        print(f"  - Error loading file '{filename}': {e}")
        
plt.show() # Keep the window open after the loop finishes