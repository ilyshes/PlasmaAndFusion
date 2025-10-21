#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified script to simulate, animate, and analyze the dispersion of a large set 
of initial trajectories of the forced Duffing oscillator.

The analysis uses the provided function to track the number of distinguishable 
trajectories (N_eps(t)) to estimate topological entropy.
"""

import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from numba import jit # Keep Numba

plt.close('all')
# -- Simulation Parameters --
# Time steps
DT = 0.01   # Time step size
T_MAX = 5  # Total simulation time
STEPS = int(T_MAX / DT)
# Downsample factor for animation frames 
FRAME_SKIP = 10 # Skip 60 steps (0.6s time difference) per frame

# Duffing oscillator constants
BETA = 0.02
ALPHA = -1.0
DELTA = 1.0
GAMMA = 5.5
OMEGA = 1.0

# --- Parameters for Trajectory Growth Analysis ---
N_TRAJECTORIES = 1000 # Number of initial trajectories

# Initial points localized in the specified range for x (position) and v (velocity)
X0_MIN, X0_MAX = -3.0, 1.0
V0_MIN, V0_MAX = -5.0, 5.0

# Define the distinguishing radius (epsilon)
EPSILON = 1

# Initialize N_TRAJECTORIES initial conditions
np.random.seed(42) # For reproducibility
x0_vals = np.random.uniform(X0_MIN, X0_MAX, N_TRAJECTORIES)
v0_vals = np.random.uniform(V0_MIN, V0_MAX, N_TRAJECTORIES)

delay = 50  # ms per frame in GIF

# -- System of Differential Equations --
def derivatives(y, t):
    """Computes the derivatives for the Duffing oscillator (vectorized)."""
    x, v = y[0], y[1]
    dvdt = -BETA * v - ALPHA * x - DELTA * x**3 + GAMMA * np.cos(OMEGA * t)
    dxdt = v
    return np.array([dxdt, dvdt])

# -- Numerical Integration (Runge-Kutta 4th order) --
def rk4_step(y, t, dt):
    """Performs a single Runge-Kutta 4th order integration step (vectorized)."""
    k1 = dt * derivatives(y, t)
    k2 = dt * derivatives(y + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * derivatives(y + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * derivatives(y + k3, t + dt)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

# -- Main Simulation Loop --
def run_simulation(x0_array, v0_array):
    """Simulates the motion of all initial trajectories."""
    x_history = np.zeros((STEPS, N_TRAJECTORIES))
    v_history = np.zeros((STEPS, N_TRAJECTORIES))
    time_list = np.arange(0, T_MAX, DT)

    y = np.array([x0_array, v0_array])
    x_history[0] = y[0]
    v_history[0] = y[1]

    print(f"Starting simulation of {N_TRAJECTORIES} trajectories...")
    
    for i in range(1, STEPS):
        y = rk4_step(y, time_list[i-1], DT)
        x_history[i] = y[0]
        v_history[i] = y[1]

    print("Simulation complete.")
    # Return time-series data
    return x_history, v_history, time_list

# -- User-Provided Analysis Function --
def count_distinguishable_2d_euclidean(segments, eps):
    """
    Counts distinguishable 2D trajectory segments using the maximum Euclidean distance
    over the trajectory length (time).
    
    Segments shape: (N, T, 2), where N is trajectories, T is time steps, 2 is (x, v)
    """
    n = len(segments)
    used = np.zeros(n, dtype=bool)
    count = 0
    trajector = np.array(segments) # (N, T, 2)
    
    for i in range(n):
        if used[i]:
            continue
        count += 1
        
        # Calculate the difference vector: shape (N, T, 2)
        # diff[j, k, l] = trajector[j, k, l] - trajector[i, k, l]
        diff = trajector - trajector[i]
        
        # Calculate squared Euclidean distance at each time step: shape (N, T)
        # sq_euc_dist[j, k] = (diff[j, k, 0])^2 + (diff[j, k, 1])^2
        sq_euc_dist = np.sum(diff**2, axis=2)
        
        # Calculate Euclidean distance: shape (N, T)
        euc_dist = np.sqrt(sq_euc_dist)
        
        # Take the maximum Euclidean distance over time (axis=1) for each trajectory j.
        # The resulting dists array has shape (N,)
        dists = np.max(euc_dist, axis=1)
        
        # Identify all segments within the eps distance
        within = dists <= eps
        
        # Mark the current segment and all segments within eps as 'used'
        used = used | within
        
    return count



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



# -- New Analysis Wrapper Function --
def calculate_distinguishable_trajectories(x_hist, v_hist, eps):
    """
    Calculates the number of distinguishable trajectories N_eps(t) as a function of time.
    """
    
    # 1. Reshape history for the analysis function: (STEPS, N, 2)
    history_steps_N_2 = np.stack((x_hist, v_hist), axis=2)
    
    # 2. Transpose to the required (N, STEPS, 2) shape
    history_N_steps_2 = np.transpose(history_steps_N_2, (1, 0, 2))

    n_distinguishable = np.zeros(STEPS)

    # 3. Iterate through increasing segment lengths (time)
    for i in range(STEPS):
        # Current segment: all N trajectories, up to time step i+1
        current_segments = history_N_steps_2[:, :i+1, :]
        print('step ' + str(i))
        # Calculate the count for the current segment length
        # Note: Calculation is skipped for the first step (t=0) as the segment length is 1, 
        # but the function will work.
        n_distinguishable[i] = count_distinguishable_2d_euclidean_jit(current_segments, eps)
        
        # Optimization: If all N trajectories are distinguishable, no need to continue
        if n_distinguishable[i] == N_TRAJECTORIES:
            # Fill the rest of the array with the maximum count
            n_distinguishable[i:] = N_TRAJECTORIES
            break
            
    return n_distinguishable

# -- Function to Generate GIF Frames (Modified for two panels) --
def create_animated_phase_portrait(x_hist, v_hist, time_list, filename="duffing_topological_entropy_analysis.gif"):
    """
    Generates an animated GIF with two panels: phase portrait (left) and 
    topological entropy analysis (right).
    """
    
    num_frames = len(x_hist[::FRAME_SKIP])
    print(f"Generating GIF with {num_frames} frames...")
    
    # --- PLOT FORMATTING ---
    plt.rcParams['text.usetex'] = True
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(r'Forced Duffing Oscillator: Trajectory Dispersion and Topological Entropy ($\epsilon = {}$)'.format(EPSILON), fontsize=18)

    # --- LEFT PANEL: Phase Portrait Animation (ax1) ---
    ax1.set_title(r'Phase Portrait (Trajectory Dispersion)', fontsize=16)
    ax1.set_xlabel(r'$x$', fontsize=18)
    ax1.set_ylabel(r'$\frac{dx}{dt}$', fontsize=20)
    
    # Set fixed limits for stability
    x_min, x_max = x_hist.min() - 0.5, x_hist.max() + 0.5
    v_min, v_max = v_hist.min() - 0.5, v_hist.max() + 0.5
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(v_min, v_max)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Initialize the animated scatter plot
    scatter = ax1.plot([], [], 'o', color='blue', alpha=1.0, lw=0.1,  label=f'$N={N_TRAJECTORIES}$ Trajectories')
    
    # Initialize the time display text
    time_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, fontsize=14, 
                         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
    
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # --- RIGHT PANEL: Topological Entropy Analysis (ax2) ---
    ax2.set_title(r'Number of Distinguishable Trajectories $N_{\epsilon}(t)$', fontsize=16)
    ax2.set_xlabel(r'Time $t$', fontsize=18)
    # The slope is approximately the topological entropy: $h_{top} \approx \frac{d}{dt} \ln(N_{\epsilon}(t))$
    ax2.set_ylabel(r'$N_{\epsilon}(t)$', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Calculate log of the count (N_eps(t) is always >= 1)
    #log_N_t = np.log(n_distinguishable)
    
    # Set fixed limits for the log plot
    ax2.set_xlim(-0.1, T_MAX + 0.1)
    # The max count is log(N_TRAJECTORIES)
    ax2.set_ylim(0, 1000)
    
    # Initialize the animated line for the analysis plot
    line_N, = ax2.plot([], [], color='red', lw=2)
    
    # Add a horizontal line to show the saturation level
    ax2.axhline(y=np.log(N_TRAJECTORIES), color='gray', linestyle=':', lw=1)
    
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    frames = []

    
    n_distinguishable =  []
    current_time_steps = []
    # Iterate through the data, skipping frames
    for idx in range(0, STEPS, FRAME_SKIP):
        current_time = time_list[idx]
        
        # --- Update LEFT Panel ---
        scatter[0].set_data(x_hist[idx, :], v_hist[idx, :])
        time_text.set_text(r'Time: $t = {:.2f}$'.format(current_time))
        
        
        # Calculate the number of distinguishable trajectories N_eps(t)
        x_trajectoies = x_hist[:idx+1, :]
        v_trajectoies = v_hist[:idx+1, :]

        
        merged_array = np.stack([x_trajectoies.T, v_trajectoies.T], axis=-1)
        count = count_distinguishable_2d_euclidean(merged_array, EPSILON)
        n_distinguishable.append(count)
        
        
        # --- Update RIGHT Panel ---
        # Plot only up to the current index
        current_time_steps.append(time_list[idx+1])
        #current_log_N = log_N_t[:idx+1]
        line_N.set_data(current_time_steps, n_distinguishable)
        
        
        plt.pause(0.01)
        
        # Capture current figure as image in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf))
    
    # Set durations for GIF
    durations = [delay] * (len(frames) - 1)
    durations.append(2000) 
    
    # Save the frames as an animated GIF
    frames[0].save(filename,
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=durations,
                   loop=0)

    plt.close(fig)
    plt.rcParams['text.usetex'] = False

# Run the simulation and plot the results
if __name__ == "__main__":
    x_history, v_history, time_list = run_simulation(x0_vals, v0_vals)
    

    
    # Generate and save the animated GIF with the two-panel analysis
    create_animated_phase_portrait(x_history, v_history, time_list)
    
    print("Animation complete.")
    print(f"A two-panel animated GIF illustrating trajectory dispersion and topological entropy analysis has been saved as '{'duffing_topological_entropy_analysis.gif'}'.")