#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified script to trace two very close trajectories of the forced Duffing oscillator
to illustrate chaotic sensitivity to initial conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

# -- Simulation Parameters --
# Time steps
DT = 0.01   # Time step size
T_MAX = 50  # Reduced T_MAX for a manageable GIF size and time
STEPS = int(T_MAX / DT)
# Downsample factor for animation frames (e.g., plot every 20th point)
FRAME_SKIP = 60

# Duffing oscillator constants (set to values that exhibit strong chaotic behavior)
BETA = 0.02    # Damping coefficient (beta)
ALPHA = -1.0   # Linear stiffness coefficient (alpha, negative for 'double well')
DELTA = 1.0    # Non-linear stiffness coefficient (delta, positive for 'softening spring')
GAMMA = 5.5    # Driving force amplitude (gamma)
OMEGA = 1.0    # Driving frequency (omega)

# Initial conditions for Trajectory 1 (Baseline)
x_0_A = 1.0     # Initial position
v_0_A = 0.0     # Initial velocity

# Initial conditions for Trajectory 2 (Slightly Perturbed)
# Introducing a tiny difference in the initial position (e.g., 1 part in a million)
PERTURBATION = 1e-2
x_0_B = x_0_A + PERTURBATION
v_0_B = v_0_A 

delay = 100  # ms per frame in GIF

# -- System of Differential Equations --
def derivatives(y, t):
    # Let y[0] = x (position)
    # Let y[1] = d(x)/dt (velocity)
    dydt = [0, 0]
    dydt[0] = y[1]
    dydt[1] = -BETA * y[1] - ALPHA * y[0] - DELTA * y[0]**3 + GAMMA * np.cos(OMEGA * t)
    return np.array(dydt)

# -- Numerical Integration (Runge-Kutta 4th order) --
def rk4_step(y, t, dt):
    """Performs a single Runge-Kutta 4th order integration step."""
    k1 = dt * derivatives(y, t)
    k2 = dt * derivatives(y + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * derivatives(y + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * derivatives(y + k3, t + dt)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

# -- Main Simulation Loop --
def run_simulation(x_0, v_0):
    """Simulates the oscillator's motion and stores data for plotting."""
    # Initialize arrays to store the results
    x_list = np.zeros(STEPS)
    v_list = np.zeros(STEPS)
    time_list = np.arange(0, T_MAX, DT)

    # Set initial state
    y = np.array([x_0, v_0])
    x_list[0] = y[0]
    v_list[0] = y[1]

    # Run the simulation
    for i in range(1, STEPS):
        # Update the state using the RK4 method
        y = rk4_step(y, time_list[i-1], DT)

        # Store the updated values
        x_list[i] = y[0]
        v_list[i] = y[1]

    return x_list, v_list

# -- Function to Generate GIF Frames --
def create_animated_phase_portrait(x_A, v_A, x_B, v_B, perturbation, filename="duffing_chaos_animation.gif"):
    """
    Generates an animated GIF of two close phase portrait trajectories.
    """
    num_frames = len(x_A[::FRAME_SKIP])
    print(f"Generating GIF with {num_frames} frames...")
    
    # --- PLOT FORMATTING ---
    # Enable LaTeX for all text in matplotlib
    plt.rcParams['text.usetex'] = True
    
    # Setup the plot once
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    fig.suptitle('Forced Duffing Oscillator - The Simple Testbed to Understand Chaotic Behaviour' +  '\n' + r'Duffing oscillator equations: $\ddot{x} + \delta\dot{x} + \beta x + \alpha x^3 = \gamma \cos(\omega t)$',  fontsize=18)

    # Combine all data to determine fixed axis limits for stable animation
    x_all = np.concatenate((x_A, x_B))
    v_all = np.concatenate((v_A, v_B))
    x_min, x_max = x_all.min() - 0.1, x_all.max() + 0.1
    v_min, v_max = v_all.min() - 0.1, v_all.max() + 0.1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(v_min, v_max)

    # Title and Labels
    title_text = r'Divergence of two trajectories - Sensitivity to Initial Conditions: $\Delta x_0 = ' + f'{perturbation:e}' + r'$'
    ax.set_title(title_text, fontsize=18)
    ax.set_xlabel(r'$x$', fontsize=22)
    ax.set_ylabel(r'$\frac{dx}{dt}$', fontsize=24) # Using LaTeX for the derivative

    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    ax.axvline(x=0, color='gray', linestyle='--', lw=1, alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', lw=1, alpha=0.5)
    
    # Plot the full trajectories in light colors as background
    #ax.plot(x_A, v_A, lw=0.5, alpha=0.3, color='blue')
    #ax.plot(x_B, v_B, lw=0.5, alpha=0.3, color='red')

    # Initialize the animated lines and points
    line_A, = ax.plot([], [], lw=1, color='darkblue', alpha=0.3, label='Trajectory A (Animated)') # Path A
    point_A, = ax.plot([], [], 'o', color='darkblue', markersize=10) # Point A
    
    line_B, = ax.plot([], [], lw=1, color='red', alpha=0.3, label='Trajectory B (Animated)') # Path B
    point_B, = ax.plot([], [], 'o', color='red', markersize=10) # Point B ('s' for square marker)
    
    # Add a legend for clarity (only animated paths/points will be tracked)
    ax.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    frames = []

    # Iterate through the data, skipping frames
    for i in range(0, STEPS, FRAME_SKIP):
        # Update the line for Trajectory A
        line_A.set_data(x_A[:i+1], v_A[:i+1])
        point_A.set_data(x_A[i], v_A[i])
        
        # Update the line for Trajectory B
        line_B.set_data(x_B[:i+1], v_B[:i+1])
        point_B.set_data(x_B[i], v_B[i])

        plt.pause(0.01)
        # Capture current figure as image in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100) # Use a specific DPI for consistency
        buf.seek(0)
        frames.append(Image.open(buf))
    
    
    
    # The duration for the first n-1 frames is the original 'delay'
    durations = [delay] * (len(frames) - 1)
    # The last frame's duration is 3000 ms (3 seconds)
    durations.append(2000)
    
    # Save the frames as an animated GIF
    frames[0].save(filename,
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=durations, # 50 ms per frame = 20 FPS
                   loop=0) # loop=0 means loop forever

    plt.close(fig) # Close the plot
    
    # Restore default settings after generation
    plt.rcParams['text.usetex'] = False

# Run the simulation and plot the results
if __name__ == "__main__":
    print("Simulating the forced Duffing oscillator (Trajectory A)...")
    x_A, v_A = run_simulation(x_0_A, v_0_A)
    
    print("Simulating the forced Duffing oscillator (Trajectory B)...")
    x_B, v_B = run_simulation(x_0_B, v_0_B)
    
    # Generate and save the animated GIF
    create_animated_phase_portrait(x_A, v_A, x_B, v_B, PERTURBATION)
    
    print("Simulation complete.")
    print("An animated phase portrait GIF illustrating chaos has been saved as 'duffing_chaos_animation.gif'.")