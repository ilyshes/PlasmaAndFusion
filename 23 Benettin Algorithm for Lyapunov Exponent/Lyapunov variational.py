#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 13:34:06 2025

@author: ilya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified script to calculate the largest Lyapunov exponent of the forced 
Duffing oscillator using the Benettin algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

# -- Simulation Parameters --
# Time steps
DT = 0.01   # Time step size
T_MAX = 500 # Increased T_MAX for better Lyapunov exponent convergence
STEPS = int(T_MAX / DT)
# Downsample factor for animation frames (e.g., plot every 20th point)
FRAME_SKIP = 20 # Still used if you later want to plot the trajectory
# Renormalization period (tau) for Benettin's algorithm
TAU = 1.0 
RESCALE_STEPS = int(TAU / DT) 

# Duffing oscillator constants (set to values that exhibit strong chaotic behavior)
BETA = 0.02    # Damping coefficient (beta)
ALPHA = -1.0   # Linear stiffness coefficient (alpha, negative for 'double well')
DELTA = 1.0    # Non-linear stiffness coefficient (delta, positive for 'softening spring')
GAMMA = 5.5    # Driving force amplitude (gamma)
OMEGA = 1.0    # Driving frequency (omega)

# Initial conditions for Trajectory A (Baseline)
x_0_A = 1.0     # Initial position
v_0_A = 0.0     # Initial velocity

# Initial perturbation vector (delta_0)
# A small, non-zero initial difference in the phase space (x, v)
DELTA_0_MAG = 1e-9 
delta_x_0 = DELTA_0_MAG / np.sqrt(2) 
delta_v_0 = DELTA_0_MAG / np.sqrt(2) 

# -- System of Differential Equations --
def derivatives_main(y, t):
    """The main Duffing oscillator ODEs: dy/dt = f(y, t)"""
    # Let y[0] = x (position), y[1] = d(x)/dt (velocity)
    dydt = [0, 0]
    dydt[0] = y[1]
    # d(v)/dt = -beta*v - alpha*x - delta*x^3 + gamma*cos(omega*t)
    dydt[1] = -BETA * y[1] - ALPHA * y[0] - DELTA * y[0]**3 + GAMMA * np.cos(OMEGA * t)
    return np.array(dydt)

def derivatives_variational(y, delta, t):
    """
    The variational equations: d(delta)/dt = J * delta, where J is the Jacobian
    J = [[df1/dx, df1/dv], [df2/dx, df2/dv]]
    f1 = v
    f2 = -BETA*v - ALPHA*x - DELTA*x^3 + GAMMA*cos(OMEGA*t)
    
    J = [[ 0, 1 ], 
         [ -ALPHA - 3*DELTA*x^2, -BETA ]]
    """
    x = y[0]
    
    # Components of the Jacobian J
    J_11 = 0.0
    J_12 = 1.0
    J_21 = -ALPHA - 3 * DELTA * x**2
    J_22 = -BETA
    
    # d(delta)/dt = J * delta
    d_delta_dt = [0, 0]
    d_delta_dt[0] = J_11 * delta[0] + J_12 * delta[1]
    d_delta_dt[1] = J_21 * delta[0] + J_22 * delta[1]
    return np.array(d_delta_dt)


# -- Numerical Integration (Runge-Kutta 4th order) --
def rk4_step_main(y, t, dt):
    """Performs RK4 step for the main trajectory."""
    k1 = dt * derivatives_main(y, t)
    k2 = dt * derivatives_main(y + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * derivatives_main(y + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * derivatives_main(y + k3, t + dt)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

def rk4_step_variational(y, delta, t, dt):
    """Performs RK4 step for the variational equation."""
    # Note: The Jacobian J depends on y(t), which means the variational 
    # equation is non-autonomous and coupled to the main trajectory.
    # We must use the 'k' values from the *same* y and t.
    
    k1 = dt * derivatives_variational(y, delta, t)
    k2 = dt * derivatives_variational(y + 0.5 * k1, delta + 0.5 * k1, t + 0.5 * dt) # y is not used in derivatives_variational here, so the y+..k is technically just for t, but we follow the full RK4 structure for robustness
    k3 = dt * derivatives_variational(y + 0.5 * k2, delta + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * derivatives_variational(y + k3, delta + k3, t + dt)
    return delta + (k1 + 2*k2 + 2*k3 + k4) / 6.0


# -- Main Simulation Loop with Benettin Algorithm --
def run_lyapunov_benettin(x_0, v_0, dx_0, dv_0):
    """
    Simulates Trajectory A and computes the largest Lyapunov Exponent (LLE) 
    using Benettin's algorithm.
    """
    # Initialize arrays to store the results
    x_list = np.zeros(STEPS)
    v_list = np.zeros(STEPS)
    # LLE estimates at renormalization steps
    lyapunov_list = [] 
    
    time_list = np.arange(0, T_MAX, DT)

    # Set initial state for the main trajectory and the perturbation vector
    y = np.array([x_0, v_0])
    delta = np.array([dx_0, dv_0]) # Initial perturbation vector
    
    # Normalize the initial perturbation vector
    delta_mag_initial = np.linalg.norm(delta)
    if delta_mag_initial > 0:
        delta = delta / delta_mag_initial
    else:
        # Avoid division by zero, set to a unit vector if magnitude is zero
        delta = np.array([1.0, 0.0]) # Or any unit vector

    x_list[0] = y[0]
    v_list[0] = y[1]
    
    # Benettin tracking variables
    total_time = 0.0
    sum_of_logs = 0.0
    renormalization_count = 0

    # Run the simulation
    for i in range(1, STEPS):
        t_prev = time_list[i-1]
        
        # 1. Propagate the main trajectory (y)
        y = rk4_step_main(y, t_prev, DT)
        
        # 2. Propagate the perturbation vector (delta)
        # Note: The propagation of delta depends on the current state y
        delta = rk4_step_variational(y, delta, t_prev, DT) # y is the state at t_prev + DT after rk4_step_main
        
        # Store the updated values
        x_list[i] = y[0]
        v_list[i] = y[1]
        
        total_time += DT

        # 3. Renormalization (if a period TAU has passed)
        if i % RESCALE_STEPS == 0:
            renormalization_count += 1
            
            # Calculate the current magnitude of the perturbation vector
            delta_mag_new = np.linalg.norm(delta)
            
            # Accumulate the logarithmic growth
            sum_of_logs += np.log(delta_mag_new)
            
            # Calculate the current largest Lyapunov Exponent (LLE) estimate
            current_lle = sum_of_logs / total_time
            lyapunov_list.append(current_lle)
            
            # 4. Normalize the perturbation vector (re-scale)
            delta = delta / delta_mag_new
            
            # Print the current LLE estimate
            print(f"Time: {total_time:.2f} | Renormalizations: {renormalization_count} | Current LLE: {current_lle:.6f}")
            

    return x_list, v_list, lyapunov_list[-1] if lyapunov_list else None


# -- Function to Generate GIF Frames (Kept for optional visualization) --
def create_animated_phase_portrait(x_A, v_A, lle_final, filename="duffing_trajectory.gif"):
    """
    Generates an animated GIF of the main trajectory (A) phase portrait.
    """
    num_frames = len(x_A[::FRAME_SKIP])
    print(f"\nGenerating GIF with {num_frames} frames of Trajectory A...")
    
    # --- PLOT FORMATTING ---
    # Enable LaTeX for all text in matplotlib
    plt.rcParams['text.usetex'] = True
    
    # Setup the plot once
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    
    # Determine fixed axis limits for stable animation
    x_min, x_max = x_A.min() - 0.1, x_A.max() + 0.1
    v_min, v_max = v_A.min() - 0.1, v_A.max() + 0.1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(v_min, v_max)

    # Title and Labels
    #title_text = r'Duffing Oscillator Phase Portrait $\lambda_{\text{max}} \approx ' + f'{lle_final:.4f}' + r'$'
    #ax.set_title(title_text, fontsize=18)
    ax.set_xlabel(r'$x$', fontsize=22)
    ax.set_ylabel(r'$\frac{dx}{dt}$', fontsize=24) # Using LaTeX for the derivative

    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    ax.axvline(x=0, color='gray', linestyle='--', lw=1, alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', lw=1, alpha=0.5)
    
    # Plot the full trajectory in light color as background
    ax.plot(x_A, v_A, lw=0.5, alpha=0.3, color='darkblue', label='Trajectory A (Full)')

    # Initialize the animated line and point
    line_A, = ax.plot([], [], lw=2, color='red', label='Trajectory A (Animated)') # Path A
    point_A, = ax.plot([], [], 'o', color='red', markersize=6) # Point A
    
    # Add a legend
    ax.legend(loc='upper right', fontsize=12)

    frames = []

    # Iterate through the data, skipping frames
    for i in range(0, STEPS, FRAME_SKIP):
        # Update the line for Trajectory A
        line_A.set_data(x_A[:i+1], v_A[:i+1])
        point_A.set_data(x_A[i], v_A[i])
        plt.pause(0.01)
        # Capture current figure as image in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100) # Use a specific DPI for consistency
        buf.seek(0)
        frames.append(Image.open(buf))
    
    # Save the frames as an animated GIF
    frames[0].save(filename,
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=50, # 50 ms per frame = 20 FPS
                   loop=0) # loop=0 means loop forever

    plt.close(fig) # Close the plot
    
    # Restore default settings after generation
    plt.rcParams['text.usetex'] = False


# Run the simulation and calculate the Lyapunov exponent
if __name__ == "__main__":
    print("--- Benettin Algorithm for Largest Lyapunov Exponent (LLE) ---")
    print(f"Integration time T_MAX = {T_MAX}, Renormalization period TAU = {TAU}\n")
    
    x_A, v_A, final_lle = run_lyapunov_benettin(x_0_A, v_0_A, delta_x_0, delta_v_0)
    
    if final_lle is not None:
        print(f"\nSimulation complete. Final Largest Lyapunov Exponent (LLE) estimate: {final_lle:.6f}")
    else:
        print("\nSimulation complete. LLE could not be calculated.")
        
    # Generate and save the animated GIF (optional)
    if final_lle is not None:
        create_animated_phase_portrait(x_A, v_A, final_lle)
        print("An animated phase portrait GIF has been saved as 'duffing_trajectory.gif'.")