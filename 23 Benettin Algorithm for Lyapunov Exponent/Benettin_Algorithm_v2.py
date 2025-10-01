#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 12:59:04 2025

@author: ilya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 14:42:12 2025

@author: ilya
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from matplotlib.quiver import Quiver
from matplotlib.text import Annotation

plt.close('all')

# --- Duffing Oscillator Parameters ---
BETA = 0.02    # Damping coefficient (beta)
ALPHA = -1.0   # Linear stiffness coefficient (alpha, negative for 'double well')
DELTA = 1.0    # Non-linear stiffness coefficient (delta, positive for 'softening spring')
GAMMA = 5.5    # Driving force amplitude (gamma)
OMEGA = 1.0    # Driving frequency (omega)

# --- Simulation and Plotting Settings ---
delay = 10  # ms per frame in GIF
plt.rcParams['text.usetex'] = True

# --- Lyapunov Exponent Settings (Direct Method) ---
D0 = 1e-4      # Initial separation distance for the direct method
T_RENORM = 0 # Time interval (or number of steps) for renormalization

def model(z, t):
    """
    Defines the system of differential equations for the forced Duffing oscillator.
    """
    x, y = z
    dxdt = y
    dydt = - BETA * y - ALPHA * x - DELTA * x**3 + GAMMA * np.cos(OMEGA * t)
    return np.array([dxdt, dydt])

# ----------------------------------------------------------------------
# Numerical Integration (Runge-Kutta 4th order)
# ----------------------------------------------------------------------

def rk4_step(func, z, t, dt):
    """
    Performs one step of the 4th-order Runge-Kutta integration method for the state vector z.
    """
    k1 = dt * func(z, t)
    k2 = dt * func(z + k1 / 2, t + dt / 2)
    k3 = dt * func(z + k2 / 2, t + dt / 2)
    k4 = dt * func(z + k3, t + dt)
    z_next = z + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return z_next

def integrate_rk4(func, z0, time_points):
    """
    Integrates the system of ODEs using the RK4 method.
    """
    solution = np.zeros((len(time_points), len(z0)))
    solution[0] = np.array(z0)
    dt = time_points[1] - time_points[0] 
    z_current = np.array(z0)
    
    for i in range(len(time_points) - 1):
        t_current = time_points[i]
        z_next = rk4_step(func, z_current, t_current, dt)
        solution[i+1] = z_next
        z_current = z_next
        
    return solution

# ----------------------------------------------------------------------
# Direct Lyapunov Exponent Estimation (Two-Trajectory Method)
# ----------------------------------------------------------------------

def integrate_lyapunov_direct(z0, initial_perturbation, time_points, d0, T_renorm):
    """
    Integrates two trajectories (central z0 and perturbed zp) and performs
    periodic renormalization (Direct Method).
    
    Args:
        z0 (list/np.array): Initial state of the central trajectory.
        initial_perturbation (list/np.array): Initial unit vector for the perturbation direction.
        time_points (np.ndarray): Array of time points.
        d0 (float): The initial/reset separation distance.
        T_renorm (float): The time interval after which renormalization occurs.

    Returns:
        np.ndarray: Solution array for the central trajectory (z0).
        np.ndarray: Solution array for the separation vector (v = zp - z0).
        np.ndarray: Array of accumulated Lyapunov exponents.
    """
    
    dt = time_points[1] - time_points[0]
    num_steps = len(time_points)
    
    # Initialize arrays
    solution_z0 = np.zeros((num_steps, len(z0)))
    solution_v = np.zeros((num_steps, len(z0)))
    lyapunov_history = np.zeros(num_steps)
    
    # Initial states
    z0_current = np.array(z0)
    
    # Calculate initial perturbed state
    initial_perturbation_unit = np.array(initial_perturbation) / np.linalg.norm(initial_perturbation)
    v_current = initial_perturbation_unit * d0 # v = zp - z0
    zp_current = z0_current + v_current
    
    # Initial bookkeeping
    solution_z0[0] = z0_current
    solution_v[0] = v_current
    lyapunov_log_sum = 0.0
    
    # Renormalization interval counter
    current_time_segment = 0.0
    
    for i in range(num_steps - 1):
        t_current = time_points[i]
        t_next = time_points[i+1]
        
        # 1. Integrate both trajectories
        z0_next = rk4_step(model, z0_current, t_current, dt)
        zp_next = rk4_step(model, zp_current, t_current, dt)
        
        # 2. Update the separation vector
        v_next_integrated = zp_next - z0_next
        
        # 3. Check for Renormalization (if we crossed the T_renorm threshold)
        current_time_segment += dt
        
        if current_time_segment >= T_renorm:
            
            # Calculate the length of the integrated vector
            v_norm_integrated = np.linalg.norm(v_next_integrated)
            
            # Accumulate the logarithm of the expansion factor
            lyapunov_log_sum += np.log(v_norm_integrated / d0) # expansion factor is v_norm / d0
            
            # Rescale the vector v_next_integrated back to the initial distance d0
            v_unit = v_next_integrated / v_norm_integrated
            v_renormalized = v_unit * d0
            
            # Calculate the new perturbed state
            zp_next = z0_next + v_renormalized
            v_next_to_store = v_next_integrated # Store the integrated vector for plotting
            
            # Reset the time segment counter
            current_time_segment = 0.0 
            
        else:
            # No renormalization, simply track the integrated vector
            v_next_to_store = v_next_integrated
        
        # 4. Update states
        z0_current = z0_next
        zp_current = zp_next
        
        # 5. Store solution and LE
        solution_z0[i+1] = z0_current
        solution_v[i+1] = v_next_to_store
        
        # Calculate the current estimate of the Largest Lyapunov Exponent (lambda_1)
        elapsed_time = time_points[i+1] - time_points[0]
        if elapsed_time > 0:
            lyapunov_history[i+1] = lyapunov_log_sum / elapsed_time
        
    return solution_z0, solution_v, lyapunov_history


def solve_and_plot_system(initial_conditions_list, initial_point_for_Jacobian, time_span, initial_perturbation):
    """
    Solves the system of ODEs and generates an animated plot, including the LE
    estimation via the Direct Method.
    """
    
    gif_filename = 'Duffing_Direct_LE.gif'
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a figure with three subplots: Global, Zoom, and LE history
    fig = plt.figure(figsize=(18, 9))
    ax_global = fig.add_subplot(1, 3, 1)
    ax_zoom = fig.add_subplot(1, 3, 2)
    ax_le = fig.add_subplot(1, 3, 3)

    plot_limit = 3.2

 
    # --- Ax1: Global Phase Space View ---
    ax_global.set_title(r'Global Phase Space View and Trajectories', fontsize=16)
    ax_global.set_xlabel(r'Position ($x$)', fontsize=14)
    ax_global.set_ylabel(r'Velocity ($\dot{x}$)', fontsize=14)
    ax_global.grid(True)
    ax_global.set_xlim(-plot_limit, plot_limit)
    ax_global.set_ylim(-2*plot_limit, 2*plot_limit)
    ax_global.tick_params(axis='both', which='major', labelsize=12)
    
    # --- Ax2: Zoomed-in Phase Volume Transformation ---
    ax_zoom.set_title(f'Zoomed-in Phase Volume (Direct LE Vector Plot)\nRenorm Period $T={T_RENORM}$', fontsize=16)
    ax_zoom.set_xlabel(r'Position ($x$)', fontsize=14)
    ax_zoom.set_ylabel(r'Velocity ($\dot{x}$)', fontsize=14)
    ax_zoom.grid(True)
    ax_zoom.tick_params(axis='both', which='major', labelsize=12)
    
    # --- Ax3: Lyapunov Exponent History ---
    ax_le.set_title(r'Largest Lyapunov Exponent ($\lambda_1$) History (Direct)', fontsize=16)
    ax_le.set_xlabel('Time ($t$)', fontsize=14)
    ax_le.set_ylabel(r'LE Estimate $\lambda_1(t)$', fontsize=14)
    ax_le.grid(True)
    ax_le.tick_params(axis='both', which='major', labelsize=12)
    ax_le.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    
    # Solve the ODEs for the boundary conditions (RK4)
    solutions = []
    for initial_conditions in initial_conditions_list:
        solution = integrate_rk4(model, initial_conditions, time_span)
        solutions.append(solution)
        
    # Solve for the central point and the Lyapunov vector (Direct Method)
    solution_Jacobian, solution_perturbation_vector, lyapunov_history = integrate_lyapunov_direct(
        initial_point_for_Jacobian, 
        initial_perturbation, 
        time_span,
        D0,
        T_RENORM
    )
    
    x_solution_Jacobian = solution_Jacobian[:, 0]
    y_solution_Jacobian = solution_Jacobian[:, 1] 
    
    # The separation vector integrated (v) components
    v_x_solution = solution_perturbation_vector[:, 0]
    v_y_solution = solution_perturbation_vector[:, 1]
        
    # Text box for displaying time and LE
    Iter_text_object = ax_zoom.text(0.05, 0.95, '', transform=ax_zoom.transAxes, 
                                    fontsize=13, color='black', 
                                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8, ec="gray"), 
                                    verticalalignment='top')
    
    frames = []    
    
    # LaTeX title for the Duffing equation
    duffing_eq_title = r'Forced Duffing Oscillator: $\ddot{x} + \beta \dot{x} + \alpha x + \delta x^3 = \gamma \cos(\omega t)$'
    
    fig.suptitle(r'Direct Lyapunov Exponent Estimation (Two-Trajectory Method) and Phase Volume Evolution' + '\n' + duffing_eq_title, fontsize=18)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for the suptitle

    # Initialize the LE plot trace
    le_plot, = ax_le.plot([], [], '-', color='orange', linewidth=2, label=r'$\lambda_1(t)$')
    
    # Iterate through time steps to generate frames (step by 4 for smoother GIF playback)
    dt_step = time_points[1] - time_points[0]
    
    # We must ensure that the step size for the GIF (4 * dt_step) is not too large relative to T_RENORM
    # For T_RENORM=1.0 and dt_step=0.1, the step (4*dt) is 0.4. This is fine.
    
    for j in range(1, len(x_solution_Jacobian), 1):    
        
        current_time = time_points[j-1] 
        current_le = lyapunov_history[j-1]
  
        
        # --- Clean previous plots in all subplots ---
        for ax in [ax_global, ax_zoom]:
            for line in ax.lines:
                line.remove()
                
            for artist in ax.collections:
                if isinstance(artist, Quiver):
                    artist.remove()   
                    
            for txt in list(ax.texts):
                if ax is ax_zoom and txt is Iter_text_object:
                    continue 
                if isinstance(txt, Annotation):
                    txt.remove()
        
        # Only update the LE line's data, don't remove the object
        le_plot.set_data(time_points[:j], lyapunov_history[:j])
        
        # --- Plot Trajectories and Current Shape ---
        x_at_j = []
        y_at_j = []
        
        # Determine current boundary points and plot trajectories
        for i, solution in enumerate(solutions):
            x_solution = solution[:, 0]
            y_solution = solution[:, 1]           
            
            # Trajectory trace (light blue) - Plot only on the global view (ax_global)
            ax_global.plot(x_solution[:j], y_solution[:j], '-', color='cyan', alpha=0.2)
            
            # Current boundary points
            x_at_j.append(x_solution[j-1])
            y_at_j.append(y_solution[j-1])
            
        # Convert to numpy arrays for easier min/max calculation
        x_at_j.append(x_at_j[0])
        y_at_j.append(y_at_j[0])
        x_at_j = np.array(x_at_j)
        y_at_j = np.array(y_at_j)

        # Plot the line connecting the red points (the deformed phase volume boundary)
        ax_global.plot(x_at_j, y_at_j, '-', color='red', linewidth=2, zorder=3)
        ax_zoom.plot(x_at_j, y_at_j, '-', color='red', linewidth=2, zorder=3)
        
        # --- Plot Central Point Trajectory and Separation Vector ---
        last_point_x = x_solution_Jacobian[j-1]
        last_point_y = y_solution_Jacobian[j-1]
        
        # Trajectory of the central point (magenta line) - Plot only on the global view
        ax_global.plot(x_solution_Jacobian[:j], y_solution_Jacobian[:j], '-', color='magenta', linewidth=1.5, alpha=0.6)
        
        # Plot the central point itself on both
        ax_global.plot(last_point_x, last_point_y, 'o', color='magenta', markersize=4, zorder=4)
        ax_zoom.plot(last_point_x, last_point_y, 'o', color='magenta', markersize=4, zorder=4)
        
        # --- Plot the Separation Vector (v = zp - z0) on ax_zoom ---
        current_v_x = 1 * v_x_solution[j-1]
        current_v_y = 1 * v_y_solution[j-1]
        v_length = np.linalg.norm([current_v_x, current_v_y])
        
        # Use Quiver to plot the vector
        ax_zoom.quiver(
            last_point_x, last_point_y, current_v_x, current_v_y, 
            scale_units='xy', angles='xy', scale=1, 
            color='darkgreen', linewidth=1.5, zorder=5,
            label='Separation Vector $v$'
        )

        
        # --- Update Ax2: Dynamic Zoom and Text ---
        
        # # Calculate new limits for the zoomed plot (ax_zoom)
        # x_min, x_max = np.min(x_at_j), np.max(x_at_j)
        # y_min, y_max = np.min(y_at_j), np.max(y_at_j)
        
        # # Ensure the vector is always visible in the zoom
        # v_scale_factor = 1.2
        # # Max of current vector component and the initial separation (D0) for minimum span
        # v_x_span = max(abs(current_v_x) * v_scale_factor, D0 * 1.5)
        # v_y_span = max(abs(current_v_y) * v_scale_factor, D0 * 1.5)
        
        # # Combine the bounding box of the deformed shape (red line) and the vector length
        # x_min = min(x_min, last_point_x - v_x_span)
        # x_max = max(x_max, last_point_x + v_x_span)
        # y_min = min(y_min, last_point_y - v_y_span)
        # y_max = max(y_max, last_point_y + v_y_span)

        
        radius = 10*0.01
        x_min = last_point_x - radius
        x_max = last_point_x + radius
        y_min = last_point_y - radius
        y_max = last_point_y + radius   

        # Add a small buffer/padding
        padding_factor = 0.1 # 10% padding
        
        # Calculate range
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Add padding to the limits
        x_pad = x_range * padding_factor if x_range > 0 else D0/5
        y_pad = y_range * padding_factor if y_range > 0 else D0/5
        
        ax_zoom.set_xlim(x_min - x_pad, x_max + x_pad)
        ax_zoom.set_ylim(y_min - y_pad, y_max + y_pad)
        #ax_zoom.set_aspect('equal', adjustable='box') # Keep aspect ratio fixed if needed
        
        # Update the time and LE text box
        Iter_text_object.set_text(f'$t={current_time:.2f}$\n'
                                  f'Separation $\delta={v_length:.3e}$\n'
                                  f'LE: $\\lambda_1={current_le:.4f}$')
       
        # # --- Update Ax3: Lyapunov Exponent History ---
        ax_le.set_xlim(time_points[0], time_points[j-1] + dt_step)
        # # Dynamic y-limits, but cap the min/max for early simulation stability
        ax_le.set_ylim(-0.5, 0.5)
        
     
        plt.pause(0.1)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100) 
        buf.seek(0)
        frames.append(Image.open(buf))

   
    # The duration for the first n-1 frames is the original 'delay'
    durations = [delay] * (len(frames) - 1)
    # The last frame's duration is longer
    durations.append(3000)

    # Save as GIF
    frames[0].save(gif_filename,
                save_all=True, append_images=frames[1:],
                duration=durations, loop=0)
    print(f"Generated GIF: {gif_filename}")


def generate_initial_conditions_circle(num_points, x0, y0, R):
    """
    Generates a list of initial conditions on a circle in the phase space.
    """
    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_values = x0 + R * np.cos(thetas)
    y_values = y0 + R * np.sin(thetas)
    initial_conditions = [[x, y] for x, y in zip(x_values, y_values)]
    return initial_conditions


# --- Example Usage ---

# Center the phase volume near a known point of interest (e.g., around x=1, y=0)
x0_val = 1.0
y0_val = 0.0
radius = 0.01 # Keep the radius small for local linear approximation
num_points = 25 # Number of points defining the circular volume

initial_conditions_list = generate_initial_conditions_circle(
    num_points=num_points, 
    x0 = x0_val, 
    y0 = y0_val, 
    R = radius
)
initial_point_for_Jacobian = [x0_val, y0_val]

# Initial perturbation vector: Direction must be unit-length for D0 calculation to work
initial_perturbation = [1.0, 0.0] 

# Define the time span for the simulation
total_time = 20.0 
num_time_steps = 1500 # Reduced steps for speed and to match T_RENORM better
time_points = np.linspace(0, total_time, num_time_steps)

# Solve the systems and plot the results
solve_and_plot_system(initial_conditions_list, initial_point_for_Jacobian, time_points, initial_perturbation)