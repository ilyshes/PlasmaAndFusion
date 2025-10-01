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
# No longer importing odeint
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



# Initial perturbation magnitude (d0_norm)
INITIAL_PERTURBATION_MAG = 1e-4 

def model(z, t):
    """
    Defines the system of differential equations for the forced Duffing oscillator.
    This function represents f(z, t) in the ODE dz/dt = f(z, t).
    
    Args:
        z (list): A list containing the current values of x (position) and y (velocity).
        t (float): The current time.
        
    Returns:
        np.ndarray: The derivatives [dx/dt, dy/dt].
    """
    x, y = z
    dxdt = y
    # dydt = - beta * y - alpha * x - delta * x^3 + gamma * cos(omega * t)
    dydt = - BETA * y - ALPHA * x - DELTA * x**3 + GAMMA * np.cos(OMEGA * t)
    return np.array([dxdt, dydt]) # Changed to return a NumPy array for easier RK4 math


# -- Numerical Integration (Runge-Kutta 4th order) rk4_step approach
def rk4_step(func, z, t, dt):
    """
    Performs one step of the 4th-order Runge-Kutta integration method.

    Args:
        func (callable): The function defining dz/dt = func(z, t).
        z (np.ndarray): The current state vector [x, y].
        t (float): The current time.
        dt (float): The time step size.

    Returns:
        np.ndarray: The state vector at time t + dt.
    """
    # k1 = f(z, t)
    k1 = dt * func(z, t)
    
    # k2 = f(z + k1/2, t + dt/2)
    k2 = dt * func(z + k1 / 2, t + dt / 2)
    
    # k3 = f(z + k2/2, t + dt/2)
    k3 = dt * func(z + k2 / 2, t + dt / 2)
    
    # k4 = f(z + k3, t + dt)
    k4 = dt * func(z + k3, t + dt)
    
    # z(t + dt) = z(t) + (k1 + 2*k2 + 2*k3 + k4) / 6
    z_next = z + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return z_next


def integrate_rk4(func, z0, time_points):
    """
    Integrates the system of ODEs using the RK4 method over a sequence of time points.

    Args:
        func (callable): The function defining dz/dt = func(z, t).
        z0 (list): The initial state vector [x0, y0].
        time_points (np.ndarray): An array of time points. Assumes fixed step size.

    Returns:
        np.ndarray: The solution array, shape (len(time_points), len(z0)).
    """
    solution = np.zeros((len(time_points), len(z0)))
    solution[0] = np.array(z0)
    
    # The fixed time step (dt) is the difference between consecutive time points
    # This assumes the time_points array is uniformly spaced.
    dt = time_points[1] - time_points[0] 
    
    z_current = np.array(z0)
    
    for i in range(len(time_points) - 1):
        t_current = time_points[i]
        z_next = rk4_step(func, z_current, t_current, dt)
        solution[i+1] = z_next
        z_current = z_next
        
    return solution

def integrate_rk4_deformation(func, z0, time_points):
    """
    Integrates the system of ODEs using the RK4 method over a sequence of time points.

    Args:
        func (callable): The function defining dz/dt = func(z, t).
        z0 (list): The initial state vector [x0, y0].
        time_points (np.ndarray): An array of time points. Assumes fixed step size.

    Returns:
        np.ndarray: The solution array, shape (len(time_points), len(z0)).
    """
    solution = np.zeros((len(time_points), len(z0)))
    solution_v = np.zeros((len(time_points), len(z0)))
    solution[0] = np.array(z0)
    
   
    
    # 2. Set initial perturbation vector (d0)
    d_unit = np.array([0.0, 1.0])
    d_0 = d_unit * INITIAL_PERTURBATION_MAG 
    solution_v[0] = d_0
    # The fixed time step (dt) is the difference between consecutive time points
    # This assumes the time_points array is uniformly spaced.
    dt = time_points[1] - time_points[0] 
    
    #  Set initial state for unperturbed Trajectory
    z_current = np.array(z0)
    
    #  Set initial state for perturbed Trajectory
    zp_current = z_current + d_0
    
    
    
    for i in range(len(time_points) - 1):
        t_current = time_points[i]
        # 1. Integrate both trajectories
        z_next = rk4_step(func, z_current, t_current, dt)
        zp_next = rk4_step(func, zp_current, t_current, dt)
        
        # 2. Update the separation vector
        v_next_integrated = zp_next - z_next
        
        # Calculate the length of the integrated vector
        v_norm_integrated = np.linalg.norm(v_next_integrated)
        
        # Accumulate the logarithm of the expansion factor
        #lyapunov_log_sum += np.log(v_norm_integrated / d0) # expansion factor is v_norm / d0
        
        # Rescale the vector v_next_integrated back to the initial distance d0
        v_renormalized = ( v_next_integrated / v_norm_integrated ) * INITIAL_PERTURBATION_MAG
        
        # Calculate the new perturbed state
        zp_next = z_next + v_renormalized
    
        solution[i+1] = z_next
        solution_v[i+1] = v_next_integrated
        
        
        z_current = z_next
        zp_current = zp_next
    return solution, solution_v

def jacobian_matrix(x, y):
    """
    Calculates the Jacobian matrix of the system at a given point (x, y).
    
    J = [[d(dxdt)/dx, d(dxdt)/dy], [d(dydt)/dx, d(dydt)/dy]]
    
    Returns:
        np.ndarray: The 2x2 Jacobian matrix.
    """
    # dxdt = y
    J11 = 0.0
    J12 = 1.0
    
    # dydt = - BETA * y - ALPHA * x - DELTA * x**3 + GAMMA * np.cos(OMEGA * t)
    # Note: The time-dependent term's derivatives w.r.t x and y are zero.
    J21 = - ALPHA - 3 * DELTA * x**2
    J22 = - BETA
    return np.array([[J11, J12], [J21, J22]])


def solve_and_plot_system(initial_conditions_list, initial_point_for_Jacobian, time_span):
    """
    Solves the system of ODEs for a given set of initial conditions and time span,
    and generates an animated plot showing phase volume deformation.

    Args:
        initial_conditions_list (list): A list of lists, where each inner list [x0, y0]
                                         represents an initial condition.
        initial_point_for_Jacobian (list): The central point of the volume to track the Jacobian.
        time_span (np.ndarray): A NumPy array representing the time points for the solution.
    """
    
    gif_filename = 'Duffing_Shape_Transformation_RK4.gif' # Updated filename
    
    plt.style.use('seaborn-v0_8-whitegrid')
    # Renamed the subplots to be explicit: ax_global and ax_zoom
    fig, (ax_global, ax_zoom) = plt.subplots(1, 2, figsize=(16.3, 7.5))

    plot_limit = 3.2

 
    # --- Ax1: Global Phase Space View ---
    ax_global.set_title('Global Phase Space View and Trajectories', fontsize=18)
    ax_global.set_xlabel(r'Position ($x$)', fontsize=18)
    ax_global.set_ylabel(r'Velocity ($\dot{x}$)', fontsize=18)
    ax_global.grid(True)
    ax_global.set_xlim(-plot_limit, plot_limit)
    ax_global.set_ylim(-2*plot_limit, 2*plot_limit)
    ax_global.tick_params(axis='both', which='major', labelsize=14)
    
    # --- Ax2: Zoomed-in Phase Volume Transformation ---
    ax_zoom.set_title('Zoomed-in Phase Volume Deformation (RK4 Integration)', fontsize=18) # Updated title
    ax_zoom.set_xlabel(r'Position ($x$)', fontsize=18)
    ax_zoom.set_ylabel(r'Velocity ($\dot{x}$)', fontsize=18)
    ax_zoom.grid(True)
    ax_zoom.tick_params(axis='both', which='major', labelsize=14)
    
    
    # Calculate grid for vector field once (using ax_global's limits for a global vector field if desired, currently commented out)
    # x_min, x_max = ax_global.get_xlim()
    # y_min, y_max = ax_global.get_ylim()
    # x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 25),
    #                              np.linspace(y_min, y_max, 25))
    
    # Placeholder for the streamplot collection for easy removal
    # stream_plot = None
    

    # Solve the ODEs for all initial conditions using RK4
    solutions = []
    for initial_conditions in initial_conditions_list:
        # !!! Replaced odeint with integrate_rk4 !!!
        solution = integrate_rk4(model, initial_conditions, time_span) 
        solutions.append(solution)
        
    # Solve for the central point (for Jacobian tracking) using RK4
    # !!! Replaced odeint with integrate_rk4 !!!
    solution_Jacobian, solution_perturbation_vector = integrate_rk4_deformation(model, initial_point_for_Jacobian, time_span) 
    x_solution_Jacobian = solution_Jacobian[:, 0]
    y_solution_Jacobian = solution_Jacobian[:, 1] 
    
    # The separation vector integrated (v) components
    v_x_solution = solution_perturbation_vector[:, 0]
    v_y_solution = solution_perturbation_vector[:, 1]
        
    # Text box for displaying time (using ax_zoom for placement)
    # Positioning adjusted for ax_zoom
    Iter_text_object = ax_zoom.text(0.05, 0.95, '', transform=ax_zoom.transAxes, 
                                    fontsize=15, color='black', 
                                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8, ec="gray"), 
                                    verticalalignment='top')
    
    frames = []    
    
    # LaTeX title for the Duffing equation
    duffing_eq_title = r'Forced Duffing Oscillator: $\ddot{x} + \beta \dot{x} + \alpha x + \delta x^3 = \gamma \cos(\omega t)$'
    
    fig.suptitle(r'Evolution of a Phase Volume Element (RK4): Global Spreading (Chaos) - The points spread and diverge from each other along one direction' + '\n' + duffing_eq_title, fontsize=18)
    
    plt.tight_layout() # Ensures titles and labels don't overlap

    # Iterate through time steps to generate frames (step by 4 for smoother GIF playback)
    for j in range(1, len(x_solution_Jacobian), 4):    
        
        current_time = time_points[j-1] 
  
        
        # --- Clean previous plots in both subplots ---
        for ax in [ax_global, ax_zoom]:
            for line in ax.lines:
                line.remove()
                
            for artist in ax.collections:
                if isinstance(artist, Quiver):
                    artist.remove()   
                    
            for txt in list(ax.texts):
                # Preserve the main time text box on ax_zoom
                if ax is ax_zoom and txt is Iter_text_object:
                    continue 
                if isinstance(txt, Annotation):
                    txt.remove()        
        
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
        # Plot on both global and zoom axes
        ax_global.plot(x_at_j, y_at_j, '-', color='red', linewidth=2, zorder=3)
        ax_zoom.plot(x_at_j, y_at_j, '-', color='red', linewidth=2, zorder=3)
        
        # --- Plot Jacobian Eigenvectors at the Central Point ---
        last_point_x = x_solution_Jacobian[j-1]
        last_point_y = y_solution_Jacobian[j-1]
        
        # Trajectory of the central point (magenta line) - Plot only on the global view
        ax_global.plot(x_solution_Jacobian[:j], y_solution_Jacobian[:j], '-', color='magenta', linewidth=1.5, alpha=0.6)
        
        
        # Plot the central point itself on both
        ax_global.plot(last_point_x, last_point_y, 'o', color='magenta', markersize=4, zorder=4)
        ax_zoom.plot(last_point_x, last_point_y, 'o', color='magenta', markersize=4, zorder=4)
        
        
        # --- Plot the Separation Vector (v = zp - z0) on ax_zoom ---
        current_v_x = 100 * v_x_solution[j-1]
        current_v_y = 100 * v_y_solution[j-1]
        v_length = np.linalg.norm([current_v_x, current_v_y])
        
        # Use Quiver to plot the vector
        ax_zoom.quiver(
            last_point_x, last_point_y, current_v_x, current_v_y, 
            scale_units='xy', angles='xy', scale=0.1, 
            color='darkgreen', linewidth=1.5, zorder=5,
            label='Separation Vector $v$'
        )
        
        # --- Update Ax2: Dynamic Zoom and Text ---
        
        # Calculate new limits for the zoomed plot (ax_zoom)
        # Determine the bounding box of the deformed shape
        x_min, x_max = np.min(x_at_j), np.max(x_at_j)
        y_min, y_max = np.min(y_at_j), np.max(y_at_j)
        
        radius = 30*0.01
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
        x_pad = x_range * padding_factor if x_range > 0 else 0.005 # Min padding for very small shapes
        y_pad = y_range * padding_factor if y_range > 0 else 0.005
        
        ax_zoom.set_xlim(x_min - x_pad, x_max + x_pad)
        ax_zoom.set_ylim(y_min - y_pad, y_max + y_pad)
        #ax_zoom.set_aspect('equal', adjustable='box') # Keep the aspect ratio fixed for better shape representation
        
        # Update the time text box (already on ax_zoom)
        Iter_text_object.set_text(f'$t={current_time:.2f}$')
       
        # J = jacobian_matrix(last_point_x, last_point_y)
        # eig_val, eig_vec = np.linalg.eig(J)
        
        #****************** Plot eigenvectors (Optional, currently commented out) ***********************
        # The eigenvectors were previously plotting to ax2, now ax_zoom, but are irrelevant for simple zoom
        # r1_x, r1_y = eig_vec[:, 0] / np.linalg.norm(eig_vec[:, 0])
        # ... (plotting logic for eigenvectors)
             
        
     
        plt.pause(0.1)
        #Save frame logic (commented out in original, keeping it out)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100) # Lower dpi for faster image generation
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
x0_val = 0.0
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


# Define the time span for the simulation
# Use a longer time span to show evolution towards the attractor
total_time = 20.0 
num_time_steps = 1500 # Total points in the solution
time_points = np.linspace(0, total_time, num_time_steps)

# Solve the systems and plot the results
solve_and_plot_system(initial_conditions_list, initial_point_for_Jacobian, time_points)