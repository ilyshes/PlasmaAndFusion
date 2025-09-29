#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 09:27:19 2025

@author: ilya
"""

import numpy as np
from scipy.integrate import odeint
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

def model(z, t):
    """
    Defines the system of differential equations for the forced Duffing oscillator.
    
    Args:
        z (list): A list containing the current values of x (position) and y (velocity).
        t (float): The current time.
        
    Returns:
        list: The derivatives [dx/dt, dy/dt].
    """
    x, y = z
    dxdt = y
    # dydt = - beta * y - alpha * x - delta * x^3 + gamma * cos(omega * t)
    dydt = - BETA * y - ALPHA * x - DELTA * x**3 + GAMMA * np.cos(OMEGA * t)
    return [dxdt, dydt]


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
    
    gif_filename = 'Duffing_Shape_Transformation.gif'
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.3, 7.5))

    plot_limit = 3.2
    ax1.set_xlim(-plot_limit, plot_limit)
    ax1.set_ylim(-2*plot_limit, 2*plot_limit)
    ax2.set_xlim(-plot_limit, plot_limit)
    ax2.set_ylim(-2*plot_limit, 2*plot_limit)
 
    #ax1.set_aspect('equal', adjustable='box')
    
    # Calculate grid for vector field once
    x_min, x_max = ax1.get_xlim()
    y_min, y_max = ax1.get_ylim()
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 25),
                                 np.linspace(y_min, y_max, 25))
    
    # Placeholder for the streamplot collection for easy removal
    stream_plot = None
    
    
    # --- Ax2: Phase Volume Transformation ---
    ax2.set_title('Shape Transformation of the Selected Phase Space Volume', fontsize=18)
    ax2.set_xlabel(r'Position ($x$)', fontsize=18)
    ax2.set_ylabel(r'Velocity ($\dot{x}$)', fontsize=18)
    ax2.grid(True)
    
    

    # Solve the ODEs for all initial conditions
    solutions = []
    for initial_conditions in initial_conditions_list:
        solution = odeint(model, initial_conditions, time_span)
        solutions.append(solution)
        
    # Solve for the central point (for Jacobian tracking)
    solution_Jacobian = odeint(model, initial_point_for_Jacobian, time_span)
    x_solution_Jacobian = solution_Jacobian[:, 0]
    y_solution_Jacobian = solution_Jacobian[:, 1] 
        
    # Text box for displaying time and Jacobian properties (Ax2)
    Iter_text_object = ax2.text(-2.9, 2.3, '', fontsize=15, color='black', 
                               bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8, ec="gray"))
    
    frames = []    
    
    # LaTeX title for the Duffing equation
    duffing_eq_title = r'Forced Duffing Oscillator: $\ddot{x} + \beta \dot{x} + \alpha x + \delta x^3 = \gamma \cos(\omega t)$' 
        # + '\n' + r'Parameters: $\beta={}$, $\alpha={}$, $\delta={}$, $\gamma={}$, $\omega={}$').format(BETA, ALPHA, DELTA, GAMMA, OMEGA)
    
    # Iterate through time steps to generate frames (step by 4 for smoother GIF playback)
    for j in range(10, len(x_solution_Jacobian), 4):    
        
        # --- Update Ax1: Streamlines based on current time ---
        
        # Remove previous streamplot, if it exists
        ax1.cla()
        
        # Define plot limits suitable for the forced Duffing attractor



        fig.suptitle(r'Evolution of a Phase Volume Element: Global Spreading (Chaos) - The points spread and diverge from each other along one direction' + '\n' + duffing_eq_title, fontsize=18)


        ax1.set_xlim(-plot_limit, plot_limit)
        ax1.set_ylim(-2*plot_limit, 2*plot_limit)
        ax2.set_xlim(-plot_limit, plot_limit)
        ax2.set_ylim(-2*plot_limit, 2*plot_limit)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        
        # --- Ax1 Setup: Phase Portrait (Will update dynamically) ---
        ax1.set_title('Phase Portrait (X vs Y) - Instantaneous Vector Field', fontsize=16)
        ax1.set_xlabel(r'Position ($x$)', fontsize=18)
        ax1.set_ylabel(r'Velocity ($\dot{x}$)', fontsize=18)
        ax1.grid(True)         
        plt.tight_layout()
                 
                 
                 
                 
                 
        
        # Calculate the vector field at the current time point
        current_time = time_points[j-1] 
        u = np.zeros_like(x_grid)
        v = np.zeros_like(y_grid)
        
        for i in range(x_grid.shape[0]):
            for k in range(x_grid.shape[1]):
                # model expects z=[x, y] and t
                dxdt, dydt = model([x_grid[i, k], y_grid[i, k]], current_time)
                u[i, k] = dxdt
                v[i, k] = dydt

        # Plot the updated streamlines
        norm = np.sqrt(u**2 + v**2)
        stream_plot = ax1.streamplot(x_grid, y_grid, u, v, density=1.5, color=norm, cmap='hot', linewidth=1)
        
        # Update Ax1's title with the current time (optional but helpful)
        ax1.set_title(f'Phase Portrait (X vs Y) - Instantaneous Vector Field at $t={current_time:.2f}$', fontsize=16)
        
        
        # --- Update Ax2: Phase Volume Deformation ---
        
        # Clear previous plots in the deformation subplot (ax2)
        for line in ax2.lines:
            line.remove()
            
        for artist in ax2.collections:
            if isinstance(artist, Quiver):
                artist.remove()   
                
        for txt in list(ax2.texts):
            if isinstance(txt, Annotation):
                if txt is not Iter_text_object: # Preserve the main text box
                    txt.remove()        
        
        # --- Plot Trajectories and Current Shape ---
        x_at_j = []
        y_at_j = []
        
        # Plot initial positions (green circle) once, or ensure they are replotted
        # for clean animation
        for i, solution in enumerate(solutions):
            x_solution = solution[:, 0]
            y_solution = solution[:, 1]           
            #ax2.plot(x_solution[0], y_solution[0], 'o', color='green', markersize=4, alpha=0.8)    
            
            # Plot all updated lines (trajectories) and current points (red shape)
            # Trajectory trace (light blue)
            ax2.plot(x_solution[:j], y_solution[:j], '-', color='cyan', alpha=0.2)
            
            # Current boundary points
            x_at_j.append(x_solution[j-1])
            y_at_j.append(y_solution[j-1])
            
        # Plot the line connecting the red points (the deformed phase volume boundary)
        ax2.plot(x_at_j, y_at_j, '-', color='red', linewidth=2, zorder=3)
        
        
        # --- Plot Jacobian Eigenvectors at the Central Point ---
        last_point_x = x_solution_Jacobian[j-1]
        last_point_y = y_solution_Jacobian[j-1]
        
        # Trajectory of the central point (magenta line)
        ax2.plot(x_solution_Jacobian[:j], y_solution_Jacobian[:j], '-', color='magenta', linewidth=1.5, alpha=0.6)
        
        # Update the time text box
        Iter_text_object.set_text(f'$t={current_time:.2f}$')
       
        J = jacobian_matrix(last_point_x, last_point_y)
        eig_val, eig_vec = np.linalg.eig(J)
        
        #****************** Plot eigenvectors ************************************************************
        # # Eigenvector 1
        # r1_x, r1_y = eig_vec[:, 0] / np.linalg.norm(eig_vec[:, 0]) # Normalize for consistent length
        # ax2.quiver(last_point_x, last_point_y, r1_x, r1_y, 
        #           color='black', scale=2, scale_units='xy', angles='xy', headwidth=5, 
        #           headlength=7, width=0.008, alpha=1, zorder=4)
        # ax2.annotate(r'$r_1$', (last_point_x + r1_x/2.0, last_point_y + r1_y/2.0),
        #       textcoords="offset points", xytext=(5,5), ha='center',
        #       fontsize=14, color='black', zorder=4)
     
        # # Eigenvector 2
        # r2_x, r2_y = eig_vec[:, 1] / np.linalg.norm(eig_vec[:, 1]) # Normalize
        # ax2.quiver(last_point_x, last_point_y, r2_x, r2_y, 
        #           color='black', scale=2, scale_units='xy', angles='xy', headwidth=5, 
        #           headlength=7, width=0.008, alpha=1, zorder=4)
        # ax2.annotate(r'$r_2$', (last_point_x + r2_x/2.0, last_point_y + r2_y/2.0),
        #       textcoords="offset points", xytext=(5,5), ha='center',
        #       fontsize=14, color='black', zorder=4)
             
        
     
        plt.pause(0.02)
        #Save frame logic (commented out in original, keeping it out)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(Image.open(buf))

   
    # # The duration for the first n-1 frames is the original 'delay'
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
num_points = 20 # Number of points defining the circular volume

initial_conditions_list = generate_initial_conditions_circle(
    num_points=num_points, 
    x0 = x0_val, 
    y0 = y0_val, 
    R = radius
)
initial_point_for_Jacobian = [x0_val, y0_val]


# Define the time span for the simulation
# Use a longer time span to show evolution towards the attractor
total_time = 30.0 
num_time_steps = 1000 # Total points in the solution
time_points = np.linspace(0, total_time, num_time_steps)

# Solve the systems and plot the results
solve_and_plot_system(initial_conditions_list, initial_point_for_Jacobian, time_points)