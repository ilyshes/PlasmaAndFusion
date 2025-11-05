#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 10:51:43 2025

@author: ilya
"""

import numpy as np
import matplotlib.pyplot as plt

time_series = []

def logistic_map(x, r):
    """
    The logistic map iteration function: x_n+1 = r * x_n * (1 - x_n)
    """
    return r * x * (1 - x)

def plot_invariant_pdf(r=4.0, N_steps=5000000, N_burn=1000):
    """
    Numerically approximates and plots the invariant probability density function (PDF) 
    for the logistic map at a given parameter 'r'.

    For r=4.0, it also plots the analytical solution for comparison.

    Args:
        r (float): The parameter value for the logistic map (default is 4.0 for full chaos).
        N_steps (int): Total number of iterations to run after the burn-in period.
        N_burn (int): Number of initial steps to discard (to allow the system to reach the attractor).
    """
    if r <= 3.0:
        print(f"R parameter ({r}) is too low for chaos. No continuous invariant PDF exists.")
        return

    # --- 1. Simulation and Data Generation ---
    x = 0.001  # Initial condition (must be in [0, 1])
    
    # Run burn-in phase
    for _ in range(N_burn):
        x = logistic_map(x, r)
    
    # Generate the time series used for the PDF approximation
    #time_series = np.empty(N_steps)
    for i in range(N_steps):
        x = logistic_map(x, r)
        time_series.append( x )
        
    # --- 2. Numerical PDF Approximation (Histogram) ---
    # Use 100 bins over the range [0, 1]
    n_bins = 100
    hist_values, bin_edges = np.histogram(
        time_series, 
        bins=n_bins, 
        range=(0, 1), 
        density=True  # Normalize the histogram to approximate the PDF
    )
    # Calculate bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- 3. Plotting ---
    plt.figure(figsize=(10, 6))

    # Plot the numerical PDF (histogram approximation)
    
    #plt.hist(time_series, bins = 100)
    
    plt.plot(bin_centers, hist_values, 
             label=f'Numerical PDF (r={r}, {N_steps} iterations)', 
             color='darkorange', 
             linewidth=2.5, 
             alpha=0.7)
    
    # If r=4, plot the analytical solution
    if np.isclose(r, 4.0):
        # Analytical solution for r=4: rho(x) = 1 / (pi * sqrt(x * (1 - x)))
        x_analytical = np.linspace(0.001, 0.999, 500) # Avoid 0 and 1 to prevent division by zero
        rho_analytical = 1 / (np.pi * np.sqrt(x_analytical * (1 - x_analytical)))
        
        plt.plot(x_analytical, rho_analytical, 
                 label=r'Analytical Solution ($\rho(x) = \frac{1}{\pi \sqrt{x(1-x)}}$)', 
                 color='darkblue', 
                 linestyle='--', 
                 linewidth=1.5)
        
    plt.title(f'Invariant Probability Density Function for Logistic Map (r={r})', fontsize=14)
    plt.xlabel('x (Attractor points)', fontsize=12)
    plt.ylabel('Probability Density $\\rho(x)$', fontsize=12)
    plt.xlim(0, 1)
    # Set y limit to show the characteristic U-shape clearly, 
    # as the density goes to infinity near the boundaries.
    plt.ylim(0, 5) 
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Run the plot for the fully chaotic case (r=4)
    plot_invariant_pdf(r=4.0)

    # You can also try other values (e.g., r=3.57 for a slightly different look)
    # plot_invariant_pdf(r=3.9, N_steps=1000000)
    # plot_invariant_pdf(r=3.5) # This will show period-doubling attractor points, not a continuous PDF
