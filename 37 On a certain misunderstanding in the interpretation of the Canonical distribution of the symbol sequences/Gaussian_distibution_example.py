#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 15:21:21 2025

@author: ilya
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_gaussian(mean: float, std_dev: float, x_range_multiplier: float = 4):
    """
    Plots the Gaussian (Normal) distribution given its central point (mean) and width (standard deviation).

    Args:
        mean (float): The central point (mu) of the distribution.
        std_dev (float): The width (sigma) or standard deviation of the distribution.
        x_range_multiplier (float): Multiplier to determine the x-axis range
                                    (e.g., 4 means plot from mean - 4*std_dev to mean + 4*std_dev).
    """
    if std_dev <= 0:
        print("Error: Standard deviation must be greater than zero.")
        return

    # 1. Define the range for x-values
    # We plot over a range that spans 'x_range_multiplier' standard deviations 
    # around the mean to capture the majority of the curve (e.g., 4 sigma covers > 99.99%).
    x_min = mean - x_range_multiplier * std_dev
    x_max = mean + x_range_multiplier * std_dev
    
    # Generate 1000 evenly spaced points within the defined range
    x = np.linspace(x_min, x_max, 1000)

    # 2. Calculate the Probability Density Function (PDF) values
    # The norm.pdf function calculates the Gaussian curve:
    # f(x | mu, sigma) = (1 / sqrt(2*pi*sigma^2)) * exp(- (x - mu)^2 / (2*sigma^2))
    # Note: We use the statistical function from scipy for robustness and ease.
    y = norm.pdf(x, loc=mean, scale=std_dev)

    # 3. Plotting the distribution
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color='darkblue', linewidth=2, label=f'')
    
    # Fill the area under the curve for better visualization
    plt.fill_between(x, y, color='lightblue', alpha=0.5)

    # Mark the central point (mean)
    plt.axvline(mean, color='red', linestyle='--', label=f'')
    
    # Mark the points one standard deviation away (inflection points)
    plt.axvline(mean - std_dev, color='green', linestyle=':', label='')
    plt.axvline(mean + std_dev, color='green', linestyle=':')
    
    plt.tick_params(axis='both', which='major', labelsize=14)
    # 4. Add labels, title, and grid
    plt.title('Gaussian (Normal) Distribution', fontsize=16, fontweight='bold')
    plt.xlabel(r'$\ell$', fontsize=20)
    plt.ylabel(r'Probability Density $\rho$($\ell$)', fontsize=20)
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize the plot appearance
   # plt.xticks(np.arange(int(x_min), int(x_max) + 1, std_dev))
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    
    # Show the plot
    plt.show()

# --- Example Usage ---

# # Example 1: Standard Normal Distribution
# print("Plotting Standard Normal Distribution (mean=0, std_dev=1)")
# plot_gaussian(mean=0, std_dev=1)

# Example 2: Different Central Point and Wider Distribution
print("Plotting Wider Distribution (mean=15, std_dev=2.5)")
plot_gaussian(mean=3e-4, std_dev=3e-5)

plt.savefig('Gaussian_distibution_example.pdf')

# # Example 3: Narrower Distribution
# print("Plotting Narrower Distribution (mean=50, std_dev=0.75)")
# plot_gaussian(mean=50, std_dev=0.75)