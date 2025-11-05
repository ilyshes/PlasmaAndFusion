#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 13:34:55 2025

@author: ilya
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# --- Configuration Parameters (Using the values from the original file) ---
R = 3.6          # The growth rate parameter (0 < r <= 4).
X0 = 0.0001         # Initial condition (must be between 0 and 1)
MAX_ITERATIONS = 12 # Total number of points (x0 to x11) for the symbol sequence

def logistic_map(r, x):
    """
    Calculates the next iteration of the logistic map: x_{n+1} = r * x_n * (1 - x_n).
    """
    return r * x * (1 - x)

def generate_symbol_sequence(r, x0, num_iterations):
    """
    Generates the symbol sequence for the logistic map using the generating partition:
    Symbol '0' for x_n in [0, 0.5)
    Symbol '1' for x_n in [0.5, 1]
    """
    trajectory = [x0]
    symbol_sequence = []
    
    # Calculate the trajectory and symbols
    current_x = x0
    for i in range(num_iterations):
        # Determine the symbol for the current x_n
        if current_x < 0.5:
            symbol = '0'
        else:
            symbol = '1'
        
        symbol_sequence.append(symbol)
        
        # Calculate the next iteration, x_{n+1}
        # Note: We calculate one more x_n than the symbols needed (x_0 to x_{MAX_ITERATIONS-1})
        if i < num_iterations - 1:
            next_x = logistic_map(r, current_x)
            trajectory.append(next_x)
            current_x = next_x
    
    return trajectory, "".join(symbol_sequence)


# 1. Generate the symbol sequence and trajectory
trajectory, symbol_sequence = generate_symbol_sequence(R, X0, MAX_ITERATIONS)

# 2. Print the results
print(f"--- Logistic Map Symbol Sequence Generation ---")
print(f"Growth Rate (r): {R}")
print(f"Initial Condition (x₀): {X0}")
print(f"Number of Iterations: {MAX_ITERATIONS}")
print(f"Generating Partition: '0' for [0, 0.5), '1' for [0.5, 1]")
print("--------------------------------------------------")

print("Trajectory (x₀ to x₁₁):")
for i, x in enumerate(trajectory):
    print(f"x_{i}: {x:.8f}")

print("\nSymbol Sequence (S₀ to S₁₁):")
print(symbol_sequence)

# 3. Suppress the animation part
# The animation and plotting code has been removed/commented out as it is no longer the primary goal.

# # 3. Set up the Matplotlib figure and axes (Removed)
# # ... (plotting code)
#
# # 4. Initialization function (Removed)
# # ... (init function)
#
# # 5. Animation function (Removed)
# # ... (animate function)
#
# # 6. Create the animation object (Removed)
# # ... (animation object creation)
#
# # Show the plot with the animation (Removed)
# # plt.legend()
# # plt.show()