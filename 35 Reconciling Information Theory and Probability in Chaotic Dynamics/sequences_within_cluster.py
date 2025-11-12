#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified from the original file (cowweb_symbol_sequence.py)
Created on Mon Nov  3 13:34:55 2025

@author: ilya
"""

import numpy as np
import collections
import matplotlib.pyplot as plt

plt.close('all')
# --- Configuration Parameters ---
R = 4.0          # The growth rate parameter (0 < r <= 4).
# N: The number of initial points, uniformly distributed in [0, 1]
N_INITIAL_POINTS = 100000 
# Initial condition (X0 from the original file is no longer used for the main calculation)
# X0 = 0.0001 
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
    
    Returns only the symbol sequence (string).
    """
    current_x = x0
    symbol_sequence = []
    
    # Calculate the trajectory and symbols
    for i in range(num_iterations):
        # Determine the symbol for the current x_n
        if current_x < 0.5:
            symbol = '0'
        else:
            symbol = '1'
        
        symbol_sequence.append(symbol)
        
        # Calculate the next iteration, x_{n+1}
        # Only calculate the next x if we haven't reached the last iteration
        if i < num_iterations - 1:
            current_x = logistic_map(r, current_x)
            
    return "".join(symbol_sequence)



def analyze_symbol_sequences(all_symbol_sequences, initial_points):
    """
    Arranges a 1D list of symbol sequences into a new structure where:
    - The first row contains only the unique symbol sequences.
    - The second row contains the count (number of repeats) and the list 
      of original indices for each unique sequence.

    The final output is a list of two lists (a 2D structure).

    Args:
        all_symbol_sequences (list): A 1D list of strings (the symbol sequences).

    Returns:
        list: A 2D list/array containing:
              [0]: List of unique symbol sequences (strings).
              [1]: List of metadata objects, where each object contains:
                   - 'count': The number of times the sequence appeared (int).
                   - 'indices': A list of original indices where the sequence 
                                was found (list of int).
    """
    # 1. Use an OrderedDict to maintain insertion order (order of first appearance)
    # The value will be a list to store the indices.
    sequence_map = collections.OrderedDict()

    # 2. Iterate through the original list and collect indices for each sequence
    for index, sequence in enumerate(all_symbol_sequences):
        if sequence not in sequence_map:
            # Initialize with an empty list of indices
            sequence_map[sequence] = []
        # Append the current index
        sequence_map[sequence].append(index)

    # 3. Construct the two "rows" (lists) for the final output
    unique_sequences = []
    metadata = []

    x_unique = np.zeros([len(sequence_map),2])
    
    it = 0

    for sequence, indices in sequence_map.items():
        
        # First Row: Unique Sequence
        unique_sequences.append(sequence)

        x_representative = np.mean(initial_points[indices])
        number_of_sequences = len(indices)
        
        x_unique[it,0] =   x_representative
        x_unique[it,1] =   number_of_sequences

        
        # Second Row: Count and Indices
        metadata.append({
            'count': number_of_sequences,
            'indices': indices,
            'x_values': initial_points[indices]           
        })

        it+=1
    # The result is a list containing the two structured lists
    result_array = [
        unique_sequences,
        metadata
    ]

    return result_array, x_unique



# 1. Generate N initial points uniformly distributed in [0, 1]
# We use a fixed seed for reproducibility of the random points
np.random.seed(42) 
initial_points = np.linspace(0.0, 1.0, N_INITIAL_POINTS)

# 2. Generate the symbol sequence for each initial point
all_symbol_sequences = []
for x0 in initial_points:
    # Trajectory is no longer returned as it is not needed for the final output
    symbol_sequence = generate_symbol_sequence(R, x0, MAX_ITERATIONS)
    all_symbol_sequences.append(symbol_sequence)

# 3. Print the results summary
print(f"--- Logistic Map Symbol Sequence Generation for Multiple Points ---")
print(f"Growth Rate (r): {R}")
print(f"Number of Initial Points (N): {N_INITIAL_POINTS}")
print(f"Initial Points Distribution: Uniform in [0, 1]")
print(f"Number of Iterations (Sequence Length): {MAX_ITERATIONS}")
print(f"Generating Partition: '0' for [0, 0.5), '1' for [0.5, 1]")
print("-----------------------------------------------------------------")
print(f"Total Symbol Sequences Generated: {len(all_symbol_sequences)}")

# Print a sample of the generated sequences for verification
print("\nFirst 5 Symbol Sequences:")
for i in range(5):
    print(f"x₀={initial_points[i]:.4f} -> S: {all_symbol_sequences[i]}")

print("\nLast 5 Symbol Sequences:")
for i in range(N_INITIAL_POINTS - 5, N_INITIAL_POINTS):
    print(f"x₀={initial_points[i]:.4f} -> S: {all_symbol_sequences[i]}")


ss, x_unique = analyze_symbol_sequences(all_symbol_sequences, initial_points)

x = x_unique[:,0]
cnt = x_unique[:,1]





# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
#fig.suptitle(r'Forced Duffing Oscillator: Trajectory Dispersion and Topological Entropy ($\epsilon = {}$)'.format(EPSILON), fontsize=18)

# --- LEFT PANEL: Phase Portrait Animation (ax1) ---
ax1.set_title(r'Cluster size $N$ ', fontsize=16)
ax1.set_xlabel(r'$x$', fontsize=20)
ax1.set_ylabel(r'Cluster size $N$ [in units of $\Delta x$]', fontsize=20)


ax1.set_xlim(-0.01, 1.01)
#ax1.set_ylim(v_min, v_max)
ax1.tick_params(axis='both', which='major', labelsize=18)

# Initialize the animated scatter plot
ax1.plot(x, cnt, color='blue', label=r'')
#scatter = ax1.plot([], [], 'o', color='blue', alpha=1.0, lw=0.1,  label=f'$N={N_TRAJECTORIES}$ Trajectories')

# Initialize the time display text
time_text = ax1.text(0.05, 0.9, '(a)', transform=ax1.transAxes, fontsize=20, verticalalignment='top', alpha=1)


ax1.grid(True, linestyle='--', linewidth=0.5)


# --- RIGHT PANEL: Topological Entropy Analysis (ax2) ---
ax2.set_title(r'Inversion of cluster size $N$ from the figure (a)', fontsize=16)
ax2.set_xlabel(r'$x$', fontsize=20)
# The slope is approximately the topological entropy: $h_{top} \approx \frac{d}{dt} \ln(N_{\epsilon}(t))$
ax2.set_ylabel(r'$P(x)$', fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=18)

ax2.plot(x, 1/cnt, color='blue', label=r'Cluster size inversion $\frac{1}{N}$')

x_analytical = np.linspace(0.001, 0.999, 500) # Avoid 0 and 1 to prevent division by zero
rho_analytical = 1 / (np.pi * np.sqrt(x_analytical * (1 - x_analytical)))
     
ax2.plot(x_analytical, rho_analytical, 
              label=r'Analytical density $\rho(x) = \frac{1}{\pi \sqrt{x(1-x)}}$', 
              color='red', 
              linestyle='--', 
              linewidth=2.5)


ax2.set_xlim(-0.01, 1.01)

ax2.set_ylim(1e-2,1e2)

ax2.set_yscale('log')

time_text = ax2.text(0.05, 0.9, '(b)', transform=ax2.transAxes, fontsize=20, verticalalignment='top', alpha=1)

ax2.legend(loc='upper right', fontsize=18)
ax2.grid(True, linestyle='--', linewidth=0.5)

plt.savefig('sequences_within_cluster.pdf')

# x_analytical = np.linspace(0.001, 0.999, 500) # Avoid 0 and 1 to prevent division by zero
# rho_analytical = 1 / (np.pi * np.sqrt(x_analytical * (1 - x_analytical)))
     
# plt.plot(x_analytical, rho_analytical, 
#               label=r'Analytical Solution ($\rho(x) = \frac{1}{\pi \sqrt{x(1-x)}}$)', 
#               color='darkblue', 
#               linestyle='--', 
#               linewidth=1.5)

















