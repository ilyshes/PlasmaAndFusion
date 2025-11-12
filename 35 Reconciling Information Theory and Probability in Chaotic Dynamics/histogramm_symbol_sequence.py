#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 15:02:54 2025

@author: ilya
"""

import numpy as np
from itertools import product
from typing import List, Tuple
import matplotlib.pyplot as plt

x_arr = []

# --- Inverse Functions for Logistic Map f(x) = 4x(1-x) ---
# The map is used in the fully chaotic regime (r=4), where the critical
# point is x_c = 0.5.
# The inverse function x_n = (1 +/- sqrt(1 - x_{n+1})) / 2

def inverse_left(x_next: float) -> float:
    """Inverse function for symbol 0 (left branch: x_n < 0.5)"""
    return (1 - np.sqrt(1 - x_next)) / 2

def inverse_right(x_next: float) -> float:
    """Inverse function for symbol 1 (right branch: x_n >= 0.5)"""
    return (1 + np.sqrt(1 - x_next)) / 2

def calculate_representative_point(symbol_sequence: Tuple[int, ...], x_terminal: float = 0.5) -> float:
    """
    Calculates the representative initial condition x_0 for a given symbol
    sequence using inverse iteration.
    
    Args:
        symbol_sequence: The sequence of symbols (0s and 1s), ordered (s_0, s_1, ..., s_{N-1}).
        x_terminal: The terminal point x_N to start the backward iteration.
                    For symbolic dynamics analysis, x_terminal=0.5 (the critical point) is common.

    Returns:
        The calculated initial condition x_0.
    """
    # Inverse iteration uses the sequence in reverse order: (s_{N-1}, s_{N-2}, ..., s_0)
    reversed_sequence = symbol_sequence[::-1]
    
    x_n = x_terminal

    for symbol in reversed_sequence:
        # Check for invalid input to sqrt (should not occur if x_n is in [0, 1])
        if x_n < 0 or x_n > 1:
            raise ValueError(f"Intermediate iterate {x_n} is outside [0, 1]. Map is not invertible here.")

        if symbol == 0:
            # Apply the left inverse branch
            x_n = inverse_left(x_n)
        elif symbol == 1:
            # Apply the right inverse branch
            x_n = inverse_right(x_n)
        else:
            raise ValueError("Symbol must be 0 or 1.")
            
    return x_n

# --- Main Execution ---

SEQUENCE_LENGTH = 12

# Generate all possible sequences of length 12
# This returns a generator of tuples, e.g., (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1), ...
all_sequences = list(product([0, 1], repeat=SEQUENCE_LENGTH))

results: List[Tuple[Tuple[int, ...], float]] = []

print(f"Starting calculation for {len(all_sequences)} sequences of length {SEQUENCE_LENGTH}...")

# Calculate the representative point for every sequence
for sequence in all_sequences:
    try:
        x0 = calculate_representative_point(sequence)
        x_arr.append(x0)
        results.append((sequence, x0))
    except ValueError as e:
        print(f"Error calculating point for sequence {sequence}: {e}")

# --- Output Summary ---

total_count = len(results)
#plt.hist(x_arr,bins = 500)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
#fig.suptitle(r'Forced Duffing Oscillator: Trajectory Dispersion and Topological Entropy ($\epsilon = {}$)'.format(EPSILON), fontsize=18)

# --- LEFT PANEL: Phase Portrait Animation (ax1) ---
ax1.set_title(r'Distribution of representative points', fontsize=16)
ax1.set_xlabel(r'$x$', fontsize=20)
ax1.set_ylabel(r'Counts', fontsize=20)


ax1.set_xlim(-0.01, 1.01)
#ax1.set_ylim(v_min, v_max)
ax1.tick_params(axis='both', which='major', labelsize=18)

# Initialize the animated scatter plot
ax1.hist(x_arr, bins = 100, color='skyblue', edgecolor='black', label=r'')
#scatter = ax1.plot([], [], 'o', color='blue', alpha=1.0, lw=0.1,  label=f'$N={N_TRAJECTORIES}$ Trajectories')

# Initialize the time display text
time_text = ax1.text(0.05, 0.9, '(a)', transform=ax1.transAxes, fontsize=20, verticalalignment='top', alpha=1)


ax1.grid(True, linestyle='--', linewidth=0.5)


# --- RIGHT PANEL: Topological Entropy Analysis (ax2) ---
ax2.set_title(r'Probability density', fontsize=16)
ax2.set_xlabel(r'$x$', fontsize=20)
# The slope is approximately the topological entropy: $h_{top} \approx \frac{d}{dt} \ln(N_{\epsilon}(t))$
ax2.set_ylabel(r'$P(x)$', fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=18)

ax2.hist(x_arr, bins = 100, color='skyblue', edgecolor='black', density=True,  label=r'Normilized histogramm')

x_analytical = np.linspace(0.001, 0.999, 500) # Avoid 0 and 1 to prevent division by zero
rho_analytical = 1 / (np.pi * np.sqrt(x_analytical * (1 - x_analytical)))
     
ax2.plot(x_analytical, rho_analytical, 
              label=r'Analytical density $\rho(x) = \frac{1}{\pi \sqrt{x(1-x)}}$', 
              color='red', 
              linestyle='--', 
              linewidth=2.5)


ax2.set_xlim(-0.01, 1.01)

ax2.set_ylim(0,7)

time_text = ax2.text(0.05, 0.9, '(b)', transform=ax2.transAxes, fontsize=20, verticalalignment='top', alpha=1)

ax2.legend(loc='center', fontsize=18)
ax2.grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('histogramm_symbol_sequence.pdf')
