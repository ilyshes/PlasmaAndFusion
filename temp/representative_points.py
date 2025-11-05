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
plt.hist(x_arr,bins = 100)

# print(f"\n--- Results Summary (Total Calculated Points: {total_count}) ---")
# print("Format: (s_0, s_1, ..., s_{11}) : x_0")

# # Print the first 10 results
# print("\n--- First 10 Results (Sequences starting with 0s) ---")
# for i in range(min(10, total_count)):
#     seq, x0 = results[i]
#     print(f"{str(seq):<40} : {x0:.15f}")

# if total_count > 20:
#     print("\n[...]\n")
#     # Print the last 10 results (Sequences ending with 1s)
#     print("--- Last 10 Results (Sequences ending with 1s) ---")
#     for i in range(total_count - 10, total_count):
#         seq, x0 = results[i]
#         print(f"{str(seq):<40} : {x0:.15f}")
        
# print("\nCalculation complete.")
