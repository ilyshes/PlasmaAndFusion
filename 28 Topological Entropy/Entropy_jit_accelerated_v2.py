#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 09:51:50 2025

@author: ilya
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit 

plt.rcParams['text.usetex'] = True

# Logistic map definition - Keep it simple
def logistic_map(x, r=4.0):
    return r * x * (1 - x)

# JIT-compiled for fast inner loop
@jit(nopython=True)
def orbit_segment(x0, t, r=4.0):
    seg = np.empty(t)
    x = x0
    for i in range(t):
        seg[i] = x
        x = r * x * (1 - x)
    return seg

# *** NEW: JIT-compiled function to handle the array creation loop ***
@jit(nopython=True)
def compute_all_trajectories(initials, t_max, r):
    num_initial = initials.size
    # Initialize the final array
    trajectories = np.empty((num_initial, t_max), dtype=initials.dtype)
    
    # Loop over initial conditions and fill the array
    for i in range(num_initial):
        # Call the JIT-compiled orbit_segment
        trajectories[i, :] = orbit_segment(initials[i], t_max, r)
        
    return trajectories


# JIT-compile the count_distinguishable function, which contains nested loops and array ops
@jit(nopython=True)
def count_distinguishable(segments, eps):
    n = segments.shape[0]
    # Numba requires explicit type for boolean array initialization
    used = np.zeros(n, dtype=np.bool_)
    count = 0
    for i in range(n):
        if used[i]:
            continue
        count += 1
        
        # Calculate sup norm distance between segment i and all others
        t_length = segments.shape[1]
        
        for j in range(n):
            if not used[j]:
                max_diff = 0.0
                # Calculate the max absolute difference (sup norm) element-wise
                for k in range(t_length):
                    diff = np.abs(segments[j, k] - segments[i, k])
                    if diff > max_diff:
                        max_diff = diff
                
                # Check for distinguishability
                if max_diff <= eps:
                    used[j] = True
    return count

# Parameters
r = 4.0
t_max = 15
np.random.seed(0)

# Define different combinations of epsilon and num_initial to plot
Number_of_points = 5000
configs = [
    {'eps': 0.2, 'num_initial': Number_of_points, 'label': r'$\epsilon$=0.1', 'color': 'orange'}]

# Create a figure with two subplots side by side
fig, ax = plt.subplots(1, 3, figsize=(16, 6))

# Customize the left subplot
ax[0].set_xlabel('Segment length t', fontsize=16)
ax[0].set_ylabel(r'N distinguishable N(n,$\epsilon$)' , fontsize=16)
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
ax[0].set_title('Linear Scale',  fontsize=15)
ax[0].grid(True)


# Customize the right subplot
ax[1].set_xlabel('Segment length t' , fontsize=16)
ax[1].set_title('Logarithmic Scale',  fontsize=15)
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
ax[1].set_yscale('log')
ax[1].grid(True, which='both')


# Customize the right subplot
ax[2].set_xlabel('Segment length t' , fontsize=16)
ax[2].set_title('log(N(n,$\epsilon$)) / n',  fontsize=15)
ax[2].tick_params(axis='x', labelsize=15)
ax[2].tick_params(axis='y', labelsize=15)
#ax[2].set_xlim(ax.get_xlim())
ax[2].set_ylim([0,1])
ax[2].grid(True, which='both')


for config in configs:
    eps = config['eps']
    num_initial = config['num_initial']
    
    # Sample random initial conditions
    initials = np.random.rand(num_initial) * 0.3 + 0.1

    # *** FIX: Call the JIT-compiled compute_all_trajectories function ***
    trajectories = compute_all_trajectories(initials, t_max, r)

    # Compute distinguishable counts
    t_values = []
    counts = []
    
    for t in range(1, t_max + 1):
        # Use pre-computed data slice for segments of length t
        segs = trajectories[:, :t] 
        
        # The JIT-compiled count_distinguishable is called here
        distinguishable = count_distinguishable(segs, eps)
        t_values.append(t - 1)
        counts.append(distinguishable)

    # Plotting remains the same
    # Plot on the left subplot (non-log scale)
    
    
    ax[0].plot(t_values, counts, marker='o', label=config['label'], color=config['color'])

    # Plot on the right subplot (log scale)
    log_counts = np.log(counts)
    ax[1].plot(t_values, counts, marker='o', color=config['color'])

    # Linear fit to estimate entropy on the right panel
    coef = np.polyfit(t_values[0:10], log_counts[0:10], 1)
    fit_line = np.polyval(coef, t_values)
    ax[1].plot(t_values, np.exp(fit_line), '--', label=r'h (fit slope) $\approx$ {0:.3f}'.format(coef[0]))


    ax[2].plot(t_values, log_counts/t_values, marker='o', color=config['color'])
    ax[2].plot(t_values, fit_line/t_values, marker='+', color='magenta')



ax[0].legend(fontsize=16)
ax[1].legend(fontsize=16)
# Add a main title for the entire figure
fig.suptitle('Distinguishable Segments of Logistic Map (r=4.0)' + '\n' + 'h - Entropy is the rate of information loss ' + r'$e^{ht}$', fontsize=18)

plt.tight_layout()
plt.show()
plt.savefig('Entropy_jit_accelerated_v2.png')
print("Plot saved as Distinguishable_Segments.png")