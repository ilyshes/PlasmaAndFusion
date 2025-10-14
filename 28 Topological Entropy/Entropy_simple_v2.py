#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 09:42:53 2025

@author: ilya
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

# Logistic map definition
def logistic_map(x, r=4.0):
    return r * x * (1 - x)

# Generate orbit segment of length t
# This function is now only used for pre-calculating the full trajectories up to t_max
def orbit_segment(x0, t, r=4.0):
    seg = np.empty(t)
    x = x0
    for i in range(t):
        seg[i] = x
        x = logistic_map(x, r)
    return seg

# Count distinguishable orbit segments at resolution eps
def count_distinguishable(segments, eps):
    n = segments.shape[0]
    used = np.zeros(n, dtype=bool)
    count = 0
    for i in range(n):
        if used[i]:
            continue
        count += 1
        # sup norm distance between segment i and all others
        # segments - segments[i] uses broadcasting
        dists = np.max(np.abs(segments - segments[i]), axis=1)
        within = dists <= eps
        used = used | within
    return count

# Parameters
r = 4.0
t_max = 20
np.random.seed(0)

# Define different combinations of epsilon and num_initial to plot
Number_of_points = 15000
configs = [
    {'eps': 0.15, 'num_initial': Number_of_points, 'label': r'$\epsilon$=0.1', 'color': 'orange'}]

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
#ax[1].set_ylabel('log(number of distinguishable segments)', fontsize=16)
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
ax[2].set_ylim([0,1])
ax[2].grid(True, which='both')


for config in configs:
    eps = config['eps']
    num_initial = config['num_initial']
    
    # Sample random initial conditions
    initials = np.random.rand(num_initial) * 0.2 + 0.1

    # *** Optimization: Pre-compute full trajectories up to t_max ***
    
    # trajectories will be a numpy array of shape (num_initial, t_max)
    trajectories = np.array([orbit_segment(x0, t_max, r) for x0 in initials])

    # Compute distinguishable counts
    t_values = []
    counts = []
    
    for t in range(1, t_max + 1):
        # *** Optimization: Use pre-computed data slice for segments of length t ***
        # segments of length t are the first t columns of trajectories
        segs = trajectories[:, :t] 
        
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
    # We fit all data points for this example
    coef = np.polyfit(t_values[0:6], log_counts[0:6], 1)
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
plt.savefig('Entropy_simple_v2.png')
print("Plot saved as Distinguishable_Segments.png")