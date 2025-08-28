#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 09:10:53 2025

@author: ilya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Logistic map function
def logistic_map(x, a):
    return a * x * (1 - x)

# Parameters
a_value = 4.0
comms_value = 'Two trajectories'
x0_1 = 0.2
x0_2 = 0.201
n_iter = 10
delay = 300  # ms per frame in GIF

plt.rcParams['text.usetex'] = True

# Function to compute steps for cobweb plot
def compute_steps(x0, a, n_iter):
    x_vals = [x0]
    for _ in range(n_iter):
        x_vals.append(logistic_map(x_vals[-1], a))

    vertical_steps = []
    horizontal_steps = []
    for i in range(len(x_vals) - 1):
        # Vertical line: (x_i, x_i) → (x_i, x_{i+1})
        line_x = [x_vals[i], x_vals[i]]
        line_y = [x_vals[i], x_vals[i+1]]
        vertical_steps.append((line_x, line_y))

        # Horizontal line: (x_i, x_{i+1}) → (x_{i+1}, x_{i+1})
        line_x = [x_vals[i], x_vals[i+1]]
        line_y = [x_vals[i+1], x_vals[i+1]]
        horizontal_steps.append((line_x, line_y))

    return vertical_steps, horizontal_steps, x_vals

# Compute steps and sequences for both points
steps1_v, steps1_h, x_vals_1 = compute_steps(x0_1, a_value, n_iter)
steps2_v, steps2_h, x_vals_2 = compute_steps(x0_2, a_value, n_iter)

# Prepare figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle(r'Lyapunov divergence', fontsize=20, fontweight='bold')

# --- Left subplot: Cobweb plot ---
x = np.linspace(0, 1, 400)
ax1.plot(x, logistic_map(x, a_value), 'k', linewidth=2)
ax1.plot(x, x, 'g--', linewidth=2)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel(r'$x_n$', fontsize=16)
ax1.set_ylabel(r'$x_{n+1}$', fontsize=16)
ax1.set_title(f'a = {a_value}\n{comms_value}', fontsize=16)
ax1.grid(True, linestyle='--', linewidth=0.5)
ax1.tick_params(axis='both', which='major', labelsize=12)


# Line for dynamic update in subplot 2
distances = []
distance_line, = ax2.plot([], [], color='purple', linewidth=2, marker='o')


# --- Right subplot: Distance evolution ---
ax2.set_xlim(0, n_iter)
ax2.set_ylim(0, 1)  # Use log scale if you like
ax2.set_xlabel(r'n', fontsize=16)
ax2.set_ylabel(r'$|x^{(1)}_n - x^{(2)}_n|$', fontsize=16)
ax2.set_title('Lyapunov exponent -- \n divergence of two trajectories', fontsize=18)
ax2.grid(True, linestyle='--', linewidth=0.5)
ax2.tick_params(axis='both', which='major', labelsize=12)


frames = []

# --- Incremental update ---
for frame in range(n_iter):
    # Cobweb: first trajectory (blue + red point)
    line_x, line_y = steps1_v[frame]
    ax1.plot(line_x, line_y, color='blue', linewidth=1)
    ax1.plot(line_x[1], line_y[1], 'ro', markersize=4)

    line_x, line_y = steps1_h[frame]
    ax1.plot(line_x, line_y, color='blue', linewidth=1)

    # Cobweb: second trajectory (orange + green point)
    line_x, line_y = steps2_v[frame]
    ax1.plot(line_x, line_y, color='orange', linewidth=1)
    ax1.plot(line_x[1], line_y[1], 'go', markersize=4)

    line_x, line_y = steps2_h[frame]
    ax1.plot(line_x, line_y, color='orange', linewidth=1)

    # Update distance subplot
    current_distance = abs(x_vals_1[frame+1] - x_vals_2[frame+1])
    distances.append(current_distance)
    distance_line.set_data(range(len(distances)), distances)
    ax2.relim()
    ax2.autoscale_view(scalex=False, scaley=True)

    plt.pause(0.3)

    # Save frame
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    frames.append(Image.open(buf))

# Save as GIF
frames[0].save('logistic_map_with_divergence.gif',
               save_all=True, append_images=frames[1:],
               duration=delay, loop=0)

print("GIF saved as logistic_map_with_divergence.gif")
