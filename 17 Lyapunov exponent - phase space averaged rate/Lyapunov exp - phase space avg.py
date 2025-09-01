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
import random

# Logistic map function
def logistic_map(x, a):
    return a * x * (1 - x)

# Parameters
a_value = 4.0
comms_value = 'Two trajectories'
gap = 1e-14
x0_1 = 0.2
x0_2 = x0_1 + gap
n_iter = 45
delay = 60  # ms per frame in GIF

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



# ORGANIZE EVALUATIONS WITHIN THE LOOP
# ------------------------------------
# Define the pairs of initial values in a list
# Number of random pairs you want to generate
num_pairs = 10

# Create a list to hold the randomly generated pairs
initial_pairs = []

for _ in range(num_pairs):
    # Choose a random initial value for the first point between 0 and 0.999
    # This ensures the second point (x0_a + 0.001) will not exceed 1.0
    x0_a = random.uniform(0.05, 0.48)
    x0_b = x0_a + gap
    initial_pairs.append((x0_a, x0_b))

# Create a list to hold the distance arrays
all_distances = []

# Loop through the initial pairs to compute trajectories and distances
for x0_a, x0_b in initial_pairs:
    _, _, x_vals_a = compute_steps(x0_a, a_value, n_iter)
    _, _, x_vals_b = compute_steps(x0_b, a_value, n_iter)
    distances = np.abs(np.array(x_vals_a) - np.array(x_vals_b))
    all_distances.append(distances)
# ------------------------------------


# Prepare figure with two subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle(r'Understanding Lyapunov exponent', fontsize=20, fontweight='bold')




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
distance_line_log, = ax3.plot([], [], color='purple', linewidth=2, marker='o',  alpha=0.6)


# --- Right subplot: Distance evolution ---
ax2.set_xlim(0, n_iter)
ax2.set_ylim(1e-16, 1)  # Use log scale if you like
ax2.set_xlabel(r'n', fontsize=16)
ax2.set_ylabel(r'$|x^{(1)}_n - x^{(2)}_n|$', fontsize=16)
ax2.set_title('Lyapunov exponent -- \n divergence of two trajectories', fontsize=18)
ax2.grid(True, linestyle='--', linewidth=0.5)
ax2.tick_params(axis='both', which='major', labelsize=12)



# --- Right subplot: Distance evolution ---
ax3.set_xlim(0, n_iter)
#ax3.set_yscale('log')
#ax3.set_ylim(1e-16, 1)  # Use log scale if you like
ax3.set_xlabel(r'n', fontsize=16)
ax3.set_ylabel(r'$ln( |x^{(1)}_n - x^{(2)}_n| )$', fontsize=16)
ax3.set_title('Evaluation of the Lyapunov exponent', fontsize=18)
ax3.grid(True, linestyle='--', linewidth=0.5)
ax3.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

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
    distance_line_log.set_data(range(len(distances)), np.log(distances))
    
    ax2.relim()
    ax2.autoscale_view(scalex=False, scaley=True)

    ax3.relim()
    ax3.autoscale_view(scalex=False, scaley=True)


    plt.pause(0.03)

    # Save frame
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    frames.append(Image.open(buf))


# A function to generate a random hexadecimal color code
def get_random_color():
    # Generate a random integer from 0 to 0xFFFFFF (16777215)
    hex_color_int = random.randint(0, 0xFFFFFF)
    # Convert to hex string and pad with zeros to ensure 6 characters
    return f'#{hex_color_int:06x}'


# --- Plot the new, looped distances



for i, dists in enumerate(all_distances):
    # Get a new random color for each line
    random_color = get_random_color()
    ax2.plot(range(len(dists)), dists, color=random_color, linewidth=2, marker='o')
    ax3.plot(range(len(dists)), np.log( dists), color=random_color, linewidth=2, marker='o', alpha=0.3)


distances_array =  np.log ( np.array(all_distances)  )

# Calculate the mean of each column (i.e., the average distance at each iteration)
average_distances = np.mean(distances_array, axis=0)

ax3.plot(range(len(dists)), average_distances, color='black', linewidth=2, marker='o')


##########################################################################
# Perform a linear fit on the data points.
####################################################################


# Create a sequence of iteration numbers (n) from 0 to n_iter
n_vals = np.arange(len(average_distances))

# Perform a linear fit on the data points.
# The result is an array containing the slope and y-intercept of the best-fit line.
# The slope of this line is an estimate of the Lyapunov exponent.
slope, intercept = np.polyfit(n_vals, average_distances, 1)

# Generate y-values for the fitted line
# y = mx + c, where m is the slope and c is the intercept
fitted_line = slope * n_vals + intercept

print(f"Estimated Lyapunov exponent (slope): {slope}")
print(f"Y-intercept: {intercept}")

# Plot the interpolated line on the ax3 subplot
ax3.plot(n_vals, fitted_line, '-', color='black', linewidth=6,marker='o', label=f'Linear Fit (slope={slope:.4f})')
ax3.legend()

# Add the text annotation showing the Lyapunov exponent
lyapunov_text = f"Estimated Lyapunov exponent   \n $\\lambda \\approx {slope:.4f}$"
ax3.text(0.08, 0.85, lyapunov_text, transform=ax3.transAxes, fontsize=18, verticalalignment='top', fontweight='bold')

plt.pause(3)

buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
frames.append(Image.open(buf))

# The duration for the first n-1 frames is the original 'delay'
durations = [delay] * (len(frames) - 1)
# The last frame's duration is 3000 ms (3 seconds)
durations.append(3000)

# # Save as GIF
frames[0].save('Lyapunov exp - phase space avg.gif',
                save_all=True, append_images=frames[1:],
                duration=durations, loop=0)

# print("GIF saved as logistic_map_with_divergence.gif")
