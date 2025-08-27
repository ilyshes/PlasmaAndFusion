#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Define logistic map
def logistic_map(x, a):
    return a * x * (1 - x)

# Parameters
a_values = [3.2, 4.0]  # Two values for side-by-side comparison
comms_values = ['Predictable development', 'Unpredictable development']
x0 = 0.2
n_iter = 50
delay = 100  # milliseconds per frame in GIF

plt.rcParams['text.usetex'] = True

# Precompute steps for cobweb plots
all_steps = []
vertical_steps = []
horizontal_steps = []
    
for a in a_values:
    x_vals = [x0]
    for _ in range(n_iter):
        x_vals.append(logistic_map(x_vals[-1], a))
    vertical_step = []
    horizontal_step = []
    for i in range(len(x_vals)-1):
        line_x = [x_vals[i], x_vals[i]]
        line_y  = [x_vals[i], x_vals[i+1]]
        vertical_step.append((line_x, line_y))  # vertical line
        
        line_x = [x_vals[i], x_vals[i+1]]
        line_y  =[x_vals[i+1], x_vals[i+1]]       
        horizontal_step.append((line_x, line_y))  # horizontal
  
    vertical_steps.append(vertical_step)
    horizontal_steps.append(horizontal_step)
    
    
# Prepare figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 7.8))
fig.suptitle(r'Cobweb Plot Animation $x_{n+1} = a x_n (1 - x_n)$  -- Logistic map', fontsize=20, fontweight='bold')

lines = [[] for _ in a_values]
points = []

# Initial static parts (maps and diagonals)
for idx, ax in enumerate(axs):
    a = a_values[idx]
    comms = comms_values[idx]
    x = np.linspace(0, 1, 400)
    ax.plot(x, logistic_map(x, a), 'k', linewidth=2)
    ax.plot(x, x, 'g--', linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x_n$', fontsize=20)
    ax.set_ylabel(r'$x_{n+1}$', fontsize=20)
    ax.set_title(f'a = {a}\n{comms}', fontsize=18)
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

# Prepare GIF frames in memory
frames = []
max_frames = max(len(steps) for steps in vertical_steps)

for frame in range(max_frames):
    for idx, ax in enumerate(axs):
        # Plot steps incrementally
        if frame < len(vertical_steps[idx]):
             # retriew data and plot vertical line
             vertical_step = vertical_steps[idx][frame]
             line_x = vertical_step[0]
             line_y = vertical_step[1]
             ax.plot(line_x, line_y, color='blue', linewidth=1)
             
             ax.plot(line_x[1], line_y[1], 'ro')  # point
             plt.pause(0.1)
             # retriew data and plot horizontal line
             horizontal_step = horizontal_steps[idx][frame]
             line_x = horizontal_step[0]
             line_y = horizontal_step[1]
             ax.plot(line_x, line_y, color='blue', linewidth=1)
           
             plt.pause(0.1)

    # Capture current figure as image in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    frames.append(Image.open(buf))

      # Optional for interactive preview

# Build GIF
frames[0].save('logistic_map.gif', save_all=True, append_images=frames[1:], duration=delay, loop=0)
print("GIF saved as logistic_map.gif")
