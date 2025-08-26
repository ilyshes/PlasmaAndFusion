#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 14:06:54 2025

@author: ilya
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# 1. Enable LaTeX rendering for all text in the plot
plt.rcParams['text.usetex'] = True

def logistic_map(x, a):
    return a * x * (1 - x)

# Parameters
a_values = [3.2, 4.0]  # Two values for side-by-side comparison
comms_values = ['Predictable development', 'Unpredictable development']  # Two values for side-by-side comparison
x0 = 0.2
n_iter = 50



font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }



# Prepare figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6.8))

fig.suptitle(r'Cobweb Plot Animation $x_{n+1} = a x_n (1 - x_n)$  -- Logistic map', fontsize=20, fontweight='bold')

# Precompute sequences and steps for both plots
all_steps = []
for a in a_values:
    x_vals = [x0]
    for _ in range(n_iter):
        x_vals.append(logistic_map(x_vals[-1], a))
    steps = []
    for i in range(len(x_vals)-1):
        # Vertical step
        steps.append(([x_vals[i], x_vals[i]], [x_vals[i], x_vals[i+1]]))
        # Horizontal step
        steps.append(([x_vals[i], x_vals[i+1]], [x_vals[i+1], x_vals[i+1]]))
    all_steps.append(steps)

# Setup each subplot
lines = []
points = []
for idx, ax in enumerate(axs):
    a = a_values[idx]
    comms = comms_values[idx]
    x = np.linspace(0, 1, 400)
    ax.plot(x, logistic_map(x, a), 'k', linewidth=2)
    ax.plot(x, x, 'k--', linewidth=2, color='green')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
   
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x_n$', fontsize=20)
    ax.set_ylabel(r'$x_{n+1}$', fontsize=20)
    ax.set_title(f'a = {a} \n {comms}', fontsize=20)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.grid(True, which='major', linestyle='-', linewidth=0.8)


   
    line, = ax.plot([], [], color='blue', linewidth=1)
    point, = ax.plot([], [], 'ro')
    lines.append(line)
    points.append(point)


# Animation function
lines_x = [[] for _ in a_values]
lines_y = [[] for _ in a_values]

def init():
    for line, point in zip(lines, points):
        line.set_data([], [])
        point.set_data([], [])
    return lines + points 

def update(frame):
    for idx in range(len(a_values)):
        N_iter = len(all_steps[idx])
        if frame < N_iter:
                     
            lines_x[idx].append(all_steps[idx][frame][0])
            lines_y[idx].append(all_steps[idx][frame][1])
            lines[idx].set_data(np.concatenate(lines_x[idx]),
                                np.concatenate(lines_y[idx]))
            points[idx].set_data(all_steps[idx][frame][0][1],
                                 all_steps[idx][frame][1][1])
    return lines + points 

# Max number of frames among both parameter sets
max_frames = max(len(steps) for steps in all_steps)

ani = animation.FuncAnimation(fig, update, frames=max_frames,
                              init_func=init, interval=100, blit=True, repeat=False)

plt.tight_layout()
plt.show()

# This is the line to save the animation as a GIF file
ani.save('my_animation.gif')

