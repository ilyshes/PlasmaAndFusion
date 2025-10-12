#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  This rescales to 1920 px width, keeping aspect ratio
#  ffmpeg -r 14 -f image2 -s 1920x1080  -i "img/u_%05d.png" -vf "scale=1920:-2"   -vcodec libx264 -crf 25  -pix_fmt yuv420p  output.mp4

import pickle
import os
import io
import time as t
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import shutil
import re
from scipy.interpolate import RegularGridInterpolator
from src.field import DecayingTurbulence
from src.fluid import Fluid
from PIL import Image

def generate_initial_conditions_circle(num_points, x0, y0, R):
    """
    Generates a list of initial conditions on a circle in the phase space.
    """
    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_values = x0 + R * np.cos(thetas)
    y_values = y0 + R * np.sin(thetas)
    initial_conditions = [[x, y] for x, y in zip(x_values, y_values)]
    initial_conditions.append(initial_conditions[0])
    return initial_conditions
###############################################################################################

###############################################################################################

plt.close('all')

delay = 150  # ms per frame in GIF
points_coords = []

delta = 0.03
# Center the phase volume near a known point of interest (e.g., around x=1, y=0)
x0_val = 3.0 #5.0
y0_val = 4.0 # 0.8
Radius_initial_region = 0.03 # Keep the radius small for local linear approximation
num_points = 25 # Number of points defining the circular volume

initial_point_for_Jacobian = [x0_val, y0_val]

points_coords = generate_initial_conditions_circle(
    num_points=num_points, 
    x0 = x0_val, 
    y0 = y0_val, 
    R = Radius_initial_region # Keep the radius small for local linear approximation
)



Npoints = len(points_coords)
            
trajectories = []
for point in points_coords:
    traj = []
    traj.append(point)
    trajectories.append(traj)


trajectory_for_Jacobian = []
trajectory_for_Jacobian.append(initial_point_for_Jacobian)

##################################################################################

# Trace u and v for Jacobian

# Array for central point 
u_traj_0 = []
v_traj_0 = []
  
# Array for point x0 + dx
u_traj_1 = []
v_traj_1 = []

    
# Array for point y0 + dy
u_traj_2 = []
v_traj_2 = []

# J_numeric
J_numeric = []

# Trace lambda
lambda_1_trace = [] 
lambda_2_trace = [] 

output_dir = 'dat'

# Get a list of all files in the directory and sort them
file_list = sorted([f for f in os.listdir(output_dir) if f.endswith('.pkl')])




fig = plt.figure(figsize=(15.0, 9.6))
ax =  plt.subplot2grid((10, 10), (0, 0), colspan=5, rowspan = 7) # Top-left subplot
ax_zoom =  plt.subplot2grid((10, 10), (0, 5), colspan=5, rowspan = 7) # Top-right subplot
ax_Jacobian = plt.subplot2grid((10, 10), (7, 1), colspan=8,  rowspan = 3) # Bottom subplot spanning both columns
 



ax.set_title("Evolution of turbulent field and tracing a set of two\n initially close fluid particles", fontsize=18)

ax.set_xlabel("X", fontsize=18)
ax.set_ylabel("Y", fontsize=18)

ax_zoom.set_xlabel("X", fontsize=18)
ax_zoom.set_ylabel("Y", fontsize=18)

xmin, xmax = 0, 2*np.pi
ymin, ymax = 0, 2*np.pi
extent = [xmin, xmax, ymin, ymax]


with open('dat/u_00002.pkl', 'rb') as f:
    # Load the data from the current file
    data = pickle.load(f)
    wh = data['wh']



x_arr = data['x']
y_arr = data['y']

im = ax.imshow(np.fft.irfft2(wh, axes=(-2,-1)), extent=extent, norm=None, cmap="bwr")

ax.set_title("Evolution of turbulent field and tracing a set of two\n initially close fluid particles", fontsize=18)

ax.set_xlabel("X", fontsize=18)
ax.set_ylabel("Y", fontsize=18)

ax.set_aspect('equal', adjustable='box') # Keep the aspect ratio fixed for better shape representation

ax3 = fig.add_axes(ax.get_position(), frameon=False)  # Same position
ax3.set_xticks(ax.get_xticks())
ax3.set_xticks(ax.get_xticks())

ax3.set_xlim(ax.get_xlim())
ax3.set_ylim(ax.get_ylim())

ax3.set_position(ax.get_position())


frames = []  



plt.tight_layout()                  

file_list = file_list[0:5000] #  Temporary for test


   
# Loop through each file in the directory
for filename in file_list:
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'rb') as f:
            # Load the data from the current file
            data = pickle.load(f)
            
            # Add the loaded data to our list
            dt = data['time']
            u = data['u']
            v = data['v']
            x = data['x']
            y = data['y']
            wh = data['wh']
            it = data['it']
            
            
            # Set up interpolators for this step
            u_interp = RegularGridInterpolator((y_arr, x_arr), u, bounds_error=False, fill_value=None)
            v_interp = RegularGridInterpolator((y_arr, x_arr), v, bounds_error=False, fill_value=None)


            def traj_interp(traj):
                last_point = traj[-1]
                point = np.array(last_point)  # y, x order for interpolator
                u_val = u_interp(np.flip(point))
                v_val = v_interp(np.flip(point))
               
                x0 = last_point[0]
                y0 = last_point[1]
                # Euler update
                x0 += u_val[0] * dt
                y0 += v_val[0] * dt
                traj.append(  [x0, y0] )
            
            def traj_interp_with_Jacobian(traj):
                # Process the central point
                last_point = traj[-1]
                point = np.array(last_point)  # y, x order for interpolator
                u_val_0 = u_interp(np.flip(point))
                v_val_0 = v_interp(np.flip(point))
               
                u_traj_0.append(u_val_0[0])
                v_traj_0.append(v_val_0[0]) 
               
                x0 = last_point[0]
                y0 = last_point[1]
                # Euler update
                x0 += u_val_0[0] * dt
                y0 += v_val_0[0] * dt
                traj.append([x0, y0] )
                
                # Process the second  point x0 + dx, this is for Jacobian
                
                delta = 0.03
                point_plus_dx = np.array(  [point[0] + delta, point[1] ] )
                u_val_1 = u_interp(np.flip(point_plus_dx))
                v_val_1 = v_interp(np.flip(point_plus_dx))
                u_traj_1.append(u_val_1[0])
                v_traj_1.append(v_val_1[0])
                
                
                # Process the second  point y0 + dy, this is for Jacobian
                
                displaced_point = np.array(  [point[0] , point[1] + delta] )
                u_val_2 = u_interp(np.flip(displaced_point))
                v_val_2 = v_interp(np.flip(displaced_point))
                u_traj_2.append(u_val_2[0])
                v_traj_2.append(v_val_2[0])   
                
                dv_dx = float(  (v_val_1 - v_val_0)/delta )
                du_dx = float(  (u_val_1 - u_val_0)/delta )
                
                dv_dy = float(  (v_val_2 - v_val_0)/delta )
                du_dy = float(  (u_val_2 - u_val_0)/delta )
                
                J = np.array([[du_dx, du_dy], [dv_dx, dv_dy]])
                eig_val, eig_vec = np.linalg.eig(J)
                if(it % 5 == 0): 
                   print (eig_val)
                
                J_numeric.append(J)
                
                lambda_1 = eig_val[0]
                lambda_2 = eig_val[1]
                
                lambda_1_trace.append(lambda_1)
                lambda_2_trace.append(lambda_2)


            for trajectory in trajectories:
               
                traj_interp(trajectory)
            
            
            traj_interp_with_Jacobian(trajectory_for_Jacobian)    
           
           
            
        
            if(it % 100 == 0): 
                plt.pause(0.1)
                im.set_data(np.fft.irfft2(wh, axes=(-2,-1)))
                fig.canvas.draw()
                fig.canvas.flush_events()
                
                print(f"  - Successfully loaded '{filename}'")
               
           
                ax3.clear()
                ax_zoom.clear()
                ax_Jacobian.clear()
                
                # --- Plot Trajectories and Current Shape ---
                
                # --- Plot Trajectories and Current Shape ---
                x_region_initial = []
                y_region_initial = []
                
                x_region = []
                y_region = []
                        
                for trajectory in trajectories:   
                    trace_x = [p[0] for p in trajectory]
                    trace_y = [p[1] for p in trajectory]
                    
                    # Current boundary points
                    x_region_initial.append(trace_x[0])
                    y_region_initial.append(trace_y[0])
                    
                    x_region.append(trace_x[-1])
                    y_region.append(trace_y[-1])
                    
                    ax3.plot(trace_x, trace_y, '-', color='black', alpha=0.2,  zorder=1 )    
                 
                    # Plot the line connecting the red points (the deformed phase volume boundary)
                ax3.plot(x_region, y_region, '-', color='red', linewidth=2, zorder=3)
                ax3.plot(x_region_initial, y_region_initial, '-', color='green', linewidth=2, zorder=3)
                
                
                trace_x_jacobian = [p[0] for p in trajectory_for_Jacobian]
                trace_y_jacobian = [p[1] for p in trajectory_for_Jacobian]
                ax3.plot(trace_x_jacobian, trace_y_jacobian, '-', color='magenta', alpha=0.7,  zorder=1 )         
                
                ax3.set_xticks(ax.get_xticks())
                ax3.set_xticks(ax.get_xticks())
                ax3.set_xlim(ax.get_xlim())
                ax3.set_ylim(ax.get_ylim())
                ax3.tick_params(axis='x', labelsize=14)
                ax3.tick_params(axis='y', labelsize=14)
                ax3.set_position(ax.get_position())   
                
                
                
                ax_zoom.plot(x_region, y_region, '-', color='red', linewidth=2, zorder=3)
           
                ax_zoom.set_aspect('equal', adjustable='box') # Keep the aspect ratio fixed for better shape representation
    
                ax_zoom.set_xlabel("X", fontsize=18)
                ax_zoom.set_ylabel("Y", fontsize=18)
    
                    # --- Plot Jacobian Eigenvectors at the Central Point ---
                last_point_x = trace_x_jacobian[-1]
                last_point_y = trace_y_jacobian[-1]
                
                ax_zoom.plot(last_point_x, last_point_y, 'o', color='magenta', linewidth=5, zorder=3)
                
                
                
                
                radius = 0.3
                x_min = last_point_x - radius
                x_max = last_point_x + radius
                y_min = last_point_y - radius
                y_max = last_point_y + radius     
                
                
                
                shape_initial = generate_initial_conditions_circle(
                    num_points=num_points, 
                    x0 = last_point_x, 
                    y0 = last_point_y, 
                    R = Radius_initial_region
                )
                
                x_shape_initial = [p[0] for p in shape_initial]
                y_shape_initial = [p[1] for p in shape_initial]
    
                # Iniial shape of the region for reference
                ax_zoom.plot(x_shape_initial, y_shape_initial, '-', color='green', linewidth=2, zorder=3)
    
               
    
    
                  # Add a small buffer/padding
                padding_factor = 0.1 # 10% padding
                
                # Calculate range
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                # Add padding to the limits
                x_pad = x_range * padding_factor 
                y_pad = y_range * padding_factor 
            
                ax_zoom.set_xlim(x_min - x_pad, x_max + x_pad)
                ax_zoom.set_ylim(y_min - y_pad, y_max + y_pad)
                ax_zoom.set_aspect('equal', adjustable='box')
    
    
                ax_zoom.grid(True)
                ax_zoom.tick_params(axis='both', which='major', labelsize=14)
    
               
                ax_Jacobian.plot(lambda_1_trace)
                ax_Jacobian.plot(lambda_2_trace)
                ax_Jacobian.tick_params(axis='both', which='major', labelsize=14)
                ax_Jacobian.text(0.3, 0.95, r'Jacobian Eigenvalues along trajectory  ' , transform=ax_Jacobian.transAxes, 
                                    fontsize=15, color='black', 
                                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8, ec="gray"), 
                                    verticalalignment='top')
    
                ax_Jacobian.set_xlabel(r'Time $t$', fontsize=18)
                ax_Jacobian.set_ylabel(r'$\lambda$', fontsize=20)
                ax_Jacobian.grid(True)
        
                plt.tight_layout()   
      
                # Save frame
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                frames.append(Image.open(buf))
            





  



    
            # print(f"  - Successfully loaded '{filename}'")
    except (pickle.UnpicklingError, IOError) as e:
        print(f"  - Error loading file '{filename}': {e}")                     
                     
                     
                     
# The duration for the first n-1 frames is the original 'delay'
durations = [delay] * (len(frames) - 1)
# The last frame's duration is 3000 ms (3 seconds)
durations.append(1000)

  # # Save as GIF
frames[0].save('Jacobian evaluation.gif',
            save_all=True, append_images=frames[1:],
            duration=durations, loop=0)                     
                     
                     
                     
                     
                     