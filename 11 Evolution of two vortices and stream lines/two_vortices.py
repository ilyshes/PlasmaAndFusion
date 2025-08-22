#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  This rescales to 1920 px width, keeping aspect ratio
#  ffmpeg -r 14 -f image2 -s 1920x1080  -i "img/u_%05d.png" -vf "scale=1920:-2"   -vcodec libx264 -crf 25  -pix_fmt yuv420p  output.mp4

#   ffmpeg -f image2 -framerate 5  -i "img/u_%05d.png" -i palette.png -lavfi "scale=800:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5"   output.gif


import time as t
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import shutil
import re
from scipy.interpolate import RegularGridInterpolator
from src.field import VortexPaar
from src.fluid import Fluid




def empty_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # delete file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # delete subdirectory
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


points = []
points_coords = []
delta = 0.1
points_coords.append([2, 4] )
points_coords.append([3, 4] )
points_coords.append([4, 4] )
points_coords.append([4, 2] )
points_coords.append([4, 3] )
points_coords.append([4, 3.5] )
points_coords.append([2, 2.5] )
points_coords.append([5, 3] )
points_coords.append([2.5, 2.5] )
points_coords.append([1.5, 3.5] )
points_coords.append([5.2, 2.5] )




for point in points_coords:
    points.append(point)
    points.append([point[0],  point[1]+delta])








Npoints = len(points)
            
trajectories = []
for point in points:
    trajectory = []
    trajectory.append(point)
    trajectories.append(trajectory)


# build fluid and solver
flow = Fluid(400, 400, 300, pad=1.)
flow.init_solver()
flow.init_field(VortexPaar,  kappa = 2)




x_arr = flow.x
y_arr = flow.y


plt.close()
print("Starting integration on field.\n")
start_time = t.time()
finish = 0.5

empty_folder('img')
empty_folder('dat')

idx_plot = 0


wh = flow.wh
plt.ion()
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)


ax.set_title("Evolution of two Vortices and Streamlines", fontsize=18)

ax.set_xlabel("X", fontsize=18)
ax.set_ylabel("Y", fontsize=18)



xmin, xmax = 0, 2*np.pi
ymin, ymax = 0, 2*np.pi
extent = [xmin, xmax, ymin, ymax]

im = ax.imshow(np.fft.irfft2(wh, axes=(-2,-1)), extent=extent, norm=None, cmap="bwr")

cb = fig.colorbar(im)
#ax.set_xticks([]); ax.set_yticks([])
plt.xticks(fontsize=14, color='blue')
plt.yticks(fontsize=14, color='red')
   

fig.canvas.manager.window.raise_()  # Bring to front (most backends)
plt.show()

plt.pause(0.1)

### add steamplot
flow.update()
x = flow.x
y = flow.y
u = flow.u
v = flow.v

ax2 = fig.add_axes(ax.get_position(), frameon=False)  # Same position

# Optional: turn off ticks on the overlay axis
ax2.set_xticks(ax.get_xticks())
ax2.set_xticks(ax.get_xticks())

# Make sure ax2 doesn't interfere with ax


u = np.flip(u, axis=0)
v = -np.flip(v, axis=0)
ax2.streamplot(x, y, u, v, zorder=1, density=2)

ax2.set_xlim(ax.get_xlim())
ax2.set_ylim(ax.get_ylim())

ax2.set_position(ax.get_position())
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)


ax3 = fig.add_axes(ax.get_position(), frameon=False)  # Same position
ax3.set_xticks(ax.get_xticks())
ax3.set_xticks(ax.get_xticks())

ax3.set_xlim(ax.get_xlim())
ax3.set_ylim(ax.get_ylim())

ax3.set_position(ax.get_position())









   # loop to solve
while(flow.time<=finish):
      flow.update()
    
      
      wh = flow.wh
      x = flow.x
      y = flow.y
      u = flow.u
      v = flow.v
      dt = flow.dt
 
      u = np.flip(u, axis=0)
      v = -np.flip(v, axis=0)
         # Set up interpolators for this step
      u_interp = RegularGridInterpolator((y_arr, x_arr), u, bounds_error=False, fill_value=None)
      v_interp = RegularGridInterpolator((y_arr, x_arr), v, bounds_error=False, fill_value=None)
    
      for trajectory in trajectories:
            last_point = trajectory[-1]
          
            point = np.array(last_point)  # y, x order for interpolator
            u_val = u_interp(np.flip(point))
            v_val = v_interp(np.flip(point))
          
            x0 = last_point[0]
            y0 = last_point[1]
           # Euler update
            x0 += u_val[0] * dt
            y0 += v_val[0] * dt
     
            trajectory.append([x0, y0] )
 


      if(flow.it % 100 == 0):
            idx_plot+=1
            plt.pause(0.1)
            ax2.clear()
            ax2.streamplot(x, y, u, v, zorder=1, density=2)
            ax2.set_xticks(ax.get_xticks())
            ax2.set_xticks(ax.get_xticks())
            ax2.set_xlim(ax.get_xlim())
            ax2.set_ylim(ax.get_ylim())
            ax2.tick_params(axis='x', labelsize=14)
            ax2.tick_params(axis='y', labelsize=14)
            
            ax2.text(0.45, 1.10, f"Frame #{idx_plot}", transform=ax2.transAxes,
            fontsize=18, color='black', verticalalignment='top')
            
            
            
            im.set_data(np.fft.irfft2(wh, axes=(-2,-1)))
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(1e-9)
 
            print("Iteration \t %d, time \t %f, time remaining \t %f. TKE: %f, ENS: %f" %(flow.it,
                   flow.time, finish-flow.time, flow.tke(), flow.enstrophy()))
   

            ax3.clear()
            for trajectory in trajectories[::2]:   
                trace_x = [p[0] for p in trajectory]
                trace_y = [p[1] for p in trajectory]
                ax3.plot(trace_x, trace_y, lw = 4,  color='black', zorder=1 )    
            
                
            for trajectory in trajectories[1::2]:   
                trace_x = [p[0] for p in trajectory]
                trace_y = [p[1] for p in trajectory]
                ax3.plot(trace_x, trace_y, lw = 4,  color='green', zorder=1 )                           
            
            ax3.set_xticks(ax.get_xticks())
            ax3.set_xticks(ax.get_xticks())
            ax3.set_xlim(ax.get_xlim())
            ax3.set_ylim(ax.get_ylim())
            ax3.tick_params(axis='x', labelsize=14)
            ax3.tick_params(axis='y', labelsize=14)
            ax3.set_position(ax.get_position())
            
            

            
            plt.savefig('img/u_' + str(idx_plot).zfill(5) + '.png',dpi=500)
                
    
    
#   flow.run_live(finish, every=200)        
 #  flow.display()
 #  fig = plt.gcf()
 #  plt.savefig('img/u_end.png',dpi=500)
   # flow.run_live(finish, every=200)

end_time = t.time()
print("\nExecution time for %d iterations is %f seconds." %(flow.it, end_time-start_time))
   
  
 #             divergence of two close points                     
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)    
        
for idx in range(0, Npoints,2):
    trace_x_1 = np.array(      [p[0] for p in trajectories[idx]]  )
    trace_y_1 = np.array(      [p[1] for p in trajectories[idx]]  )
    
    trace_x_2 = np.array(      [p[0] for p in trajectories[idx+1]]  )
    trace_y_2 = np.array(      [p[1] for p in trajectories[idx+1]]  )
    
    distance = np.sqrt( (trace_x_1 - trace_x_2)**2 + (trace_y_1 - trace_y_2)**2 )
    
    ax.plot(distance, lw = 2,  color='red')           
                 
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     