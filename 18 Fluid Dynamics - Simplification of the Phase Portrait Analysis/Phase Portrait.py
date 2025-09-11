__author__ = "Ilya Shesterikov"
__copyright__ = "Ilya Shesterikov"
__license__ = "GPL"
__version__ = "1.0.1"
__email__  = "ilyshes@gmail.com"


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from PIL import Image
import io

plt.close('all')

delay = 200  # ms per frame in GIF
plt.rcParams['text.usetex'] = True

def model(z, t):
    """
    Defines the system of differential equations.
    
    Args:
        z (list): A list containing the current values of x and y.
        t (float): The current time.
        
    Returns:
        list: The derivatives [dx/dt, dy/dt].
    """
    x, y = z
    dxdt = x * (3 - x - 2 * y)
    dydt = y * (2 - x - y)
    return [dxdt, dydt]

def solve_and_plot_system(initial_conditions_list, time_span, title="Lotka–Volterra equations \n Dynamics of the selected sample points in the phase space"):
    """
    Solves the system of ODEs for a given set of initial conditions and time span,
    and then plots the results on a single figure.

    Args:
        initial_conditions_list (list): A list of lists, where each inner list [x0, y0]
                                         represents a set of initial populations.
        time_span (np.ndarray): A NumPy array representing the time points for the solution.
        title (str): The title for the plot.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.2))
    #fig.suptitle(title, fontsize=18)
    #fig.suptitle(r'Phase Portrait of the Lotka–Volterra equations and trajectories of some sample initial points \\  $\frac{dx}{dt} = x(3 - x - 2y)$ \\ $\frac{dy}{dt} = y(2 - x - y)$', fontsize=18)
    
    #fig.suptitle(r'Phase Portrait of the Lotka–Volterra equations and trajectories of some sample initial points: \\ $\left\{ \begin{array}{l} \frac{dx}{dt} = x(3 - x - 2y) \\ \frac{dy}{dt} = y(2 - x - y) \end{array} \right.$', fontsize=18)
    fig.suptitle(r'Phase Portrait of the Lotka–Volterra equations and trajectories of some sample initial points:' + '\n' + r'$\left\{ \begin{array}{l} \frac{dx}{dt} = x(3 - x - 2y) \\ \frac{dy}{dt} = y(2 - x - y) \end{array} \right.$', fontsize=18)

    
    ax1.set_xlim(-0.1, 3.5)
    ax1.set_ylim(-0.1, 2.5)
    # Set titles and labels for the time series plot
    ax1.set_title('Phase Portrait (X vs Y)', fontsize=16)
    ax1.set_xlabel(r'X', fontsize=16)
    ax1.set_ylabel('Y', fontsize=16)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    #ax1.legend(fontsize=8)
    ax1.grid(True)
    ax1.set_aspect('equal', adjustable='box')
    
    
    
    
    # Set titles and labels for the phase portrait
    ax2.set_title('Trajectories of selected  initial points in the phase space', fontsize=18)
    ax2.set_xlabel(r'X', fontsize=16)
    ax2.set_ylabel('Y', fontsize=16)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.set_xlim(-0.1, 3.5)
    ax2.set_ylim(-0.1, 2.5)
    #ax2.legend(fontsize=8)
    ax2.grid(True)
    ax2.set_aspect('equal', adjustable='box')
    
    
    # --- Plotting Vector Field ---
    x_min, x_max = ax2.get_xlim()
    y_min, y_max = ax2.get_ylim()
    
    # Create a grid of points
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 20),
                                 np.linspace(y_min, y_max, 20))
    
    # Calculate the derivatives at each point
    u = x_grid * (3 - x_grid - 2 * y_grid)
    v = y_grid * (2 - x_grid - y_grid)
    
    # Normalize the vectors for plotting
    # norm = np.sqrt(u**2 + v**2)
    # u_norm = u / norm
    # v_norm = v / norm

    # ax1.quiver(x_grid, y_grid, u_norm, v_norm, norm, cmap='plasma', scale=20, width=0.003)
    # ax1.quiverkey(ax1.quiver(x_grid, y_grid, u_norm, v_norm, norm, cmap='plasma', scale=20, width=0.003), 0.9, 0.9, 1, 
    #               label=r'Vector of magnitude 1', coordinates='axes')
    
    
    # Plot the streamlines
    ax1.streamplot(x_grid, y_grid, u, v, density=1.5, color=np.sqrt(u**2 + v**2), cmap='hot')
    
    
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.tight_layout()

    # Use a color map for different trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, len(initial_conditions_list)))
    solutions = []
    for i, initial_conditions in enumerate(initial_conditions_list):
        # Solve the ODEs for each set of initial conditions
        solution = odeint(model, initial_conditions, time_span)
        
        # Extract the solutions for x and y
        x_solution = solution[:, 0]
        y_solution = solution[:, 1]
        solutions.append(solution)
        
        # # --- Time Series Plot (ax1) ---
        # ax1.plot(time_span, x_solution, '-', color=colors[i])
        # ax1.plot(time_span, y_solution, '--', color=colors[i])
    Iter_text_object = ax2.text(0.2, 2.3, '', fontsize=20, color='black')   
    frames = []    
    for j in range(2, len(x_solution)):    
        # --- Phase Portrait (ax2) ---
        Iter_text_object.set_text( f't = {j}')
        print(j)
        plt.pause(1)
        for line in ax2.lines:
            line.remove()
        
        for i, solution in enumerate(solutions):
            x_solution = solution[:, 0]
            y_solution = solution[:, 1]           
            ax2.plot(x_solution[:j], y_solution[:j], '-', color='blue')
            ax2.plot(x_solution[j-1], y_solution[j-1], 'o', color='red')
        
        
        # Save frame
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(Image.open(buf))

   
   
   
    # The duration for the first n-1 frames is the original 'delay'
    durations = [delay] * (len(frames) - 1)
    # The last frame's duration is 3000 ms (3 seconds)
    durations.append(1000)

      # # Save as GIF
    frames[0].save('Phase Portrait.gif',
                save_all=True, append_images=frames[1:],
                duration=durations, loop=0)
    

def generate_initial_conditions_yran(num_points, fixed_x0, y_range):
    """
    Generates a list of initial conditions with a fixed x0 and random y0 values.
    
    Args:
        num_points (int): The number of initial conditions to generate.
        fixed_x0 (float): The fixed initial value for x.
        y_range (tuple): A tuple (min, max) defining the range for random y0 values.
    
    Returns:
        list: A list of initial conditions [[x0, y0], [x0, y1], ...].
    """
    #y_values = np.random.uniform(y_range[0], y_range[1], num_points)
    y_values = np.linspace(y_range[0], y_range[1], num_points)
    initial_conditions = [[fixed_x0, y] for y in y_values]
    return initial_conditions

def generate_initial_conditions_xran(num_points, fixed_y0, x_range):
    """
    Generates a list of initial conditions with a fixed x0 and random y0 values.
    
    Args:
        num_points (int): The number of initial conditions to generate.
        fixed_x0 (float): The fixed initial value for x.
        y_range (tuple): A tuple (min, max) defining the range for random y0 values.
    
    Returns:
        list: A list of initial conditions [[x0, y0], [x0, y1], ...].
    """
    #y_values = np.random.uniform(y_range[0], y_range[1], num_points)
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    initial_conditions = [[x, fixed_y0] for x in x_values]
    return initial_conditions


# --- Example Usage ---

# Define a list of initial conditions
# Generate 5 initial conditions with x0 = 1.0 and y0 between 0 and 0.6
initial_conditions_list = generate_initial_conditions_yran(num_points=5, fixed_x0=0.05, y_range=(0.05, 2.5)) \
     + generate_initial_conditions_yran(num_points=5, fixed_x0=3.4, y_range=(0.05, 2.5)) \
     + generate_initial_conditions_xran(num_points=5, fixed_y0=2.4, x_range=(0.05, 3.45)) \
     + generate_initial_conditions_xran(num_points=5, fixed_y0=0.05, x_range=(0.05, 3.45)) \
         + generate_initial_conditions_xran(num_points=5, fixed_y0=0.05, x_range=(0.01, 1.0)) 

initial_conditions_list.append( [0.02, 0.05] )
initial_conditions_list.append( [0.03, 0.05] )
initial_conditions_list.append( [0.01, 0.07] )
initial_conditions_list.append( [0.01, 0.1] )


# Define the time span for the simulation
time_points = np.linspace(0, 10, 50)

# Solve the systems and plot the results on a single figure
solve_and_plot_system(initial_conditions_list, time_points)
