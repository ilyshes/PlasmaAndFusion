import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from PIL import Image
import io
from matplotlib.quiver import Quiver
from matplotlib.text import Annotation

plt.close('all')

delay = 150  # ms per frame in GIF
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


def jacobian_matrix(x, y):
    """
    Calculates the Jacobian matrix of the system at a given point (x, y).
    
    Returns:
        np.ndarray: The 2x2 Jacobian matrix.
    """
    J11 = 3 - 2 * x - 2 * y
    J12 = -2 * x
    J21 = -y
    J22 = 2 - x - 2 * y
    return np.array([[J11, J12], [J21, J22]])


def solve_and_plot_system(initial_conditions_list, initial_point_for_Jacobian, time_span, title="Lotka–Volterra equations \n Dynamics of the selected sample points in the phase space"):
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.3, 7.5))
    #fig.suptitle(title, fontsize=18)
    #fig.suptitle(r'Phase Portrait of the Lotka–Volterra equations and trajectories of some sample initial points \\  $\frac{dx}{dt} = x(3 - x - 2y)$ \\ $\frac{dy}{dt} = y(2 - x - y)$', fontsize=18)
    
    #fig.suptitle(r'Phase Portrait of the Lotka–Volterra equations and trajectories of some sample initial points: \\ $\left\{ \begin{array}{l} \frac{dx}{dt} = x(3 - x - 2y) \\ \frac{dy}{dt} = y(2 - x - y) \end{array} \right.$', fontsize=18)
    fig.suptitle(r'Understanding Jacobian of the ODEs and its role in the Shape deformation of the sample phase space volume' + '\n' + r'Lotka–Volterra equations :  $\left\{ \begin{array}{l} \frac{dx}{dt} = x(3 - x - 2y) \\ \frac{dy}{dt} = y(2 - x - y) \end{array} \right.$', fontsize=18)

    
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
    ax2.set_title('Shape transformation of the selected phase space volume and its link to \n the eigenvalues and eigenvectors of the Jacobian', fontsize=18)
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
        

    solution_Jacobian = odeint(model, initial_point_for_Jacobian, time_span)
    x_solution_Jacobian = solution_Jacobian[:, 0]
    y_solution_Jacobian = solution_Jacobian[:, 1] 
        
    Iter_text_object = ax2.text(0.2, 2.0, '', fontsize=20, color='black')   
    frames = []    
    for j in range(2, len(x_solution)):    
        # --- Phase Portrait (ax2) ---
        
        print(j)
        plt.pause(1)
        
        # remove all previous lines
        for line in ax2.lines:
            line.remove()
            
        for artist in ax2.collections:
            if isinstance(artist, Quiver):
                artist.remove()   
                
        # Remove only annotations
        for txt in list(ax2.texts):  # make a copy of the list
            if isinstance(txt, Annotation):
                txt.remove()        
        
        # Plot initial positions of the circle
        for i, solution in enumerate(solutions):
            x_solution = solution[:, 0]
            y_solution = solution[:, 1]           
            ax2.plot(x_solution[0], y_solution[0], 'o', color='green', markersize=5)    
            
        # Plot all updated lines and actual positions 
        x_at_j = []
        y_at_j = []
        for i, solution in enumerate(solutions):
            x_solution = solution[:, 0]
            y_solution = solution[:, 1]           
            ax2.plot(x_solution[:j], y_solution[:j], '-', color='blue', alpha=0.3)
            ax2.plot(x_solution[j-1], y_solution[j-1], 'o', color='red', markersize=2)
            x_at_j.append(x_solution[j-1])
            y_at_j.append(y_solution[j-1])
        # Plot the line connecting the red points
        ax2.plot(x_at_j, y_at_j, '-', color='red', linewidth=2)
        
        
        
        # Tracing the Jacobian properties
        last_point_x = x_solution_Jacobian[j-1]
        last_point_y = y_solution_Jacobian[j-1]
        ax2.plot(x_solution_Jacobian[:j], y_solution_Jacobian[:j], '-', color='magenta', linewidth=2.5)
        
        # Emphasize the point of evaluation for the Jacobian
        ax2.plot(last_point_x, last_point_y, 'o', color='black', markersize=10, zorder=5)
        
        J = jacobian_matrix(last_point_x, last_point_y)
        eig_val, eig_vec = np.linalg.eig(J)
        
        r1_x =  eig_vec[0,0]
        r1_y =  eig_vec[1,0]
        ax2.quiver(last_point_x, last_point_y, r1_x, r1_y, 
                  color='black', 
                  scale=1, 
                  scale_units='xy', 
                  angles='xy',
                  headwidth=5, 
                  headlength=7,
                  width=0.008,
                  alpha=1)
        r2_x =  eig_vec[0,1]
        r2_y =  eig_vec[1,1]
        ax2.quiver(last_point_x, last_point_y, r2_x, r2_y, 
                  color='black', 
                  scale=1, 
                  scale_units='xy', 
                  angles='xy',
                  headwidth=5, 
                  headlength=7,
                  width=0.008,
                  alpha=1)
        lambda_1 = eig_val[0]
        lambda_2 = eig_val[1]
        Iter_text_object.set_text( f't = {j}'+ '\n' +
                                  r'Eigenvalues at  $\bullet$: ' +  f' $\lambda_1$ = {lambda_1:.2f},   $\qquad$   $\qquad$  $\lambda_2$ = {lambda_2:.2f}' + '\n' +
                                  r'Eigenvectors at $\bullet$: ' + f'$r_1$ = ( {r1_x:.2f}, {r1_y:.2f} ), $r_2$ = ( {r2_x:.2f}, {r2_y:.2f} )'
                                  )
        ax2.annotate(r'$r_2$', (last_point_x + r2_x, last_point_y + r2_y),
             textcoords="offset points", xytext=(5,5), ha='center',
             fontsize=14, color='black')
        ax2.annotate(r'$r_1$', (last_point_x + r1_x, last_point_y + r1_y),
             textcoords="offset points", xytext=(5,5), ha='center',
             fontsize=14, color='black')
        
        plt.show()

        
        
        
        
        
        
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
    frames[0].save('Shape transformation and Jacobian.gif',
                save_all=True, append_images=frames[1:],
                duration=durations, loop=0)
    

def generate_initial_conditions_circle(num_points, x0, y0, R):
    """
    Generates a list of initial conditions on a circle in the phase space.
    
    Args:
        num_points (int): The number of initial conditions to generate.
        x0 (float): The x-coordinate of the circle's center.
        y0 (float): The y-coordinate of the circle's center.
        R (float): The radius of the circle.
    
    Returns:
        list: A list of initial conditions [[x, y], ...].
    """
    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_values = x0 + R * np.cos(thetas)
    y_values = y0 + R * np.sin(thetas)
    initial_conditions = [[x, y] for x, y in zip(x_values, y_values)]
    return initial_conditions



# --- Example Usage ---

# Define a list of initial conditions
# Generate 5 initial conditions with x0 = 1.0 and y0 between 0 and 0.6
# Define a list of initial conditions

x0_val = 3
y0_val = 1.5
# initial_conditions_list = generate_initial_conditions_circle(num_points=20, x0_val, y0_val, R=0.1)
x0_val = 0.3
y0_val = 0.3
initial_conditions_list = generate_initial_conditions_circle(num_points=20, x0 = x0_val, y0 = y0_val, R=0.1)
initial_point_for_Jacobian = [x0_val, y0_val]


# Define the time span for the simulation
time_points = np.linspace(0, 3, 50)

# Solve the systems and plot the results on a single figure
solve_and_plot_system(initial_conditions_list, initial_point_for_Jacobian, time_points)
