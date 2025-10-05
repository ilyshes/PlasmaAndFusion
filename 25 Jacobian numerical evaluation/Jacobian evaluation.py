import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from PIL import Image
import io
from matplotlib.quiver import Quiver
from matplotlib.text import Annotation
from scipy.interpolate import RegularGridInterpolator

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

def f_x(x, y):

    result = x * (3 - x - 2 * y)
  
    return result

def f_y(x, y):

    result = y * (2 - x - y)
  
    return result


def J00(x, y):
    """
    Calculates the Jacobian matrix of the system at a given point (x, y).
    
    Returns:
        np.ndarray: The 2x2 Jacobian matrix.
    """
    J11 = 3 - 2 * x - 2 * y
  
    return J11

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


def solve_and_plot_system(initial_condition, time_span, title="Lotka–Volterra equations \n Dynamics of the selected sample points in the phase space"):
    """
    Solves the system of ODEs for a given set of initial conditions and time span,
    and then plots the results on a single figure.

    Args:
        initial_condition (list): A list of lists, where each inner list [x0, y0]
                                         represents a set of initial populations.
        time_span (np.ndarray): A NumPy array representing the time points for the solution.
        title (str): The title for the plot.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.3, 7.5))
    #fig.suptitle(title, fontsize=18)
    #fig.suptitle(r'Phase Portrait of the Lotka–Volterra equations and trajectories of some sample initial points \\  $\frac{dx}{dt} = x(3 - x - 2y)$ \\ $\frac{dy}{dt} = y(2 - x - y)$', fontsize=18)
    
    #fig.suptitle(r'Phase Portrait of the Lotka–Volterra equations and trajectories of some sample initial points: \\ $\left\{ \begin{array}{l} \frac{dx}{dt} = x(3 - x - 2y) \\ \frac{dy}{dt} = y(2 - x - y) \end{array} \right.$', fontsize=18)
    fig.suptitle(r'Comparison of the theoretical and numerical Jacobian estimation' + '\n' + r'Lotka–Volterra equations :  $\left\{ \begin{array}{l} \frac{dx}{dt} = x(3 - x - 2y) \\ \frac{dy}{dt} = y(2 - x - y) \end{array} \right.$', fontsize=18)

    
    #ax2.set_xlim(2, 3.5)
    #ax2.set_ylim(-5, 5)
    # Set titles and labels for the time series plot
    ax2.set_title('Comparison of the theoretical and numerical eigenvalues of Jacobian ', fontsize=16)
    ax2.set_xlabel(r't [a.u.]', fontsize=16)
    ax2.set_ylabel('Y', fontsize=16)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    #ax1.legend(fontsize=8)
    ax2.grid(True)
    
    # # Dynamic y-limits, but cap the min/max for early simulation stability
    ax2.set_ylim(-10, 2)
    
    #ax2.set_aspect('equal', adjustable='box')
    
    
    
    
    # Set titles and labels for the phase portrait
    ax1.set_title('Shape transformation of the selected phase space volume and its link to \n the eigenvalues and eigenvectors of the Jacobian', fontsize=18)
    ax1.set_xlabel(r'X', fontsize=16)
    ax1.set_ylabel('Y', fontsize=16)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.set_xlim(-0.1, 3.5)
    ax1.set_ylim(-0.1, 2.5)
    #ax2.legend(fontsize=8)
    ax1.grid(True)
    ax1.set_aspect('equal', adjustable='box')
    
    
    
    plt.tight_layout()

    
        

    solution_Jacobian = odeint(model, initial_condition, time_span)
    x_solution_Jacobian = solution_Jacobian[:, 0]
    y_solution_Jacobian = solution_Jacobian[:, 1] 
        
    Iter_text_object = ax1.text(0.2, 2.0, '', fontsize=20, color='black')   
    frames = []  
    # List of stored eigenvalues along the trajectory
    lambda_1_trace = [] 
    lambda_2_trace = [] 
    
    lambda_1_num_trace = [] 
    lambda_2_num_trace = [] 
    
    
        
    x_min =  0
    x_max =  3.5
    
    y_min =  0
    y_max =  4.0
    Nx = 500
    Ny = 700
    
    # Create a grid of points
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, Nx),
                                 np.linspace(y_min, y_max, Ny))
    
        
    x_arr = np.linspace(x_min, x_max, Nx)
    y_arr = np.linspace(y_min, y_max, Ny)   
    
    
    # Calculate the F matrix at each point
    fx_theor = x_grid * (3 - x_grid - 2 * y_grid)
    fy_theor = y_grid * (2 - x_grid - y_grid)
    
    # Set up interpolators for this step
    u_interp = RegularGridInterpolator((y_arr, x_arr), fx_theor, bounds_error=False, fill_value=None)
    v_interp = RegularGridInterpolator((y_arr, x_arr), fy_theor, bounds_error=False, fill_value=None)    
    
    
    u_val_0_arr = []
    v_val_0_arr = []
    for i in range(len(x_solution_Jacobian)): 
        point = np.array(  [x_solution_Jacobian[i], y_solution_Jacobian[i] ] )
        u_val = u_interp(np.flip(point))
        v_val = v_interp(np.flip(point))
        u_val_0_arr.append(u_val[0])
        v_val_0_arr.append(v_val[0])

    delta = 0.01


  #  Distant in the X direction

    u_val_1_arr = []
    v_val_1_arr = []
    for i in range(len(x_solution_Jacobian)):
        point = np.array(  [x_solution_Jacobian[i] + delta, y_solution_Jacobian[i] ] )
        u_val = u_interp(np.flip(point))
        v_val = v_interp(np.flip(point))
        u_val_1_arr.append(u_val[0])
        v_val_1_arr.append(v_val[0])
        
  #  Distant in the Y direction        
    u_val_2_arr = []
    v_val_2_arr = []
    for i in range(len(x_solution_Jacobian)):
        point = np.array(  [x_solution_Jacobian[i] , y_solution_Jacobian[i] + delta ] )
        u_val = u_interp(np.flip(point))
        v_val = v_interp(np.flip(point))
        u_val_2_arr.append(u_val[0])
        v_val_2_arr.append(v_val[0])
    
 
    v_val_2_arr = np.array(v_val_2_arr)     
    v_val_1_arr = np.array(v_val_1_arr)
    v_val_0_arr = np.array(v_val_0_arr)

    u_val_2_arr = np.array(u_val_2_arr)
    u_val_1_arr = np.array(u_val_1_arr)
    u_val_0_arr = np.array(u_val_0_arr)

     
    dv_dx = (v_val_1_arr - v_val_0_arr)/delta
    du_dx = (u_val_1_arr - u_val_0_arr)/delta
    
    dv_dy = (v_val_2_arr - v_val_0_arr)/delta
    du_dy = (u_val_2_arr - u_val_0_arr)/delta
    
   
    
   
    # Plot both eigenvalues
    lambda1_th, = ax2.plot([], [], '-', color='yellow', markersize=10, lw=6, zorder=5, label=f'Theoretical $\lambda_1$')
    lambda2_th, = ax2.plot([], [], '-', color='blue',  markersize=10, lw=6, zorder=5, label=f'Theoretical $\lambda_2$')
    
    lambda1_num, = ax2.plot([], [], '-', color='black',  markersize=10, lw=2, zorder=5, label=f'Numerical $\lambda_1$')
    lambda2_num, = ax2.plot([], [], '-', color='red',  markersize=10, lw=2, zorder=5, label=f'Numerical $\lambda_2$') 
   
   
    for j in range(2, len(x_solution_Jacobian)):    
        # --- Phase Portrait (ax2) ---
        
        print(j)
        plt.pause(0.01)
        
        # remove all previous lines
        for line in ax1.lines:
            line.remove()
            
        for artist in ax1.collections:
            if isinstance(artist, Quiver):
                artist.remove()   
                
        # Remove only annotations
        for txt in list(ax1.texts):  # make a copy of the list
            if isinstance(txt, Annotation):
                txt.remove()        
        
       
        for txt in list(ax2.texts):  # make a copy of the list
            if isinstance(txt, Annotation):
                txt.remove()         
        
        
        
        # Tracing the Jacobian properties
        last_point_x = x_solution_Jacobian[j]
        last_point_y = y_solution_Jacobian[j]
        
       
        J_numeric = np.array([[du_dx[j], du_dy[j]], [dv_dx[j], dv_dy[j]]])
        
        ax1.plot(x_solution_Jacobian[:j+1], y_solution_Jacobian[:j+1], '-', color='magenta', linewidth=2.5)
        
        # Emphasize the point of evaluation for the Jacobian
        ax1.plot(last_point_x, last_point_y, 'o', color='black', markersize=10, zorder=5)
        
        # Analytic Jacobian evaluation
        J = jacobian_matrix(last_point_x, last_point_y)
        eig_val, eig_vec = np.linalg.eig(J)
        
        
        eig_val_num, eig_vec_num = np.linalg.eig(J_numeric)
        
        
        # Numeric Jacobian evaluation
        # J_num = np.array([[dfx_dx[j], dfx_dy[j]], [dfy_dx[j], dfy_dy[j]]])
        # eig_val_num, eig_vec_num = np.linalg.eig(J_num)
        # lambda_1_num = eig_val_num[0]
        # lambda_2_num = eig_val_num[1]
        # lambda_1_num_trace.append(J_num[0,1])
        # lambda_2_num_trace.append(lambda_2_num)
        
        
        
        r1_x =  eig_vec[0,0]
        r1_y =  eig_vec[1,0]
        ax1.quiver(last_point_x, last_point_y, r1_x, r1_y, 
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
        ax1.quiver(last_point_x, last_point_y, r2_x, r2_y, 
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
        
        lambda_1_num = eig_val_num[0]
        lambda_2_num = eig_val_num[1]
        
        lambda_1_trace.append(lambda_1)
        lambda_2_trace.append(lambda_2)
        
        lambda_1_num_trace.append(lambda_1_num)
        lambda_2_num_trace.append(lambda_2_num)
        
        
        Iter_text_object.set_text( f't = {j}'+ '\n' +
                                  r'Eigenvalues at  $\bullet$: ' +  f' $\lambda_1$ = {lambda_1:.2f},   $\qquad$   $\qquad$  $\lambda_2$ = {lambda_2:.2f}' + '\n' +
                                  r'Eigenvectors at $\bullet$: ' + f'$r_1$ = ( {r1_x:.2f}, {r1_y:.2f} ), $r_2$ = ( {r2_x:.2f}, {r2_y:.2f} )'
                                  )
        ax1.annotate(r'$r_2$', (last_point_x + r2_x, last_point_y + r2_y),
             textcoords="offset points", xytext=(5,5), ha='center',
             fontsize=14, color='black')
        ax1.annotate(r'$r_1$', (last_point_x + r1_x, last_point_y + r1_y),
             textcoords="offset points", xytext=(5,5), ha='center',
             fontsize=14, color='black')
        
        
#        1. Get the length of the list
        N = len(lambda_1_trace) # N = 7

        # 2. Create the numerical time vector
        # This generates N evenly spaced points from 0 up to (N-1) * dt
        # For N=7 and dt=0.01, the max value is 6 * 0.01 = 0.06
        time_vector = np.linspace(start=0, stop=(N - 1) * 1, num=N)
        ax2.set_xlim(0, N)

        # Plot both eigenvalues
        lambda1_th.set_data(time_vector, lambda_1_trace )
        lambda2_th.set_data(time_vector, lambda_2_trace )
        
        lambda1_num.set_data(time_vector, lambda_1_num_trace )
        lambda2_num.set_data(time_vector, lambda_2_num_trace )
        ax2.legend(loc='upper right', fontsize=16)
        
        # ax2.plot(lambda_1_trace, color='red', markersize=10, zorder=5)
       # ax2.plot(lambda_1_num_trace, color='blue', markersize=10, zorder=5)
        


        
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
    frames[0].save('Jacobian evaluation.gif',
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

x0_val = 2
y0_val = 2.2
# initial_conditions_list = generate_initial_conditions_circle(num_points=20, x0_val, y0_val, R=0.1)
#x0_val = 0.3
#y0_val = 0.3
initial_condition  = [x0_val, y0_val]


# Define the time span for the simulation
time_points = np.linspace(0, 0.5, 100)
dt = time_points[2] - time_points[1]

# Solve the systems and plot the results on a single figure
solve_and_plot_system(initial_condition,  time_points)
