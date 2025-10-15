import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

plt.rcParams['text.usetex'] = True

# --- Modified Functions for Topological Pressure ---

# Logistic map definition
def logistic_map(x, r=4.0):
    return r * x * (1 - x)

# Generate orbit segment of length t
def orbit_segment(x0, t, r=4.0):
    seg = np.empty(t)
    x = x0
    for i in range(t):
        seg[i] = x
        x = logistic_map(x, r)
    return seg

# The absolute value of the derivative of the logistic map f(x) = r*x*(1-x)
def derivative_logistic_map_abs(x, r=4.0):
    return np.abs(r * (1 - 2 * x))

# Lyapunov potential function (instantaneous weight)
# This calculates the instantaneous weight: beta * log|f'(x)|
def lyapunov_potential_instantaneous(x, r=4.0, beta=1.0):
    deriv = derivative_logistic_map_abs(x, r)
    # Small value to prevent issues with log(0) at the critical point x=0.5
    deriv[deriv < 1e-10] = 1e-10 
    return beta * np.log(deriv)

# Compute the topological pressure approximation P_t(beta, eps)
# Segments is the array of trajectory points (num_initial x t)
# Weights is the array of instantaneous Lyapunov potential values (num_initial x t)
def compute_pressure_approx(segments, weights, eps):
    n = segments.shape[0] # Number of initial conditions
    t = segments.shape[1] # Segment length
    used = np.zeros(n, dtype=bool)
    
    # Store the accumulated potential sum (S_t * phi) for each *distinguishable* orbit
    accumulated_potentials = [] 
    
    for i in range(n):
        if used[i]:
            continue
            
        # 1. Accumulated potential S_t * phi = sum_i beta * log|f'(x_i)|
        S_t_phi = np.sum(weights[i, :])
        accumulated_potentials.append(S_t_phi)      # It is like the energy term in spin system in the magnetic field
        #accumulated_potentials.append(0)           # Cancel potential, ignore energy term, it is like spin system without magnetic field 
        
        
        # 2. Find all segments that are eps-close to segment i (distinguishable block)
        # The distinguishable logic remains based on the segments (points)
        dists = np.max(np.abs(segments - segments[i]), axis=1)
        within = dists <= eps
        used = used | within
        
    # 3. Calculate the Partition Function Z_t = sum_S exp(S_t * phi)
    Z_t = np.sum(np.exp(accumulated_potentials))
    
    # 4. Calculate the pressure P_t = (1/t) * log(Z_t)
    if Z_t > 0 and t > 0:
        pressure_approx = (1.0 / t) * np.log(Z_t)
    else:
        pressure_approx = np.nan 
        
    return pressure_approx, Z_t

# --- Parameters ---
r = 4.0
t_max = 20 # Segment length up to 20
np.random.seed(0)

# Potential parameter
beta = 1.0 

Number_of_points = 2000
configs = [
    {'eps': 0.2, 'num_initial': Number_of_points, 'label': r'$\epsilon$=0.1, $\beta$=1.0', 'color': 'darkred'}]

# Create a figure with a single subplot for the pressure
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Customize the subplot
ax[0].set_xlabel('Segment length t' , fontsize=16)
ax[0].set_ylabel(r'Partition function $Z_t(\beta=1.0, \epsilon)$', fontsize=16)
ax[0].set_title(r'Partition function $P(\beta=1.0)$ for Logistic Map ($r=4.0$)',  fontsize=15)
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
ax[0].grid(True)


# Customize the right subplot
ax[1].set_xlabel('Segment length t' , fontsize=16)
#ax[1].set_ylabel('log(number of distinguishable segments)', fontsize=16)
ax[1].set_title('Logarithmic Scale',  fontsize=15)
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
ax[1].set_yscale('log')
ax[1].grid(True, which='both')


# The theoretical value for P(1) for the r=4 logistic map (conjugate to the doubling map)
# is P(1) = h_top(f) = log(2)
theoretical_pressure_P1 = np.log(2.0)


for config in configs:
    eps = config['eps']
    num_initial = config['num_initial']
    
    # Sample random initial conditions
    initials = np.random.rand(num_initial) * 0.5 + 0.05

    # Pre-compute full trajectories (points) and instantaneous weights
    
    # 1. Trajectories: (num_initial, t_max)
    trajectories = np.array([orbit_segment(x0, t_max, r) for x0 in initials])

    # 2. Instantaneous Weights: (num_initial, t_max)
    instantaneous_weights = lyapunov_potential_instantaneous(trajectories, r=r, beta=beta)


    # Compute pressure approximations
    t_values = []
    pressures = []
    Zt_values = []
    
    
    
    for t in range(1, t_max + 1):
        # Slice the trajectories and weights up to segment length t
        segs = trajectories[:, :t] 
        weights_t = instantaneous_weights[:, :t]
        
        pressure_approx, Z_t = compute_pressure_approx(segs, weights_t, eps)
        t_values.append(t)
        pressures.append(pressure_approx)
        Zt_values.append(Z_t)

    # Plotting
    ax[0].plot(t_values, Zt_values, marker='o', label=config['label'], color=config['color'])



    ax[1].plot(t_values, Zt_values, marker='o', label=config['label'], color=config['color'])
    
    # Plot on the right subplot (log scale)
    log_Zt = np.log(Zt_values)
    # Linear fit to estimate entropy on the right panel
    # We fit all data points for this example
    coef = np.polyfit(t_values[0:10], log_Zt[0:10], 1)
    fit_line = np.polyval(coef, t_values)
    ax[1].plot(t_values, np.exp(fit_line), '--', label=r'Topological pressure P (fit slope) $\approx$ {0:.3f}'.format(coef[0]))

    # Plot the theoretical value for P(1) = log(2)
    #ax.axhline(theoretical_pressure_P1, color='blue', linestyle='--', label=r'$P(\beta=1.0) \approx \ln(2) \approx {0:.3f}$'.format(theoretical_pressure_P1))


ax[0].legend(fontsize=16)
ax[1].legend(fontsize=16)
#fig.suptitle(r'Topological Pressure $P(\beta)$ using Lyapunov Weights $\phi_\beta(x) = \beta \log|f\'(x)|$' + '\n' + 'Estimated Pressure $P_t(\beta) \approx (1/t) \ln(\sum \exp(S_t \phi_\beta))$', fontsize=18)

plt.tight_layout()
plt.savefig('Topological_Pressure_Lyapunov_Weight.png')