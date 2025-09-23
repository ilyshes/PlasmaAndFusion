import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

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

# Count distinguishable orbit segments at resolution eps
def count_distinguishable(segments, eps):
    n = segments.shape[0]
    used = np.zeros(n, dtype=bool)
    count = 0
    for i in range(n):
        if used[i]:
            continue
        count += 1
        # sup norm distance between segment i and all others
        dists = np.max(np.abs(segments - segments[i]), axis=1)
        within = dists <= eps
        used = used | within
    return count

# Parameters
r = 4.0
t_max = 6
np.random.seed(0)

# Define different combinations of epsilon and num_initial to plot
Number_of_points = 50000
configs = [
    {'eps': 0.1, 'num_initial': Number_of_points, 'label': r'$\epsilon$=0.1', 'color': 'orange'},
    {'eps': 0.05, 'num_initial': Number_of_points, 'label': r'$\epsilon$=0.05', 'color': 'c'},
    {'eps': 0.02, 'num_initial': Number_of_points, 'label': r'$\epsilon$=0.02', 'color': 'g'},
    {'eps': 0.01, 'num_initial': Number_of_points, 'label': r'$\epsilon$=0.01', 'color': 'y'},
    {'eps': 0.005, 'num_initial': Number_of_points, 'label': r'$\epsilon$=0.005', 'color': 'b'},
    {'eps': 0.002, 'num_initial': Number_of_points, 'label': r'$\epsilon$=0.002', 'color': 'r'},
]

# Create a figure with two subplots side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Customize the left subplot
ax[0].set_xlabel('Segment length t', fontsize=16)
ax[0].set_ylabel('Number of distinguishable segments' , fontsize=16)
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
ax[0].set_title('Linear Scale',  fontsize=15)
ax[0].grid(True)


# Customize the right subplot
ax[1].set_xlabel('Segment length t' , fontsize=16)
#ax[1].set_ylabel('log(number of distinguishable segments)', fontsize=16)
ax[1].set_title('Logarithmic Scale',  fontsize=15)
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
ax[1].set_yscale('log')
ax[1].grid(True, which='both')



for config in configs:
    eps = config['eps']
    num_initial = config['num_initial']
    
    # Sample random initial conditions
    initials = np.random.rand(num_initial) * 0.999 + 0.0005

    # Compute distinguishable counts
    t_values = []
    counts = []
    
    for t in range(1, t_max + 1):
        segs = np.array([orbit_segment(x0, t, r) for x0 in initials])
        distinguishable = count_distinguishable(segs, eps)
        t_values.append(t)
        counts.append(distinguishable)

    # Plot on the left subplot (non-log scale)
    ax[0].plot(t_values, counts, marker='o', label=config['label'], color=config['color'])

    # Plot on the right subplot (log scale)
    log_counts = np.log(counts)
    ax[1].plot(t_values, counts, marker='o', color=config['color'])

    # Linear fit to estimate entropy on the right panel
    # We fit all data points for this example
    coef = np.polyfit(t_values, log_counts, 1)
    fit_line = np.polyval(coef, t_values)
    ax[1].plot(t_values, np.exp(fit_line), '--', label=r'h (fit slope) $\approx$ {0:.3f}'.format(coef[0]))




ax[0].legend(fontsize=16)
ax[1].legend(fontsize=16)
# Add a main title for the entire figure
fig.suptitle('Distinguishable Segments of Logistic Map (r=4.0)' + '\n' + 'h - Entropy is the rate of information loss ' + r'$e^{ht}$', fontsize=18)

plt.tight_layout()
plt.show()
plt.savefig('Distinguishable_Segments.png')
print("Plot saved as Distinguishable_Segments.png")