import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# --- Configuration Parameters ---
R = 4.0          # The growth rate parameter (0 < r <= 4).
                 # Try 3.2 (stable cycle), 3.5 (4-cycle), or 3.8 (chaos).
X0 = 0.0001         # Initial condition (must be between 0 and 1)
MAX_ITERATIONS = 1000 # Total number of points (x0 to x99) to calculate for the trajectory

def logistic_map(r, x):
    """
    Calculates the next iteration of the logistic map: x_{n+1} = r * x_n * (1 - x_n).
    """
    return r * x * (1 - x)

# 1. Pre-calculate the entire trajectory (x_0, x_1, x_2, ...)
trajectory = [X0]
for _ in range(MAX_ITERATIONS - 1):
    next_x = logistic_map(R, trajectory[-1])
    trajectory.append(next_x)

# 2. Pre-calculate the cobweb path vertices
cobweb_x = []
cobweb_y = []

# The cobweb path consists of segments:
# 1. Vertical: (x_n, x_n) -> (x_n, x_{n+1}) (Hits the parabola)
# 2. Horizontal: (x_n, x_{n+1}) -> (x_{n+1}, x_{n+1}) (Hits the identity line)

for i in range(len(trajectory) - 1):
    x_n = trajectory[i]
    x_n_plus_1 = trajectory[i+1]

    # Start point of the iteration: (x_n, x_n) (Only needed for the very first step)
    if i == 0:
        cobweb_x.append(x_n)
        cobweb_y.append(x_n)

    # Vertical step: (x_n, x_n) to (x_n, x_{n+1})
    cobweb_x.append(x_n)
    cobweb_y.append(x_n_plus_1)

    # Horizontal step: (x_n, x_{n+1}) to (x_{n+1}, x_{n+1})
    cobweb_x.append(x_n_plus_1)
    cobweb_y.append(x_n_plus_1)

TOTAL_STEPS = len(cobweb_x)

# 3. Set up the Matplotlib figure and axes
fig, ax = plt.subplots(figsize=(8, 8))

# Set plot limits and labels
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title(f'Logistic Map Cobweb Diagram: r={R}, x₀={X0}', fontsize=16)
ax.set_xlabel('$x_n$', fontsize=12)
ax.set_ylabel('$x_{n+1}$', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_aspect('equal', adjustable='box') # Ensure x and y axes have the same scale

# Plot the identity line (y=x)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='y = x')

# Plot the map function (f(x) = R * x * (1-x))
x_func = np.linspace(0, 1, 500)
y_func = logistic_map(R, x_func)
ax.plot(x_func, y_func, 'b-', linewidth=2, label=f'f(x) = {R}x(1-x)')

# Initialize the line object for the animated cobweb path
# '-' creates a continuous line
line, = ax.plot([], [], '-', color='#e91e63', linewidth=1.5, alpha=0.9)

# 4. Initialization function
def init():
    """Initializes the animation by setting empty data."""
    line.set_data([], [])
    return line,

# 5. Animation function (called for each frame)
def animate(i):
    """
    Updates the plot data for the i-th frame, adding one cobweb segment vertex at a time.
    i is the current step index.
    """
    # Plot up to the i-th vertex
    x_data = cobweb_x[:i+1]
    y_data = cobweb_y[:i+1]

    # Update the line object's data
    line.set_data(x_data, y_data)

    # Calculate the current iteration number for the title
    # Iteration 0: start point. Iteration 1: first loop, etc.
    iteration = int(i / 2)
    ax.set_title(f'Logistic Map Cobweb Diagram: r={R}, x₀={X0} (Iteration: {iteration})', fontsize=16)

    return line,

# 6. Create the animation object
# frames: The total number of steps in the cobweb path
# interval: Delay between frames in milliseconds
ani = animation.FuncAnimation(
    fig,
    animate,
    frames=TOTAL_STEPS,
    init_func=init,
    interval=50,
    blit=True,
    repeat=False
)

# Show the plot with the animation
plt.legend()
plt.show()
