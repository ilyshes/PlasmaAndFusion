import numpy as np
from itertools import product
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt

# --- Core Dynamical System Functions ---

EN = []
x_arr = []

def logistic_map(x: float, mu: float = 4.0) -> float:
    """
    The logistic map: f(x) = mu * x * (1 - x).
    Assumes x is in [0, 1] for mu=4.0.
    """
    return mu * x * (1.0 - x)

def logistic_derivative(x: float, mu: float = 4.0) -> float:
    """
    The first derivative of the logistic map: f'(x) = mu * (1 - 2x).
    """
    return mu * (1.0 - 2.0 * x)

# --- Cylinder Length Calculation based on Derivative Approximation ---

def calculate_cylinder_info(
    symbol_sequence: List[int], 
    mu: float = 4.0,
) -> Tuple[float, int]:
    """
    Calculates the approximate length of the N-cylinder defined by the 
    symbolic sequence, and identifies the final cell index.
    
    The approximation uses the derivative product (related to the expansion rate 
    E_N in Eq. 16.2.1) at a representative point of the cylinder.
    
    Args:
        symbol_sequence (List[int]): The sequence i_0, i_1, ..., i_N. The length is N+1.
        mu (float): The logistic map parameter.
        
    Returns:
        Tuple[float, int]: (Approximate cylinder length l_j^(N+1), Final cell index i_{N+1})
    """
    N_plus_1 = len(symbol_sequence) # This is the length of the cylinder sequence
    
    # Representative Point Approximation: 
    # For a sequence, the center of the cylinder is approximated by its
    # binary representation. This is more accurate for maps like the Tent map,
    # but provides a deterministic starting point for full enumeration here.
    x = 0.0
    for k, symbol in enumerate(symbol_sequence):
        # We use a slight modification of the binary fraction for the starting point
        # to ensure it is deep within the symbolic sequence structure.
        x += symbol / (2**(k + 1))
        
    # The representative point x0 for the full (N+1)-cylinder is taken as x
    x0 = x 
    
    # Calculate the total derivative product for (f^{N+1})'
   
    x_i = x0
    x_arr.append(x_i)
    derivative_product = 1.0
    x_loc_arr = []
    fprime_loc_arr = []
    for _ in range(N_plus_1):
        f_prime = logistic_derivative(x_i, mu)
        
        # Guard against the critical point f'(0.5) = 0
        if abs(f_prime) < 1e-15:
             # Cylinder containing the critical pre-image has zero derivative, 
             # making its contribution to the sum negligible (or 0)
             return 0.0, 0 
        x_loc_arr.append(x_i)    
        fprime_loc_arr.append(f_prime)
        
        derivative_product *= abs(f_prime)
        x_i = logistic_map(x_i, mu)
        
    # The cylinder length is inversely proportional to the modulus of the product
    # l_j^(N+1) ~ 1 / |(f^{N+1})'(x0)|
    EN.append( derivative_product )
    cylinder_length = 1.0 / abs(derivative_product)
    
    # The final cell index i_{N+1} is defined by the cell where the last iterate lands.
    # f(x) lands in I_0=[0, 0.5) if x < 0.5, I_1=[0.5, 1] if x >= 0.5.
    # The symbolic sequence used here is already N+1 length, so we use the last symbol
    # to denote the image cell.
    final_cell_index = symbol_sequence[-1]
    
    return cylinder_length, final_cell_index

def evaluate_topological_pressure_enumeration(
    beta: float, 
    N: int, 
    mu: float = 4.0
) -> float:
    """
    Evaluates the Topological Pressure Phi(beta) using the length scale partition 
    function and full enumeration over all N-cylinder sequences (i_0, ..., i_{N-1}),
    as required by the generating partition definition (Eq. 16.2.2).
    
    Z_N^top(beta) = sum_j ( l_j^(N+1) / L_i(j) )^beta
    Phi(beta) = (1/N) * ln( Z_N^top(beta) )
    
    Args:
        beta (float): The inverse temperature (exponent).
        N (int): The symbolic sequence length (N-cylinder order).
        mu (float): The logistic map parameter (default 4.0 for full chaos).
        
    Returns:
        float: The estimated topological pressure Phi(beta).
    """
    if N <= 0:
        return 0.0

    # 1. Define the generating partition cell length L_i(j)
    # For the mu=4 logistic map on [0, 1], the partition cells I_0=[0, 0.5] 
    # and I_1=[0.5, 1] both have length 0.5.
    L_i_j = 0.5
    
    # 2. Iterate over all 2^{N+1} symbolic sequences (i_0, i_1, ..., i_N)
    # The partition function Z_N is calculated using the (N+1)-cylinder length
    num_sequences = 2**(N + 1)
    sequence_length = N + 1 
    
    # The itertools.product creates all combinations (0, 0, 0...), (0, 0, 1...), etc.
    symbolic_sequences = product([0, 1], repeat=sequence_length)
    
    sum_Z_N_top = 0.0
    
    for j, seq in enumerate(symbolic_sequences):
        symbol_sequence = list(seq)
        
        # Calculate l_j^(N+1) and the final cell index
        l_N_plus_1, final_cell_index = calculate_cylinder_info(symbol_sequence, mu)
        
        # Calculate the term for the partition function (Eq. 16.2.2)
        # Note: L_i(j) is the length of the final cell, L_{i_N}. Here L_{i_N}=0.5 always.
        
        # Term = (l_j^(N+1) / L_i(j))^beta
        if l_N_plus_1 > 0:
            term = (l_N_plus_1 / L_i_j)**beta
            sum_Z_N_top += term
            
    # 3. Calculate Phi(beta) = (1/N) * ln( Z_N^top(beta) )
    
    if sum_Z_N_top <= 0:
        print("Error: Partition function sum is non-positive.")
        return 0.0
        
    # As the limit is N->inf, the denominator in the final formula is N
    topological_pressure = (1.0 / N) * np.log(sum_Z_N_top)
    
    print(f"--- Calculation Parameters ---")
    print(f"Map Parameter (mu): {mu}")
    print(f"Inverse Temperature (beta): {beta}")
    print(f"Sequence Length (N): {N} (Summing over {num_sequences} sequences of length {sequence_length})")
    print(f"Generating Partition Cell Length L_i(j): {L_i_j}")
    print(f"------------------------------")
    
    return topological_pressure

# --- Example Usage ---
if __name__ == '__main__':
    # Full chaos logistic map (mu=4.0)
    MU = 4.0
    # N must be kept small (e.g., N <= 18) for full enumeration
    N_MAX = 12 
    
    # Test 1: Topological Entropy (beta=1). Should approach ln(2) ~ 0.693
    BETA_1 = 1.0
    print("--- Test 1: Topological Entropy (beta=1) ---")
    phi_1 = evaluate_topological_pressure_enumeration(BETA_1, N_MAX, MU)
    print(f"Estimated Phi({BETA_1}): {phi_1:.6f}")
    print(f"Theoretical ln(2): {np.log(2):.6f}\n")
    
    plt.hist(EN)
    
