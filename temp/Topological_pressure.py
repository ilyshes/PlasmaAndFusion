#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 12:25:08 2025

@author: ilya
"""

import numpy as np
import math
from itertools import product
from scipy.optimize import brentq

import matplotlib.pyplot as plt

# --- Constants and Definitions ---
R = 3.9  # Parameter for the fully chaotic Logistic Map: f(x) = 4x(1-x)
CRITICAL_POINT = 0.5
MAX_N = 14 # Maximum N to keep computation time reasonable (2^N cylinders)

EN = []
x_arr = []

def logistic_map(x):
    """The Logistic Map function f(x) = R * x * (1 - x)."""
    return R * x * (1.0 - x)

def logistic_prime(x):
    """The derivative of the Logistic Map: f'(x) = R * (1 - 2x)."""
    return R * (1.0 - 2.0 * x)

def accumulated_derivative(x0, N):
    """
    Calculates the magnitude of the N-th iteration derivative, |(f^N)'(x0)|.
    
    |(f^N)'(x0)| = Product_{n=0}^{N-1} |f'(x_n)|, where x_{n+1} = f(x_n).
    """
    x_loc_arr = []
    fprime_loc_arr = []
    
    x = x0
    derivative_abs = 1.0
    for _ in range(N):
        derivative = logistic_prime(x)
        derivative_abs *= abs(derivative)
        x_loc_arr.append(x)  
        fprime_loc_arr.append(derivative)
        x = logistic_map(x)
        
        
        
    return derivative_abs

def get_symbol(x):
    """Assigns the symbolic dynamics (0 or 1) based on the critical point."""
    if x < CRITICAL_POINT:
        return 0
    else:
        return 1

def f_N_minus_yc(x0, N, target_symbol_sequence):
    """
    Calculates f^N(x0) - 0.5, and checks if the trajectory follows the
    given symbolic sequence. This is the function whose root we seek.

    If the symbolic sequence is violated at any point, we return a large
    value with the appropriate sign to guide the root-finding algorithm
    out of the wrong cylinder.
    """
    x = x0
    for n in range(N):
        current_symbol = get_symbol(x)
        if current_symbol != target_symbol_sequence[n]:
            # Trajectory leaves the cylinder. Return a value that indicates the
            # search needs to move left or right to re-enter the cylinder.
            # If the current x is too far right (symbol 1 expected, got 0),
            # f^N(x) will be too low, so we return a large negative number.
            if current_symbol == 0 and target_symbol_sequence[n] == 1:
                return -1e10
            # If the current x is too far left (symbol 0 expected, got 1),
            # f^N(x) will be too high, so we return a large positive number.
            elif current_symbol == 1 and target_symbol_sequence[n] == 0:
                return 1e10
            else: # Should not happen if symbols are only 0 or 1
                return 0

        x = logistic_map(x)
    
    # After N iterations, the root is found when f^N(x0) = 0.5
    return x - CRITICAL_POINT

def find_cylinder_point(symbol_sequence):
    """
    Numerically finds the representative point x0 in the cylinder C_sigma
    such that f^N(x0) = 0.5, using Brent's method (brentq for root-finding).
    
    This point is often used as the canonical representative for the cylinder.
    We search in the domain [0, 1].
    """
    N = len(symbol_sequence)
    
    # We must ensure that the endpoints of the search interval [a, b]
    # have g(a) and g(b) with opposite signs, or else brentq will fail.
    # The simplest brute-force way is to search over a very fine grid
    # to find the two points that bound the root for the given symbol sequence.
    
    try:
        # brentq is robust, but requires a bracket [a, b] where f(a)f(b) < 0.
        # Since the function f_N_minus_yc is well-behaved and monotonic inside
        # the cylinder, we can search for the boundaries.
        
        # Search strategy: Find two adjacent points a and b on a fine grid
        # where the symbolic trajectory is correct and f^N(x)-0.5 changes sign.
        
        num_points = 2**(N + 2) # Heuristically chosen grid size
        X = np.linspace(0.0001, 0.9999, num_points) # Avoid endpoints 0 and 1
        
        # Filter for points that respect the symbolic sequence
        points_in_cylinder = []
        for x_test in X:
            if abs(f_N_minus_yc(x_test, N, symbol_sequence)) < 1.0:
                 points_in_cylinder.append(x_test)

        if not points_in_cylinder:
            # Fallback for very small cylinders where grid misses the points
            # In a real application, one would use the inverse map method
            # to precisely determine the cylinder's [a, b] endpoints.
            # For simplicity, we just return NaN and skip this sequence.
            return np.nan
        
        # Use the first and last valid point as the search bracket
        a = points_in_cylinder[0]
        b = points_in_cylinder[-1]
        
        # Check signs at the bracket edges
        if f_N_minus_yc(a, N, symbol_sequence) * f_N_minus_yc(b, N, symbol_sequence) > 0:
            # If signs are the same, try a tighter bracket near 0.5, as the
            # root must exist within the region where the symbolic sequence holds.
            # We rely on the initial search (f_N_minus_yc < 1.0) to filter correctly.
            a = points_in_cylinder[len(points_in_cylinder)//4]
            b = points_in_cylinder[len(points_in_cylinder)*3//4]
            if f_N_minus_yc(a, N, symbol_sequence) * f_N_minus_yc(b, N, symbol_sequence) > 0:
                 # If still same sign, something is wrong with the sequence/N.
                 return np.nan # Skip this cylinder
        
        
        # Find the root of f^N(x) - 0.5 = 0 within the bracket [a, b]
        x_root = brentq(f_N_minus_yc, a, b, args=(N, symbol_sequence))
        return x_root

    except Exception as e:
        # print(f"Error finding root for sequence {symbol_sequence}: {e}")
        return np.nan # Return NaN if root-finding fails

def calculate_Z_N_top(N, beta):
    """
    Calculates the topological partition function Z_N^top(beta)
    for the N-cylinder ensemble.
    """
    if N > MAX_N:
        print(f"N is too large. Setting N = {MAX_N}")
        N = MAX_N

    # Generate all 2^N possible symbolic sequences (N-cylinders)
    symbol_sequences = list(product([0, 1], repeat=N))
    
    Z_N_top = 0.0
    valid_cylinders = 0

    print(f"\n--- Calculating Z_N^top for N={N}, beta={beta} ---")
    
    for k, sequence in enumerate(symbol_sequences):
        # 1. Find the representative point x0 for the cylinder
        x0_k = find_cylinder_point(sequence)
        
        
        #print(sequence)
        print(x0_k)
        if np.isnan(x0_k):
            
            # This cylinder is numerically inaccessible or invalid. Skip.
            continue
        x_arr.append(x0_k)
           
        # 2. Calculate the accumulated derivative |(f^N)'(x0)|
        deriv_abs = accumulated_derivative(x0_k, N)
        EN.append( deriv_abs )
        
        # 3. Add the term to the partition function
        # Z_N^top(beta) = Sum |(f^N)'(x0_k)|^(-beta)
        if deriv_abs > 0:
            term = deriv_abs**(-beta)
            Z_N_top += term
            valid_cylinders += 1

    print(f"Processed {len(symbol_sequences)} cylinders, found {valid_cylinders} valid points.")
    return Z_N_top

def calculate_P_top(N, beta):
    """
    Calculates the N-th order approximation of the topological pressure:
    P_N(beta) = (1/N) * ln(Z_N^top(beta))
    """
    Z_N_top = calculate_Z_N_top(N, beta)
    
    if Z_N_top <= 0:
        print("Error: Z_N_top is non-positive.")
        return np.nan

    # The formula provided in the text defines P(beta) with a negative sign
    # in the limit: P(beta) = -lim (1/N) ln(Z_N^top).
    # However, in many contexts, including the one for the logistic map (R=4),
    # the topological pressure is defined positively as P(beta) = lim (1/N) ln(Z_N^top).
    # Since the derivative terms are < 1, Z_N^top < 1, and ln(Z_N^top) is negative.
    # To follow the convention for topological pressure of expanding maps:
    # We will use the common definition P_N(beta) = (1/N) * ln(Z_N^top(beta))
    # which will yield a negative result due to the exponent being negative.
    # To match standard literature results where P(0) = log(2) > 0, the text's
    # formula seems to have a typo or uses a non-standard convention for the sign
    # of the pressure when defining the local expansion rate *E_N* as energy.
    
    # We will use the convention: P_N(beta) = (1/N) * ln(Z_N^top(beta))
    # Note: If P(beta) = - lim (1/N) ln(Z_N^top), use the commented line below.
    # P_N = -(1.0 / N) * math.log(Z_N_top)
    
    P_N = (1.0 / N) * math.log(Z_N_top)
    
    return P_N

# --- Execution ---
if __name__ == '__main__':
    # Test cases:
    # 1. beta = 0: Topological Entropy (Should approach log(2) approx 0.693 for R=4)
    # 2. beta = 1: Topological Pressure P(1) (Should approach 0 for R=4)
    # 3. beta = 2: Example case
    
    N_values = [13] # Test different cylinder lengths
    beta_values = [2.0]

    
    
    for beta in beta_values:
        print(f"\n==============================================")
        print(f"Evaluating Topological Pressure for beta = {beta}")
        print(f"==============================================")
        
        for N in N_values:
            P_N = calculate_P_top(N, beta)
            
            # Note on sign convention: The formula Z_N^top = Sum |(f^N)'|^-beta
            # leads to P_N = (1/N) ln(Z_N^top) < 0 since |(f^N)'| > 1 for R=4
            # (except for the critical point).
            # If we use the sign convention P(beta) = lim (1/N) ln(Sum |(f^N)'|^beta)
            # then P(0) = log(2) > 0.
            
            # The structure used here is: Z_N^top = Sum exp(-beta * N * E_N)
            # where E_N = (1/N) ln |(f^N)'|. Since E_N > 0, Z_N^top < 1.
            # ln(Z_N^top) < 0. P_N < 0.
            
            # To get the conventional positive topological pressure, we use the
            # negative of the calculated value, which aligns with the text's
            # formula P(beta) = -lim (1/N) ln(Z_N^top) when |(f^N)'| > 1.
            
            P_N_conventional = -P_N
            
            print(f"N={N} (2^{N}={2**N} cylinders):")
            print(f"  P_N(beta) approx (using text's sign): {P_N_conventional:.6f}")
            
        if beta == 0.0:
            print("\n*** For beta=0, the result should converge to the Topological Entropy, ln(2) ≈ 0.693 ***")
        if beta == 1.0:
            print("\n*** For the R=4 map, the result P(1) should converge to 0 ***")

        plt.hist(x_arr, bins = 100)