#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 15:02:54 2025

@author: ilya
"""
import numpy as np

# Define the inverse functions for the logistic map f(x) = 4x(1-x)
def inverse_left(x_next):
    """Inverse function for symbol 0 (left branch)"""
    return (1 - np.sqrt(1 - x_next)) / 2

def inverse_right(x_next):
    """Inverse function for symbol 1 (right branch)"""
    return (1 + np.sqrt(1 - x_next)) / 2

# Given symbol sequence (s0, s1, ..., s11)
symbol_sequence = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]

# Reverse the sequence for inverse iteration: (s11, s10, ..., s0)
reversed_sequence = symbol_sequence[::-1]

# Start the iteration from the terminal point x_N.
# A common choice is the critical point xc = 0.5.
x_n = 0.5 

# Iterate backwards
for i, symbol in enumerate(reversed_sequence):
    # n is the time index we are calculating (starts at 11, ends at 0)
    # The symbol determines which inverse branch to use
    if symbol == 0:
        x_n = inverse_left(x_n)
    elif symbol == 1:
        x_n = inverse_right(x_n)
        
# The final value is the initial condition x0
x_0 = x_n

print(f"The representative point x_0 is approximately: {x_0}")
