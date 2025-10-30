# This module runs the code for assignment 1 of Stochastic Simulation

import numpy as np
import matplotlib.pyplot as plt

# Define shape of box to sample from
# case a box from -1 to 1 for all dimensions)

# random nr generator (module or ourselves)

def sphere(x, y, z, k):
    #raise error for k <= 0

    # Check if the values are inside the sphere and return True
    if x*x + y*y + z*z <= k:
        return True
    
    return False


def torus(x, y, z, bigR, smallr):
    #raise error for bigR, smallr <= 0

    # Check if values are inside Torus and return True
    if (np.sqrt(x*x + y*y) - bigR) ** 2 + z*z <= smallr ** 2:
        return True
    
    return False


# Monte carlo integration
def montecarlo(radius, k, bigr, smallr, throws):
    
    for _ in range(throws):
        x, y, z = uniformrandom(radius)
        if sphere(x, y, z, k) and torus(x, y, z, bigr, smallr):
            print(f"intersection at point x:{x}, y:{y}, z:{z}")

    return #volume


# Uniform random sampling
def uniformrandom(radius):
    x = np.random.uniform(-radius, radius)
    print(f"x:{x}")
    y = np.random.uniform(-radius, radius)
    z = np.random.uniform(-radius, radius)
    
    return x, y, z


def plotintersection():
    return 

def main():
    radius = 1.1
    k = 1
    bigr = 0.75
    smallr = 0.4

    # number of measurements
    throws = 10
    montecarlo(radius, k, bigr, smallr, throws)

main()
# Sampling function for volume intersection with bounding box? -> check what is bounding box
# Bounding box: space that encompases both together fully from which we sample 
# do not sample any 3 combinations multiple times

