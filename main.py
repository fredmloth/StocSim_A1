# This module runs the code for assignment 1 of Stochastic Simulation

import numpy as np
import matplotlib as plt

# Define shape of box to sample from
# case a box from -1 to 1 for all dimensions)

# random nr generator (module or ourselves)

def sphere(x, y, z, k):
    #raise error for k <= 0

    # Check if the values are inside the sphere and return True
    if x*x + y*y + z*z <= k ** 2:
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
    hits = 0 # number of hits in intersection

    for _ in range(throws):
        x, y, z = uniformrandom(radius)

    if sphere(x, y, z, k) and torus(x, y, z, bigr, smallr):
        hits += 1

    box_volume = (2 * radius) ** 3
    intersection_volume = box_volume * (hits / throws)

    return intersection_volume, hits


# Uniform random sampling
def uniformrandom(radius):
    x = np.random.uniform(-radius, radius)
    y = np.random.uniform(-radius, radius)
    z = np.random.uniform(-radius, radius)
    
    return x, y, z


def plotintersection():
    return 

def deterministic_sampling():

    return


def main():
    radius = 1.1 # radius of bounding box
    k = 1
    bigr = 0.75
    smallr = 0.4

    # number of measurements
    throws = 10
    montecarlo(radius, k, bigr, smallr, throws)


# do not sample any 3 combinations multiple times

def run_monte_carlo(
        N=1000, 
        sampling=uniformrandom, 
        radius=1.1, 
        k=1, 
        bigr=0.75, 
        smallr=0.4, 
        throws=10):
    """Runs the monte carlo simulation N times."""

    all_volumes = []
    all_hits = []

    for _ in range(N):
        intersection_volume, hits = montecarlo(radius, k, bigr, smallr, throws)
        all_volumes.append(intersection_volume)
        all_hits.append(hits)

    sample_variance = np.var(all_volumes)
    average_volume = np.average(all_volumes)
    print(f"average_volume: {average_volume}, sample variance: {sample_variance}")

run_monte_carlo()
        
