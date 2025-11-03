# This module runs the code for assignment 1 of Stochastic Simulation

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
from tqdm import trange

# random nr generator (module or ourselves)


# --------------
# Volume dimensions
# --------------
def sphere(x, y, z, k):
    """Checks if the point is within the sphere and passes True if so."""
    if k <= 0:
        raise ValueError(f"k needs to be > 0. You got k: {k}")

    # Sphere dimensions
    if x*x + y*y + z*z <= k ** 2:
        return True
    
    return False


def torus(x, y, z, R, r):
    """Checks if the point is within the torus and passes True if so."""
    if R <= 0 or r <= 0:
        raise ValueError(f"R and r need to be > 0." 
                         f"You got R: {R} and r: {r}")

    # Torus dimensions
    if (np.sqrt(x*x + y*y) - R) ** 2 + z*z <= r ** 2:
        return True
    
    return False


# --------------
# Sampling
# --------------
def uniformrandom(radius):
    x, y, z = np.random.uniform(-radius, radius, size=3)
    
    return x, y, z


def deterministic_sampling():
    return


# --------------
# monte carlo
# --------------
def montecarlo(radius, k, R, r, throws):
    hits = 0 # number of hits in intersection

    for _ in range(throws):
        x, y, z = uniformrandom(radius)

        if sphere(x, y, z, k) and torus(x, y, z, R, r):
            hits += 1

    box_volume = (2 * radius) ** 3
    intersection_volume = box_volume * (hits / throws)

    return intersection_volume, hits


def run_monte_carlo(
        N=100000, 
        sampling=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.75, 
        r=0.4, 
        throws=100000):
    """Runs the monte carlo simulation N times."""

    all_volumes = []
    all_hits = []

    t0 = time.perf_counter()
    for _ in trange(N, desc="Monte Carlo runs", leave=False):
        intersection_volume, hits = montecarlo(radius, k, R, r, throws)
        all_volumes.append(intersection_volume)
        all_hits.append(hits)
    t1 = time.perf_counter()

    sample_variance = np.var(all_volumes)
    average_volume = np.mean(all_volumes)

    elapsed = t1 - t0
    print(f"average_volume: {average_volume}, sample variance: {sample_variance}")
    print(f"Elapsed: {elapsed:.3f}s  ({elapsed/N:.6f}s per run)")

    return sample_variance, average_volume

# --------------
# Plotting code
# --------------
def get_coords(points_list):
    if not points_list: # check for empty list
        return np.array([]), np.array([]), np.array([])
    
    arr = np.array(points_list)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def plotintersection(N, radius, k, bigr, smallr, xc=0, yc=0, zc=0, title=""):
    # store points for each category
    points_sphere_only = []
    points_torus_only = []
    points_intersection = []

    for _ in range(N):
        x, y, z = uniformrandom(radius)

        in_sphere = sphere(x, y, z, k)
        in_torus = torus(x, y, z, bigr, smallr)

        if in_sphere and in_torus:
            points_intersection.append((x, y, z))
        elif in_sphere:
            points_sphere_only.append((x, y, z))
        elif in_torus:
            points_torus_only.append((x, y, z))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plotting by category
    xs, ys, zs = get_coords(points_sphere_only)
    ax.scatter(xs, ys, zs, color='blue', alpha=0.1, s=2, label='Sphere Only')
    
    xt, yt, zt = get_coords(points_torus_only)
    ax.scatter(xt, yt, zt, color='green', alpha=0.1, s=2, label='Torus Only')
    
    xi, yi, zi = get_coords(points_intersection)
    ax.scatter(xi, yi, zi, color='red', alpha=0.5, s=5, label='Intersection')

    return


def main():
    # case a:
    run_monte_carlo(
        N=100000, 
        sampling=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.75, 
        r=0.4, 
        throws=100)

    # case b:
    run_monte_carlo(
        N=10000, 
        sampling=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.5, 
        r=0.5, 
        throws=100)


main()
        
