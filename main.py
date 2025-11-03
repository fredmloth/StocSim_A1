# This module runs the code for assignment 1 of Stochastic Simulation

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    print(f"x")
    y = np.random.uniform(-radius, radius)
    z = np.random.uniform(-radius, radius)
    
    return x, y, z


def deterministic_sampling():
    return


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
    radius = 1.1 # radius of bounding box
    k = 1
    bigr = 0.75
    smallr = 0.4

    # number of measurements
    throws = 10
    montecarlo(radius, k, bigr, smallr, throws)

main()

# do not sample any 3 combinations multiple times
# error counting
# plotting intersection

