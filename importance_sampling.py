# This module runs the code for assignment 1 of Stochastic Simulation

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
from tqdm import trange

# --------------
# Volume dimensions
# --------------
def sphere(x, y, z, k):
    """Checks if the point is within the sphere and passes True if so."""
    if k <= 0:
        raise ValueError(f"k needs to be > 0. You got k: {k}")

    # Sphere dimensions
    hits = x*x + y*y + z*z <= k ** 2

    return hits


def torus(x, y, z, R, r):
    """Checks if the point is within the torus and passes True if so."""
    if R <= 0 or r <= 0:
        raise ValueError(f"R and r need to be > 0." 
                         f"You got R: {R} and r: {r}")

    # Torus dimensions
    hits = (np.sqrt(x*x + y*y) - R) ** 2 + z*z <= r ** 2
    
    return hits


# --------------
# Sampling
# --------------
def uniformrandom(N, seed=None):
    np.random.seed(seed=seed)
    x = np.random.rand(N)
    y = np.random.rand(N)
    z = np.random.rand(N)

    return np.array([x, y, z])


def deterministic_XYZ(N, seed=None):
    sequence = np.empty((3, N))
    m = 3.8

    # define a self-contained region
    b = m * 0.5 * (1 - 0.5)
    a = m * b * (1 - b)
    
    # 'seed' the deterministic sequence
    np.random.seed(seed=seed)
    sequence[:, 0] = np.random.uniform(a, b, 3)
    
    for i in range(1, N):
        sequence[:, i] = m * sequence[:, i - 1] * (1 - sequence[:, i - 1])

    return sequence 


# define random number generator

# --------------
# Errors
# --------------
def standard_error(sample_std, N):
    """Calculates the standard error of the mean."""
    return sample_std / np.sqrt(N)


# --------------
# monte carlo
# --------------
def montecarlo(prng, radius, k, R, r, throws, xc=0, yc=0, zc=0, plot=False):
    rand = prng(throws)
    x, y, z = radius * (np.ones((3, throws)) - 2 * rand)

    sphereHits = sphere(x, y, z, k)
    torusHits = torus(x-xc, y-yc, z-zc, R, r)

    totalHits = np.sum(np.logical_and(sphereHits, torusHits))

    box_volume = (2 * radius) ** 3
    intersection_volume = box_volume * (totalHits / throws)

    if plot == True:
        plotintersection(x, y, z, sphereHits, torusHits, radius)

    return intersection_volume, totalHits

def montecarlo_importance(prng, b1_r, p, k, R, r, throws, xc=0, yc=0, zc=0.1, plot=False):
    """b1_r : radius of box 1
    p : sampling probability inside box 1
    w: weight for importance sampling
    """
    # box 1 (half)
    box1_x = box1_y = box1_z = b1_r
    
    # radius and height of small box (half)
    box2_x = box2_z = R+r
    box2_y = r

    if box2_x > box1_x:
        box2_x = box2_z = box1_x

    # create mask to determine where to sample (True = sample from box 1)
    rng = np.random.default_rng()
    choose1 = rng.random(throws) <= p

    # create x, y and z grid of dimension throws
    rx, ry, rz = prng(throws)

    # Define outsides of box
    xx = np.where(choose1, box1_x, box2_x)
    yy = np.where(choose1, box1_y, box2_y)
    zz = np.where(choose1, box1_z, box2_z)

    # define centers
    xc = np.where(choose1, 0, xc)
    yc = np.where(choose1, 0, yc)
    zc = np.where(choose1, 0, zc)

    x = xc + (2*xx)*(rx - 0.5)
    y = yc + (2*yy)*(ry - 0.5)
    z = zc + (2*zz)*(rz - 0.5)

    # check if points in sphere and torus
    sphereHits = sphere(x, y, z, k)
    torusHits = torus(x-xc, y-yc, z-zc, R, r)
    totalHits = np.sum(np.logical_and(sphereHits, torusHits))

    box1_volume = (2 * b1_r) ** 3
    box2_volume = (2*box2_x)*(2*box2_y)*(2*box2_z)

    # weight accordingly
    



    # if True in choose1: sample from box 1 so for every choose1 where true, 
    # the corresponding position in x, y and z is from a position in box 1

    # If false then for box 2


    # do this for all throws so that you create a list of all sampling points:
        # sample randomly from 0 to 1, if sample < p then take sample in box 1
        # otherwise take sample from box 2.create mask for each point True is value from box 1
        # False is value from box 2. 

    # test if sample points are an intersection of the sphere and torus
    # weight the sampled points accordingly to the mask that tells you 
    # where the point is sampled from and apply the correct weight.


    return 

def run_monte_carlo(
        N=100000, 
        prng=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.75, 
        r=0.4, 
        throws=100000,
        xc=0,
        yc=0,
        zc=0):
    """Runs the monte carlo simulation N times."""

    all_volumes = []
    all_hits = []

    # Time progress bar
    t0 = time.perf_counter()
    for _ in trange(N, desc="Monte Carlo runs", leave=False):
        intersection_volume, hits = montecarlo(prng, radius, k, R, r, throws, xc, yc, zc)
        all_volumes.append(intersection_volume)
        all_hits.append(hits)
    t1 = time.perf_counter()

    sample_std = np.std(all_volumes)
    average_volume = np.mean(all_volumes)

    elapsed = t1 - t0
    print(f"for radius={radius}, k={k}, R={R}, r={r}, and throws={throws}, we get:")
    print(f"average_volume: {average_volume}, sample variance: {sample_std}")
    print(f"Elapsed: {elapsed:.3f}s  ({elapsed/N:.6f}s per run)")

    return sample_std, average_volume

# --------------
# Plotting code
# --------------
def plotintersection(x, y, z, sphereHits, torusHits, radius):
    # store points for each category
    
    # Add code
    intersection = sphereHits & torusHits

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot intersection
    ax.scatter(
        x[intersection],
        y[intersection],
        z[intersection],
        s=6, alpha=0.7, label=f"Intersection ({intersection.sum()})"
    )

    # labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title("Intersection")
    ax.legend()
    
    # set limits for each axis
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([-radius, radius])

    plt.show()

# Plot error changes and estimates


# --------------
# Running code
# --------------
def main():
    # case a:
    a_sample_std, a_average_volume = run_monte_carlo(
        N=10000, 
        prng=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.75, 
        r=0.4, 
        throws=100)

    # case b:
    b_sample_std, b_average_volume = run_monte_carlo(
        N=10000, 
        prng=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.5, 
        r=0.5, 
        throws=100)
 
    # plot code
    montecarlo(prng=uniformrandom, radius=1.1, k=1, R=0.75, r=0.4, throws=10000, plot=True)
    

#main()

montecarlo_importance(
    prng=uniformrandom, 
    b1_r=1.1, 
    p=0.8,
    k=1, 
    R=0.75, 
    r=0.4, 
    throws=100, 
    xc=0, 
    yc=0, 
    zc=0.1, 
    plot=False)
