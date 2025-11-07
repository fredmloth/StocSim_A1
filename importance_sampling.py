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
    """Checks if passed coordinates are within the sphere. 
    Returns a boolean array with True for points within the sphere."""
    if k <= 0:
        raise ValueError(f"k needs to be > 0. You got k: {k}")

    # Sphere dimensions
    hits = x*x + y*y + z*z <= k ** 2

    return hits


def torus(x, y, z, R, r):
    """Checks if passed coordinates are within the torus. 
    Returns a boolean array with True for points within the torus."""
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
    """Uniform random sampler for N number of x, y z, coordinates as 
    an array."""
    np.random.seed(seed=seed)
    x = np.random.rand(N)
    y = np.random.rand(N)
    z = np.random.rand(N)

    return np.array([x, y, z])


def deterministic_XYZ(N, seed=None):
    """Deterministic sampler for a sequence."""
    sequence = np.empty((3, N))
    m = 3.8

    # Definea a self-contained region
    b = m * 0.5 * (1 - 0.5)
    a = m * b * (1 - b)
    
    # 'seed' the deterministic sequence
    np.random.seed(seed=seed)
    sequence[:, 0] = np.random.uniform(a, b, 3)
    
    for i in range(1, N):
        sequence[:, i] = m * sequence[:, i - 1] * (1 - sequence[:, i - 1])

    return sequence 


# --------------
# Errors
# --------------
def standard_error(sample_std, N):
    """Calculates the standard error of the mean."""
    return sample_std / np.sqrt(N)


# --------------
# monte carlo
# --------------
def montecarlo(
        prng, 
        radius, 
        k, 
        R, 
        r, 
        throws, 
        xc=0, 
        yc=0, 
        zc=0, 
        p=None, 
        plot=False):
    """Runs a single monte carlo method to solve the intersection volume 
    of two shapes: sphere and torus. Returns the volume and the sum of 
    the total amount of points calculated to be in the intersection."""
    # Assign random sampler and get x, y and z arrays
    rand = prng(throws)
    x, y, z = radius * (np.ones((3, throws)) - 2 * rand)

    # Finds which points hit the sphere and torus
    sphereHits = sphere(x, y, z, k)
    torusHits = torus(x-xc, y-yc, z-zc, R, r)

    totalHits = np.sum(np.logical_and(sphereHits, torusHits))

    # Determine volume of intersection
    box_volume = (2 * radius) ** 3
    intersection_volume = box_volume * (totalHits / throws)

    if plot == True:
        plotintersection(x, y, z, sphereHits, torusHits, radius)

    return intersection_volume, totalHits


def mc_importance(prng, b1_r, k, R, r, throws, xc=0, yc=0, zc=0.1, p=0.5, plot=False):
    """Runs asingle monte carlo method with importance sampling 
    (for p and 1-p) to solve the intersection volume of two shapes: 
    sphere and torus. Returns the volume and the sum of the total amount 
    of points calculated to be in the intersection. """
    # box 1 (half)
    box1_x = box1_y = box1_z = b1_r
    
    # radius and height of small box (half)
    box2_x = box2_z = min(R+r, b1_r)
    box2_y = r

    # box volumes
    box1_volume = (2 * b1_r) ** 3
    box2_volume = (2*box2_x)*(2*box2_y)*(2*box2_z)

    # create mask to determine where to sample (True = sample from box 1)
    rng = np.random.default_rng()
    choose1 = rng.random(throws) < p

    # create x, y and z grid of dimension throws
    rx, ry, rz = prng(throws)

    # Define outsides of box
    xx = np.where(choose1, box1_x, box2_x)
    yy = np.where(choose1, box1_y, box2_y)
    zz = np.where(choose1, box1_z, box2_z)

    # define centers
    cx = np.where(choose1, 0, xc)
    cy = np.where(choose1, 0, yc)
    cz = np.where(choose1, 0, zc)

    x = cx + (2*xx)*(rx - 0.5)
    y = cy + (2*yy)*(ry - 0.5)
    z = cz + (2*zz)*(rz - 0.5)

    # check if points in sphere and torus
    sphereHits = sphere(x, y, z, k)
    torusHits = torus(x-xc, y-yc, z-zc, R, r)
    hits = np.logical_and(sphereHits, torusHits).astype(float)
    totalHits = int(hits.sum())

    # weight accordingly
    in_box2 = (
        (np.abs(x-xc) <= box2_x) &
        (np.abs(y-yc) <= box2_y) &
        (np.abs(z-zc) <= box2_z)
    )

    q = np.where(
        in_box2, p/box1_volume + (1-p)/ box2_volume, p/box1_volume)
    w = 1.0 / q

    # determine weighted intersection volume
    intersection = np.sum(w*hits)/throws

    if plot:
        plotintersection(x, y, z, sphereHits, torusHits, b1_r)

    return intersection, totalHits


def run_monte_carlo(
        mc=montecarlo,
        N=100000, 
        prng=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.75, 
        r=0.4, 
        throws=100000,
        xc=0,
        yc=0,
        zc=0,
        p=None):
    """Runs a monte carlo simulation (regular or importance sampling) N 
    times and prints the average volume and variance of the intersection 
    volume."""
    all_volumes = []
    all_hits = []

    # Time progress bar
    t0 = time.perf_counter()
    for _ in trange(N, desc="Monte Carlo runs", leave=False):
        intersection_volume, hits = mc(prng, radius, k, R, r, throws, xc, yc, zc, p)
        all_volumes.append(intersection_volume)
        all_hits.append(hits)
    t1 = time.perf_counter()

    # Determines standard deviation and average volume
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
    """Plots the intersected points."""
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


# plot histogram for deterministic sequence -> show that it is not uniform
def plotDeterministicHistogram(N):
    xDet, _, _ = deterministic_XYZ(N)
    xRand, _, _ = uniformrandom(N)

    plt.hist(xDet, density=True, label="deterministic")
    plt.hist(xRand, density=True, histtype="step", label="numpy prng")
    plt.xlabel("Sample value")
    plt.ylabel("Density of occurence")
    plt.legend()
    plt.show()
    
    return



# Plot error changes and estimates
def convergencePlot(N, radius, k, R, r, maxThrows, throwsSamples):
    """Runs the Monte Carlo simulation N times and plots convergence for 
    uniform vs deterministic samplers."""
    # log-spaced list of throw counts between 10 and maxThrows
    throwsList = np.logspace(1, np.log(maxThrows) / np.log(10), 
                             throwsSamples, base=10, dtype=np.int32)

    means = np.empty((2, throwsSamples))
    stds = np.empty((2, throwsSamples))
    for i in trange(throwsSamples, desc="Convergence plot: ", leave=False):
        throws = throwsList[i]
        vols = np.empty((2, N))
        
        # run N repetitions for both samplers
        for n in range(N):
            vols[0, n], _ = montecarlo(uniformrandom, radius, k, R, r, throws)
            vols[1, n], _ = montecarlo(deterministic_XYZ, radius, k, R, r, throws)

        # aggregate: mean and std across repetitions
        means[:, i] = np.mean(vols, axis=1)
        stds[:,  i] = np.std(vols, axis=1)

    # plot mean with std as error bars
    fig, ax = plt.subplots()
    ax.errorbar(throwsList, means[0, :], yerr=stds[0, :], fmt='o', label="uniform")
    ax.errorbar(throwsList, means[1, :], yerr=stds[1, :], fmt='o', label="deterministic")
    ax.set_xscale("log")
    ax.set_xlabel("Amount of throws")
    ax.set_ylabel("Volume estimate (a.u.)")
    ax.legend()

    plt.show()


# --------------
# Running code
# --------------
def main():
    """
    TO DO: standard error calculation/plotting
    """
    # case a:
    print("Starting case a ...")
    a_sample_std, a_average_volume = run_monte_carlo(
        mc=montecarlo,
        N=10000, 
        prng=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.75, 
        r=0.4, 
        throws=100)
    
    print("Starting convergence plot for case a...")
    convergencePlot(
        N=10000,
        radius=1.1, 
        k=1, 
        R=0.75, 
        r=0.4, 
        maxThrows=100,
        throwsSamples=10)


    print("Starting case b ...")
    # case b:
    b_sample_std, b_average_volume = run_monte_carlo(
        mc=montecarlo,
        N=10000, 
        prng=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.5, 
        r=0.5, 
        throws=100)
    
    print("Starting convergence plot for case b...")
    convergencePlot(
        N=10000,
        radius=1.1, 
        k=1, 
        R=0.7, 
        r=0.5, 
        maxThrows=100,
        throwsSamples=10)
 
    # plot points for case a
    montecarlo(prng=uniformrandom, radius=1.1, k=1, R=0.75, r=0.4, throws=10000, plot=True)

    # plot points for case b
    montecarlo(prng=uniformrandom, radius=1.1, k=1, R=0.5, r=0.5, throws=10000, plot=True)
    
    # Part 2: deterministic sampling
    plotDeterministicHistogram(int(1e5))

    # Part 3: off center measurements
    print("Starting off-center measurements ...")
    oc_sample_std, oc_average_volume = run_monte_carlo(
        mc=montecarlo,
        N=100000, 
        prng=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.75, 
        r=0.4, 
        throws=100,
        xc=0,
        yc=0,
        zc=0.1)

    # importance sampling (once)
    print("Starting importance sampling (once) ...")
    mc_importance(
        prng=uniformrandom, 
        b1_r=1.1, 
        k=1, 
        R=0.75, 
        r=0.4, 
        throws=100000, 
        xc=0, 
        yc=0, 
        zc=0.1, 
        plot=True,
        p=0.6)
    
    # Importance sampling (multiple)
    print("Starting importance sampling (multiple) ...")
    is_sample_std, is_average_volume = run_monte_carlo(
        mc=mc_importance,
        N=100000, 
        prng=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.75, 
        r=0.4, 
        throws=100,
        xc=0,
        yc=0,
        zc=0.1,
        p=0.6)
    

main()

















