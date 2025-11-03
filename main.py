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

    # normalise [a, b] -> [0, 1]
    sequence = (sequence - a) / (b - a)

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

    # normalise [0, 1] -> [-radius, radius]
    x, y, z = radius * (np.ones((3, throws)) - 2 * rand)

    sphereHits = sphere(x, y, z, k)
    torusHits = torus(x-xc, y-yc, z-zc, R, r)

    totalHits = np.sum(sphereHits & torusHits)

    box_volume = (2 * radius) ** 3
    intersection_volume = box_volume * (totalHits / throws)

    if plot == True:
        plotintersection(x, y, z, sphereHits, torusHits, radius)

    return intersection_volume, totalHits


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
    throwsList = np.logspace(1, np.log(maxThrows) / np.log(10), throwsSamples, base=10, dtype=np.int32)

    means = np.empty((2, throwsSamples))
    stds = np.empty((2, throwsSamples))
    for i in trange(throwsSamples, desc="Convergence plot: ", leave=False):
        throws = throwsList[i]
        vols = np.empty((2, N))
        for n in range(N):
            vols[0, n], _ = montecarlo(uniformrandom, radius, k, R, r, throws)
            vols[1, n], _ = montecarlo(deterministic_XYZ, radius, k, R, r, throws)

        means[:, i] = np.mean(vols, axis=1)
        stds[:, i] = np.std(vols, axis=1)

    fig, ax = plt.subplots()

    ax.errorbar(throwsList, means[0, :], yerr=stds[0, :], fmt='o', label="uniform")
    ax.errorbar(throwsList, means[1, :], yerr=stds[1, :], fmt='o', label="deterministic")
    ax.set_xscale("log")
    ax.set_xlabel("Amount of throws")
    ax.set_ylabel("Volume estimate (a.u.)")
    ax.legend()

    plt.show()

    return
        



# --------------
# Running code
# --------------
def main():
    # case a:

    a_sample_std, a_average_volume = run_monte_carlo(
        N=100, 
        prng=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.75, 
        r=0.4, 
        throws=int(1e5))

    # case b:
    b_sample_std, b_average_volume = run_monte_carlo(
        N=100, 
        prng=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.5, 
        r=0.5, 
        throws=int(1e5))
 
    # plot case a
    montecarlo(prng=uniformrandom, 
               radius=1.1, 
               k=1, 
               R=0.75, 
               r=0.4, 
               throws=int(1e4), 
               plot=True)

    convergencePlot(N=100,
                    radius=1.1,
                    k=1,
                    R=0.75,
                    r=0.4,
                    maxThrows=int(1e5),
                    throwsSamples=20)

    plotDeterministicHistogram(int(1e5))

main()