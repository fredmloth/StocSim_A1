# This module runs the code for assignment 1 of Stochastic Simulation

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
# Exact solution
# --------------
def volumeAnalytical(k, R, r):
    """Analytically determines the volume, only works for equally
    centered bodies."""

    R_1 = (R ** 2 + k ** 2 - r ** 2) / (2 * R)

    F_1 = lambda x, a : -np.power(a ** 2 - x ** 2, 1.5) / 3
    F_2 = lambda x, a : (np.arcsin(x / a) + 0.5 * np.sin(2 * np.arcsin(x / a))) * 0.5 * a ** 2

    V = F_1(k, k) - F_1(R_1, k)
    V += F_1(R_1 - R, r) - F_1(-r, r)
    V += R * (F_2(R_1 - R, r) - F_2(-r, r))

    return V * 4 * np.pi


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

    # normalise [a, b] -> [0, 1]
    sequence = (sequence - a) / (b - a)

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

    for _ in range(N):
        intersection_volume, hits = mc(prng, radius, k, R, r, throws, xc, yc, zc, p)
        all_volumes.append(intersection_volume)
        all_hits.append(hits)

    # Determines standard deviation and average volume
    sample_std = np.std(all_volumes)
    average_volume = np.mean(all_volumes)

    print(f"for radius={radius}, k={k}, R={R}, r={r}, and throws={throws}, we get:")
    print(f"average_volume: {average_volume}, sample variance: {sample_std}")

    return sample_std, average_volume, all_volumes


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
    ax.set_title(f"Intersection")
    ax.legend()
    
    # set limits for each axis
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([-radius, radius])

    fig.savefig("intersection_plot.png")
    plt.show()


# plot histogram for deterministic sequence -> show that it is not uniform
def plotDeterministicHistogram(N):
    xDet, _, _ = deterministic_XYZ(N)
    xRand, _, _ = uniformrandom(N)

    fig, ax = plt.subplots()
    ax.hist(xDet, density=False, label="deterministic")
    ax.hist(xRand, density=False, histtype="step", label="numpy prng")
    ax.set_xlabel("Sample value")
    ax.set_ylabel("Density of occurence")
    ax.legend()
    
    fig.savefig("deterministic_histogram.png")
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

    for i in range(throwsSamples):
        throws = throwsList[i]
        vols = np.empty((2, N))
        
        for n in range(N):
            vols[0, n], _ = montecarlo(uniformrandom, radius, k, R, r, throws)
            vols[1, n], _ = montecarlo(deterministic_XYZ, radius, k, R, r, throws)

        means[:, i] = np.mean(vols, axis=1)
        stds[:,  i] = np.std(vols, axis=1)

    exactVolume = volumeAnalytical(k, R, r)

    # plot mean with std as error bars
    fig, ax = plt.subplots()
    ax.errorbar(throwsList, means[0, :], yerr=stds[0, :], fmt='o', label="uniform (mean ± 1σ)", capsize=3, elinewidth=1)
    ax.errorbar(throwsList, means[1, :], yerr=stds[1, :], fmt='o', label="deterministic (mean ± 1σ)", capsize=3, elinewidth=1)
    ax.hlines(exactVolume, np.min(throwsList), np.max(throwsList), linestyle="dashed", color="grey", label="exact")
    ax.set_xscale("log")
    ax.set_xlabel("Amount of throws")
    ax.set_ylabel("Mean Volume (± Std. Deviation)")
    ax.set_title(f"Convergence Plot (N={N} repeats)")
    ax.legend()

    fig.savefig("convergence_plot.png")
    plt.show()


def plot_pvalues(
    mean_volumes,          # shape [P]
    all_volumes_by_p,      # list of length P, each is [N] runs for that p
    p_values,              # shape [P]
    sem=None,              # optional SEM per p (std/sqrt(N))
    show_std=True,         # also show sample std across runs on the error panel
    title_prefix='Importance sampling'
):
    p = np.asarray(p_values, dtype=float)
    mu = np.asarray(mean_volumes, dtype=float)

    # min/max and std across runs for each p
    vmin = np.array([np.min(v) for v in all_volumes_by_p], dtype=float)
    vmax = np.array([np.max(v) for v in all_volumes_by_p], dtype=float)
    sdev = np.array([np.std(v, ddof=1) for v in all_volumes_by_p], dtype=float)

    # consistent left-to-right order
    order = np.argsort(p)
    p, mu, vmin, vmax, sdev = p[order], mu[order], vmin[order], vmax[order], sdev[order]
    if sem is not None:
        sem = np.asarray(sem, dtype=float)[order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # --- Left: mean + min–max envelope (no error bars) ---
    ax = axes[0]
    ax.plot(p, mu, 'o-', label='mean')
    ax.fill_between(p, vmin, vmax, alpha=0.2, label='min–max across runs')
    ax.set_xlabel('p')
    ax.set_ylabel('Estimated volume')
    ax.set_title(f'{title_prefix}: mean with min–max envelope')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Right: error-focused view (SEM / std vs p) ---
    ax = axes[1]
    lines = []
    if sem is not None:
        lines += ax.plot(p, sem, 'o-', label='SEM')
    if show_std:
        lines += ax.plot(p, sdev, 'o--', label='Std across runs')
    ax.set_xlabel('p')
    ax.set_ylabel('Error')
    ax.set_title('Error vs p (lower is better)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig("importance_sampling.png")
    plt.show()


# --------------
# Running code
# --------------
def main():
    # case a:
    print("Starting case a ...")
    a_sample_std, a_average_volume, _ = run_monte_carlo(
        mc=montecarlo,
        N=100, 
        prng=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.75, 
        r=0.4, 
        throws=10000)
    
    print("Starting convergence plot for case a...")
    convergencePlot(
        N=100,
        radius=1.1, 
        k=1, 
        R=0.75, 
        r=0.4, 
        maxThrows=int(1e5),
        throwsSamples=10)


    print("Starting case b ...")
    # case b:
    b_sample_std, b_average_volume, _ = run_monte_carlo(
        mc=montecarlo,
        N=100, 
        prng=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.5, 
        r=0.5, 
        throws=10000)
    
    print("Starting convergence plot for case b...")
    convergencePlot(
        N=100,
        radius=1.1, 
        k=1, 
        R=0.5, 
        r=0.5, 
        maxThrows=10000,
        throwsSamples=10)
 
    # plot points for case a
    montecarlo(prng=uniformrandom, radius=1.1, k=1, R=0.75, r=0.4, throws=10000, plot=True)

    # plot points for case b
    montecarlo(prng=uniformrandom, radius=1.1, k=1, R=0.5, r=0.5, throws=10000, plot=True)
    
    # Part 2: deterministic sampling
    plotDeterministicHistogram(int(1e5))

    # plot points for case a deterministic sampling
    montecarlo(prng=deterministic_XYZ, radius=1.1, k=1, R=0.75, r=0.4, throws=10000, plot=True)

    # plot points for case b deterministic sampling
    montecarlo(prng=deterministic_XYZ, radius=1.1, k=1, R=0.5, r=0.5, throws=10000, plot=True)


    # Part 3: off center measurements
    print("Starting off-center measurements ...")
    oc_sample_std, oc_average_volume, _ = run_monte_carlo(
        mc=montecarlo,
        N=100, 
        prng=uniformrandom, 
        radius=1.1, 
        k=1, 
        R=0.75, 
        r=0.4, 
        throws=10000,
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
    

    #Importance sampling (different p_values)
    p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    is_avg_volumes = []
    is_all_volumes = []
    is_std = []
    N = 100

    for p_value in p_values:
        print(f"Starting importance sampling for p = {p_value} ...")
        std, volume, all_volumes = run_monte_carlo(
            mc=mc_importance,
            N=N, 
            prng=uniformrandom, 
            radius=1.1, 
            k=1, 
            R=0.75, 
            r=0.4, 
            throws=1000000,
            xc=0,
            yc=0,
            zc=0.1,
            p=p_value)
        is_avg_volumes.append(volume)
        is_all_volumes.append(all_volumes)
        is_std.append(std)
    
    s_error = np.array(is_std) / np.sqrt(N)

    # Simple min–max shading + mean line
    plot_pvalues(
        mean_volumes=is_avg_volumes,
        all_volumes_by_p=is_all_volumes,
        p_values=p_values,
        sem=s_error)
    
    
main()
