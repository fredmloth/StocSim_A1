import numpy as np
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

# Uniform random sampling

# Sampling function for volume intersection with bounding box? -> check what is bounding box
# Bounding box: space that encompases both together fully from which we sample 
# do not sample any 3 combinations multiple times