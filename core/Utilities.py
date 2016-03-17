import numpy as np

from math import pi

# Normalizes a numpy vector
# This methods modifies its argument in place, but also returns a reference to that
# array for chaining.
# Works on both single vectors and nx3 arrays of vectors (perfomed in-place).
# If zeroError=False, then this function while silently return a same-sized 0
# for low-norm vectors. If zeroError=True it will throw an exception
def normalize(vec, zeroError=False):

    # Used for testing zeroError
    eps = 0.00000000001

    # Use separate tests for 1D vs 2D arrays (TODO is there a nicer way to do this?)
    if(len(vec.shape) == 1):

        norm = np.linalg.norm(vec)
        if(norm < 0.0000001):
            if(zeroError):
                raise ArithmeticError("Cannot normalize function with norm near 0")
            else:
                vec[0] = 0
                vec[1] = 0
                vec[2] = 0
                return vec
        vec[0] /= norm
        vec[1] /= norm
        vec[2] /= norm
        return vec

    elif(len(vec.shape) == 2):

        # Compute norms for each vector
        norms = np.sqrt( vec[:,0]**2 + vec[:,1]**2 + vec[:,2]**2 )

        # Check for norm zero, if checking is enabled
        if(zeroError and np.any(norms < 0.00000000001)):
            raise ArithmeticError("Cannot normalize function with norm near 0")

        # Normalize in place
        # oldSettings = np.seterr(invalid='ignore')    # Silence warnings since we check above if the user cares
        vec[:,0] /= norms
        vec[:,1] /= norms
        vec[:,2] /= norms
        # np.seterr(**oldSettings)

    else:
        raise ValueError("I don't know how to normalize a vector array with > 2 dimensions")

    return vec


# Normalizes a numpy vector.
# This method returns a new (normalized) vector
# Works on both single vectors and nx3 arrays of vectors (perfomed in-place).
# If zeroError=False, then this function while silently return a same-sized 0
# for low-norm vectors. If zeroError=True it will throw an exception
def normalized(vec, zeroError=False):

    # Used for testing zeroError
    eps = 0.00000000001

    # Use separate tests for 1D vs 2D arrays (TODO is there a nicer way to do this?)
    if(len(vec.shape) == 1):

        norm = np.linalg.norm(vec)
        if(norm < 0.0000001):
            if(zeroError):
                raise ArithmeticError("Cannot normalize function with norm near 0")
            else:
                return np.zeros_like(vec)
        return vec / norm

    elif(len(vec.shape) == 2):

        # Compute norms for each vector
        norms = np.sqrt( vec[:,0]**2 + vec[:,1]**2 + vec[:,2]**2 )

        # Check for norm zero, if checking is enabled
        if(zeroError and np.any(norms < 0.00000000001)):
            raise ArithmeticError("Cannot normalize function with norm near 0")

        # Normalize in place
        # oldSettings = np.seterr(invalid='ignore')    # Silence warnings since we check above if the user cares
        vec = vec.copy()
        vec[:,0] /= norms
        vec[:,1] /= norms
        vec[:,2] /= norms
        # np.seterr(**oldSettings)

    else:
        raise ValueError("I don't know how to normalize a vector array with > 2 dimensions")

    return vec

# An alias for np.linal.norm, because typing that is ugly
def norm(vec, *args, **kwargs):
    return np.linalg.norm(vec, *args, **kwargs)


# A quicker cross method when calling on a single vector
def cross(u, v):
    return np.array((
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
        ))

def dot(u,v):
    return np.dot(u,v)

def clamp(val, lower = -float('inf'), upper = float('inf')):
    if val > upper:
        val = upper
    if val < lower:
        val = lower
    return val

def regAngle(theta):
    """
    Returns the argument mapped in to (-pi,pi]
    """
    while theta > pi: theta = theta - 2*pi
    while theta <= -pi: theta = theta + 2*pi
    return theta

def circlePairs(lst):
    """
    Iterate through a list returning [i],[(i+1)%N] circular pairs, including the
    (last,first) pair
    """
    i = iter(lst)
    first = prev = item = i.next()
    for item in i:
        yield prev, item
        prev = item
    yield item, first


## A wrapper for the purposes of this class, to avoid interacting with numpy
def Vector3D(x,y,z):
    return np.array([float(x),float(y),float(z)])

def printVec3(v):
    return "({:.5f}, {:.5f}, {:.5f})".format(v[0], v[1], v[2])
