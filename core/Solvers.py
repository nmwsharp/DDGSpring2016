import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import time

# PyAMG algebraic multigrid
from pyamg import *

def solvePoisson(mesh, boundaryVals=dict(), method='numpy-sparse', verbose=True):
    """
    Solve a Poisson problem on a mesh. Returns a vertex => value dictionary containing
    every vertex in the mesh (including any boundary vertices).
        boundaryVals: a {vertex => float} dictionary containing Dirichlet boundary
            conditions
        method: which solver should be used
                    {numpy-sparse, numpy-dense, pyamg-RS}
    """

    if verbose:
        print("Solving Poisson problem on mesh with " + str(len(mesh.verts)) + " vertices")
        print("  - Dirichlet boundary conditions were given at " + str(len(boundaryVals)) + " vertices")
        print("  - Using method: " + method)

    # Make sure a valid method was given
    if method not in ['numpy-sparse', 'numpy-dense','pyamg-RS']:
        raise ValueError("ERROR: Invalid method ("+method+") given to solvePoisson")

    # Announce setup and start timing
    if verbose:
        print("  Setting up problem...")
        t0 = time.time()


    # Enumerate the interior vertices at which we will compute a solution
    interiorVerts = mesh.verts - set(boundaryVals.keys())
    nInteriorVerts = len(interiorVerts)
    ind = mesh.enumerateVertices(interiorVerts)

    ## Build the poisson matrix

    if method == 'numpy-dense':
        A = np.zeros((nInteriorVerts,nInteriorVerts))
    else:
        A = lil_matrix((nInteriorVerts, nInteriorVerts))
    b = np.zeros((nInteriorVerts,1))

    # Iterate over the interior vertices, adding terms for the neighbors of each
    for vert in interiorVerts:
        i = ind[vert]
        for (neighEdge, neighVert) in vert.adjacentEdgeVertexPairs():
            w = neighEdge.cotanWeight

            # Boundary neighbor
            if neighVert in boundaryVals:
                A[i,i] -= w
                b[i] -= w * boundaryVals[neighVert]
            # Interior neighbor
            else:
                j = ind[neighVert]
                A[i,j] += w
                A[i,i] -= w

    # Announce solve and more timing
    if verbose:
        tSetup = time.time() - t0
        print("  ...setup complete.")
        print("  Solving system...")
        t0 = time.time()

    # Solve the system of equations

    if method == 'numpy-dense':
        sol = np.linalg.solve(A,b)
    elif method == 'numpy-sparse':
        A = A.tocsr()
        sol = spsolve(A, b)
    elif method == 'pyamg-RS':
        A = A.tocsr()
        ml = ruge_stuben_solver(A)
        sol = ml.solve(b, tol=1e-8)

    # Announce solve completed and timing info
    if verbose:
        tSolve = time.time() - t0
        print("  ...solve complete.")
        print("  setup time: "+str(tSetup)+" solve time: " + str(tSolve))


    # Build the result dictionary
    res = dict()
    for v in interiorVerts:
        res[v] = sol[ind[v]]

    # Add the values from the boundary conditions
    res.update(boundaryVals)

    return res
