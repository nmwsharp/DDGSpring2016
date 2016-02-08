# Basic application to load a mesh from file and view it in a window

# Python imports
import sys, os
import euclid as eu
import time
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

## Imports from this project
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core')) # hack to allow local imports without creaing a module or modifying the path variable
from InputOutput import *
from MeshDisplay import MeshDisplay
from HalfEdgeMesh import *
from Utilities import *

def main():

    # Get the path for the mesh to load from the program argument
    if(len(sys.argv) == 3):
        partString = sys.argv[1]
        if partString not in ['part1','part2','part3']:
            print("ERROR part specifier not recognized. Should be one of 'part1', 'part2', or 'part3'")
            exit()
        filename = sys.argv[2]
    else:
        print("ERROR: Incorrect call syntax. Proper syntax is 'python Assignment3.py partN path/to/your/mesh.obj'.")
        exit()

    # Read in the mesh
    mesh = HalfEdgeMesh(readMesh(filename))

    # Create a viewer object
    winName = 'DDG Assignment3 ' + partString + '-- ' + os.path.basename(filename)
    meshDisplay = MeshDisplay(windowTitle=winName)
    meshDisplay.setMesh(mesh)



    ###################### BEGIN YOUR CODE
    # implement the body of each of these functions

    ############################
    # Part 0: Helper functions #
    ############################
    # Implement a few useful functions that you will want in the remainder of
    # the assignment.

    @property
    @cacheGeometry
    def cotanWeight(self):
        """
        Return the cotangent weight for an edge. Since this gets called on
        an edge, 'self' will be a reference to an edge.

        This will be useful in the problems below.

        Don't forget, everything you implemented for the last homework is now
        available as part of the library (normals, areas, etc). (Moving forward,
        Vertex.normal will mean area-weighted normals, unless otherwise specified)
        """

        return 0.0 # placeholder value


    @property
    @cacheGeometry
    def dualArea(self):
        """
        Return the dual area associated with a vertex. Since this gets called on
        a vertex, 'self' will be a reference to a vertex.

        Recall that the dual area can be defined as 1/3 the area of the surrounding
        faces.
        """

        return 0.0 # placeholder value


    def enumerateVertices(mesh):
        """
        Assign a unique index from 0 to (N-1) to each vertex in the mesh. Should
        return a dictionary containing mappings {vertex ==> index}.

        You will want to use this function in your solutions below.
        """

        return None # placeholder value


    #################################
    # Part 1: Dense Poisson Problem #
    #################################
    # Solve a Poisson problem on the mesh. The primary function here
    # is solvePoissonProblem_dense(), it will get called when you run
    #   python Assignment3.py part1 /path/to/your/mesh.obj
    # and specify density values with the mouse (the press space to solve).
    #
    # Note that this code will be VERY slow on large meshes, because it uses
    # dense matrices.

    def buildLaplaceMatrix_dense(mesh, index):
        """
        Build a Laplace operator for the mesh, with a dense representation

        'index' is a dictionary mapping {vertex ==> index}

        Returns the resulting matrix.
        """

        return None # placeholder value


    def buildMassMatrix_dense(mesh, index):
        """
        Build a mass matrix for the mesh.

        Returns the resulting matrix.
        """

        return None # placeholder value


    def solvePoissonProblem_dense(mesh, densityValues):
        """
        Solve a Poisson problem on the mesh. The results should be stored on the
        vertices in a variable named 'solutionVal'. You will want to make use
        of your buildLaplaceMatrix_dense() function from above.

        densityValues is a dictionary mapping {vertex ==> value} that specifies
        densities. The density is implicitly zero at every vertex not in this
        dictionary.

        When you run this program with 'python Assignment3.py part1 path/to/your/mesh.obj',
        you will get to click on vertices to specify density conditions. See the
        assignment document for more details.
        """

        pass # remove this line once you have implemented the method


    ##################################
    # Part 2: Sparse Poisson Problem #
    ##################################
    # Solve a Poisson problem on the mesh. The primary function here
    # is solvePoissonProblem_sparse(), it will get called when you run
    #   python Assignment3.py part2 /path/to/your/mesh.obj
    # and specify density values with the mouse (the press space to solve).
    #
    # This will be very similar to the previous part. Be sure to see the wiki
    # for notes about the nuances of sparse matrix computation. Now, your code
    # should scale well to larger meshes!

    def buildLaplaceMatrix_sparse(mesh, index):
        """
        Build a laplace operator for the mesh, with a sparse representation.
        This will be nearly identical to the dense method.

        'index' is a dictionary mapping {vertex ==> index}

        Returns the resulting sparse matrix.
        """

        return None # placeholder value

    def buildMassMatrix_sparse(mesh, index):
        """
        Build a sparse mass matrix for the system.

        Returns the resulting sparse matrix.
        """

        return None # placeholder value


    def solvePoissonProblem_sparse(mesh, densityValues):
        """
        Solve a Poisson problem on the mesh, using sparse matrix operations.
        This will be nearly identical to the dense method.
        The results should be stored on the vertices in a variable named 'solutionVal'.

        densityValues is a dictionary mapping {vertex ==> value} that specifies any
        densities. The density is implicitly zero at every vertex not in this dictionary.

        Note: Be sure to look at the notes on the github wiki about sparse matrix
        computation in Python.

        When you run this program with 'python Assignment3.py part2 path/to/your/mesh.obj',
        you will get to click on vertices to specify density conditions. See the
        assignment document for more details.
        """


        pass # remove this line once you have implemented the method


    ###############################
    # Part 3: Mean Curvature Flow #
    ###############################
    # Perform mean curvature flow on the mesh. The primary function here
    # is meanCurvatureFlow(), which will get called when you run
    #   python Assignment3.py part3 /path/to/your/mesh.obj
    # You can adjust the step size with the 'z' and 'x' keys, and press space
    # to perform one step of flow.
    #
    # Of course, you will want to use sparse matrices here, so your code
    # scales well to larger meshes.

    def buildMeanCurvatureFlowOperator(mesh, index, h):
        """
        Construct the (sparse) mean curvature operator matrix for the mesh.
        It might be helpful to use your buildLaplaceMatrix_sparse() and
        buildMassMatrix_sparse() methods from before.

        Returns the resulting matrix.
        """

        return None # placeholder value

    def meanCurvatureFlow(mesh, h):
        """
        Perform mean curvature flow on the mesh. The result of this operation
        is updated positions for the vertices; you should conclude by modifying
        the position variables for the mesh vertices.

        h is the step size for the backwards euler integration.

        When you run this program with 'python Assignment3.py part3 path/to/your/mesh.obj',
        you can press the space bar to perform this operation and z/x to change
        the step size.

        Recall that before you modify the positions of the mesh, you will need
        to set mesh.staticGeometry = False, which disables caching optimizations
        but allows you to modfiy the geometry. After you are done modfiying
        positions, you should set mesh.staticGeometry = True to re-enable these
        optimizations. You should probably have mesh.staticGeometry = True while
        you assemble your operator, or it will be very slow.
        """

        pass # remove this line once you have implemented the method



    ###################### END YOUR CODE

    Edge.cotanWeight = cotanWeight
    Vertex.dualArea = dualArea

    # A pick function for choosing density conditions
    densityValues = dict()
    def pickVertBoundary(vert):
        value = 1.0 if pickVertBoundary.isHigh else -1.0
        print("   Selected vertex at position:" + printVec3(vert.position))
        print("   as a density with value = " + str(value))
        densityValues[vert] = value
        pickVertBoundary.isHigh = not pickVertBoundary.isHigh
    pickVertBoundary.isHigh = True



    # Run in part1 mode
    if partString == 'part1':

        print("\n\n === Executing assignment 2 part 1")
        print("""
        Please click on vertices of the mesh to specify density conditions.
        Alternating clicks will specify high-value (= 1.0) and low-value (= -1.0)
        density conditions. You may select as many density vertices as you want,
        but >= 2 are necessary to yield an interesting solution. When you are done,
        press the space bar to execute your solver and view the results.
        """)

        meshDisplay.pickVertexCallback = pickVertBoundary
        meshDisplay.drawVertices = True

        def executePart1Callback():
            print("\n=== Solving Poisson problem with your dense solver\n")

            # Print and check the density values
            print("Density values:")
            for key in densityValues:
                print("    " + str(key) + " = " + str(densityValues[key]))
            if len(densityValues) < 2:
                print("Aborting solve, not enough density vertices specified")
                return

            # Call the solver
            print("\nSolving problem...")
            t0 = time.time()
            solvePoissonProblem_dense(mesh, densityValues)
            tSolve = time.time() - t0
            print("...solution completed.")
            print("Solution took {:.5f} seconds.".format(tSolve))

            print("Visualizing results...")

            # Error out intelligently if nothing is stored on vert.solutionVal
            for vert in mesh.verts:
                if not hasattr(vert, 'solutionVal'):
                    print("ERROR: At least one vertex does not have the attribute solutionVal defined.")
                    exit()
                if not isinstance(vert.solutionVal, float):
                    print("ERROR: The data stored at vertex.solutionVal is not of type float.")
                    print("   The data has type=" + str(type(vert.solutionVal)))
                    print("   The data looks like vert.solutionVal="+str(vert.solutionVal))
                    exit()

            # Visualize the result
            # meshDisplay.setShapeColorFromScalar("solutionVal", definedOn='vertex', cmapName="seismic", vMinMax=[-1.0,1.0])
            meshDisplay.setShapeColorFromScalar("solutionVal", definedOn='vertex', cmapName="seismic")
            meshDisplay.generateAllMeshValues()

        meshDisplay.registerKeyCallback(' ', executePart1Callback, docstring="Solve the Poisson problem and view the results")

        # Start the GUI
        meshDisplay.startMainLoop()




    # Run in part2 mode
    elif partString == 'part2':
        print("\n\n === Executing assignment 2 part 2")
        print("""
        Please click on vertices of the mesh to specify density conditions.
        Alternating clicks will specify high-value (= 1.0) and low-value (= -1.0)
        density conditions. You may select as many density vertices as you want,
        but >= 2 are necessary to yield an interesting solution. When you are done,
        press the space bar to execute your solver and view the results.
        """)

        meshDisplay.pickVertexCallback = pickVertBoundary
        meshDisplay.drawVertices = True

        def executePart2Callback():
            print("\n=== Solving Poisson problem with your sparse solver\n")

            # Print and check the density values
            print("Density values:")
            for key in densityValues:
                print("    " + str(key) + " = " + str(densityValues[key]))
            if len(densityValues) < 2:
                print("Aborting solve, not enough density vertices specified")
                return

            # Call the solver
            print("\nSolving problem...")
            t0 = time.time()
            solvePoissonProblem_sparse(mesh, densityValues)
            tSolve = time.time() - t0
            print("...solution completed.")
            print("Solution took {:.5f} seconds.".format(tSolve))

            print("Visualizing results...")

            # Error out intelligently if nothing is stored on vert.solutionVal
            for vert in mesh.verts:
                if not hasattr(vert, 'solutionVal'):
                    print("ERROR: At least one vertex does not have the attribute solutionVal defined.")
                    exit()
                if not isinstance(vert.solutionVal, float):
                    print("ERROR: The data stored at vertex.solutionVal is not of type float.")
                    print("   The data has type=" + str(type(vert.solutionVal)))
                    print("   The data looks like vert.solutionVal="+str(vert.solutionVal))
                    exit()

            # Visualize the result
            # meshDisplay.setShapeColorFromScalar("solutionVal", definedOn='vertex', cmapName="seismic", vMinMax=[-1.0,1.0])
            meshDisplay.setShapeColorFromScalar("solutionVal", definedOn='vertex', cmapName="seismic")
            meshDisplay.generateAllMeshValues()

        meshDisplay.registerKeyCallback(' ', executePart2Callback, docstring="Solve the Poisson problem and view the results")

        # Start the GUI
        meshDisplay.startMainLoop()



    # Run in part3 mode
    elif partString == 'part3':

        print("\n\n === Executing assignment 2 part 3")
        print("""
        Press the space bar to perform one step of mean curvature
        flow smoothing, using your solver. Pressing the 'z' and 'x'
        keys will decrease and increase the step size (h), respectively.
        """)


        stepSize = [0.01]
        def increaseStepsize():
            stepSize[0] += 0.001
            print("Increasing step size. New size h="+str(stepSize[0]))
        def decreaseStepsize():
            stepSize[0] -= 0.001
            print("Decreasing step size. New size h="+str(stepSize[0]))
        meshDisplay.registerKeyCallback('z', decreaseStepsize, docstring="Increase the value of the step size (h) by 0.1")
        meshDisplay.registerKeyCallback('x', increaseStepsize, docstring="Decrease the value of the step size (h) by 0.1")



        def smoothingStep():
            print("\n=== Performing mean curvature smoothing step\n")
            print("  Step size h="+str(stepSize[0]))

            # Call the solver
            print("  Solving problem...")
            t0 = time.time()
            meanCurvatureFlow(mesh, stepSize[0])
            tSolve = time.time() - t0
            print("  ...solution completed.")
            print("  Solution took {:.5f} seconds.".format(tSolve))

            print("Updating display...")
            meshDisplay.generateAllMeshValues()

        meshDisplay.registerKeyCallback(' ', smoothingStep, docstring="Perform one step of your mean curvature flow on the mesh")

        # Start the GUI
        meshDisplay.startMainLoop()




if __name__ == "__main__": main()
