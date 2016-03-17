# Basic application to load a mesh from file and view it in a window

# Python imports
import sys, os
import euclid as eu
import time
import random
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

## Imports from this project
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core')) # hack to allow local imports without creaing a module or modifying the path variable
from InputOutput import *
from MeshDisplay import MeshDisplay
from HalfEdgeMesh import *
from Utilities import *
from Solvers import solvePoisson


def main():

    # Get the path for the mesh to load from the program argument
    if(len(sys.argv) == 3 and sys.argv[1] == 'simple'):
        filename = sys.argv[2]
        simpleTest = True
    elif(len(sys.argv) == 3 and sys.argv[1] == 'fancy'):
        filename = sys.argv[2]
        simpleTest = False 
    else:
        print("ERROR: Incorrect call syntax. Proper syntax is 'python Assignment5.py MODE path/to/your/mesh.obj', where MODE is either 'simple' or 'fancy'")
        exit()

    # Read in the mesh
    mesh = HalfEdgeMesh(readMesh(filename))


    # Create a viewer object
    winName = 'DDG Assignment5 -- ' + os.path.basename(filename)
    meshDisplay = MeshDisplay(windowTitle=winName)
    meshDisplay.setMesh(mesh)


    ###################### BEGIN YOUR CODE
    
    # DDGSpring216 Assignment 5
    # 
    # In this programming assignment you will implement Helmholtz-Hodge decomposition of covectors.
    #
    # The relevant mathematics and algorithm are described in section 8.1 of the course notes.
    # You will also need to implement the core operators in discrete exterior calculus, described mainly in 
    # section 3.6 of the course notes.
    #
    # This code can be run with python Assignment5.py MODE /path/to/you/mesh.obj. MODE should be
    # either 'simple' or 'fancy', corresponding to the complexity of the input field omega that is given.
    # It might be easier to debug your algorithm on the simple field first. The assignment code will read in your input 
    # mesh, generate a field 'omega' as input, run your algorithm, then display the results.
    # The results can be viewed as streamlines on the surface that flow with the covector field (toggle with 'p'),
    # or, as actual arrows on the faces (toggle with 'l'). The keys '1'-'4' will switch between the input, exact,
    # coexact, and harmonic fields (respectively).
    # 
    # A few hints:
    #   - Try performing some basic checks on your operators if things don't seem right. For instance, applying the 
    #     exterior derivative twice to anything should always yield zero.
    #   - The streamline visualization is easy to look at, but can be deceiving at times. For instance, streamlines
    #     are not very meaningful where the actual covectors are near 0. Try looking at the actual arrows in that case
    #     ('l').
    #   - Many inputs will not have any harmonic components, especially genus 0 inputs. Don't stress if the harmonic 
    #     component of your output is exactly or nearly zero.
    
    
    # Implement the body of each of these functions...
   
    def assignEdgeOrientations(mesh):
        """
        Assign edge orientations to each edge on the mesh.
        
        This method will be called from the assignment code, you do not need to explicitly call it in any of your methods.

        After this method, the following values should be defined:
            - edge.orientedHalfEdge (a reference to one of the halfedges touching that edge)
            - halfedge.orientationSign (1.0 if that halfedge agrees with the orientation of its
                edge, or -1.0 if not). You can use this to make much of your subsequent code cleaner.

        This is a pretty simple method to implement, any choice of orientation is acceptable.
        """

        pass # remove once you have implemented

    def diagonalInverse(A):
        """
        Returns the inverse of a sparse diagonal matrix. Makes a copy of the matrix.
        
        We will need to invert several diagonal matrices for the algorithm, but scipy does
        not offer a fast method for inverting diagonal matrices, which is a very easy special
        case. As such, this is a useful helper method for you.

        Note that the diagonal inverse is not well-defined if any of the diagonal elements are
        0.0. This needs to be acconuted for when you construct the matrices.
        """

        return None # placeholder
    

    @property
    @cacheGeometry
    def circumcentricDualArea(self):
        """
        Compute the area of the circumcentric dual cell for this vertex. Returns a positive scalar.

        This gets called on a vertex, so 'self' will be a reference to the vertex.

        The image on page 78 of the course notes may help you visualize this.
        """
        
        return 0.0 # placeholder
    Vertex.circumcentricDualArea = circumcentricDualArea


    def buildHodgeStar0Form(mesh, vertexIndex):
        """
        Build a sparse matrix encoding the Hodge operator on 0-forms for this mesh.
        Returns a sparse, diagonal matrix corresponding to vertices.

        The discrete hodge star is a diagonal matrix where each entry is
        the (area of the dual element) / (area of the primal element). You will probably
        want to make use of the Vertex.circumcentricDualArea property you just defined.

        By convention, the area of a vertex is 1.0.
        """
       
        return None # placeholder

    
    def buildHodgeStar1Form(mesh, edgeIndex):
        """
        Build a sparse matrix encoding the Hodge operator on 1-forms for this mesh.
        Returns a sparse, diagonal matrix corresponding to edges.
        
        The discrete hodge star is a diagonal matrix where each entry is
        the (area of the dual element) / (area of the primal element). The solution
        to exercise 26 from the previous homework will be useful here.

        Note that for some geometries, some entries of hodge1 operator may be exactly 0.
        This can create a problem when we go to invert the matrix. To numerically sidestep
        this issue, you probably want to add a small value (like 10^-8) to these diagonal 
        elements to ensure all are nonzero without significantly changing the result.
        """
        
        return None # placeholder
    
    
    def buildHodgeStar2Form(mesh, faceIndex):
        """
        Build a sparse matrix encoding the Hodge operator on 2-forms for this mesh
        Returns a sparse, diagonal matrix corresponding to faces.

        The discrete hodge star is a diagonal matrix where each entry is
        the (area of the dual element) / (area of the primal element).

        By convention, the area of a vertex is 1.0.
        """
        
        return None # placeholder

    
    def buildExteriorDerivative0Form(mesh, edgeIndex, vertexIndex):
        """
        Build a sparse matrix encoding the exterior derivative on 0-forms.
        Returns a sparse matrix.

        See section 3.6 of the course notes for an explanation of DEC.
        """
        
        return None # placeholder
    
    def buildExteriorDerivative1Form(mesh, faceIndex, edgeIndex):
        """
        Build a sparse matrix encoding the exterior derivative on 1-forms.
        Returns a sparse matrix.
         
        See section 3.6 of the course notes for an explanation of DEC.
        """
        
        return None # placeholder

    def decomposeField(mesh):
        """
        Decompose a covector in to exact, coexact, and harmonic components

        The input mesh will have a scalar named 'omega' on its edges (edge.omega)
        representing a discretized 1-form. This method should apply Helmoltz-Hodge 
        decomposition algorithm (as described on page 107-108 of the course notes) 
        to compute the exact, coexact, and harmonic components of omega.

        This method should return its results by storing three new scalars on each edge, 
        as the 3 decomposed components: edge.exactComponent, edge.coexactComponent,
        and edge.harmonicComponent.

        Here are the primary steps you will need to perform for this method:
            
            - Create indexer objects for the vertices, faces, and edges. Note that the mesh
              has handy helper functions pre-defined for each of these: mesh.enumerateEdges() etc.
            
            - Build all of the operators we will need using the methods you implemented above:
              hodge0, hodge1, hodge2, d0, and d1. You should also compute their inverses and
              transposes, as appropriate.

            - Build a vector which represents the input covector (from the edge.omega values)

            - Perform a linear solve for the exact component, as described in the algorithm
            
            - Perform a linear solve for the coexact component, as described in the algorithm

            - Compute the harmonic component as the part which is neither exact nor coexact

            - Store your resulting exact, coexact, and harmonic components on the mesh edges

        This method will be called by the assignment code, you do not need to call it yourself.
        """

        pass # remove once you have implemented


    ###################### END YOUR CODE


    ### More prep functions
    def generateFieldConstant(mesh):
        print("\n=== Using constant field as arbitrary direction field")
        for vert in mesh.verts:
            vert.vector = vert.projectToTangentSpace(Vector3D(1.4, 0.2, 2.4))

    def generateFieldSimple(mesh):
        for face in mesh.faces:
            face.vector = face.center + Vector3D(-face.center[2], face.center[1], face.center[0])
            face.vector = face.projectToTangentSpace(face.vector)

    def gradFromPotential(mesh, potAttr, gradAttr):
        # Simply compute gradient from potential
        for vert in mesh.verts:
            sumVal = Vector3D(0.0,0.0,0.0)
            sumWeight = 0.0
            vertVal = getattr(vert, potAttr)
            for he in vert.adjacentHalfEdges():
                sumVal += he.edge.cotanWeight * (getattr(he.vertex, potAttr) - vertVal) * he.vector
                sumWeight += he.edge.cotanWeight
            setattr(vert, gradAttr, normalize(sumVal))

    def generateInterestingField(mesh):
        print("\n=== Generating a hopefully-interesting field which has all three types of components\n")


        # Somewhat cheesy hack: 
        # We want this function to generate the exact same result on repeated runs of the program to make
        # debugging easier. This means ensuring that calls to random.sample() return the exact same result
        # every time. Normally we could just set a seed for the RNG, and this work work if we were sampling
        # from a list. However, mesh.verts is a set, and Python does not guarantee consistency of iteration
        # order between runs of the program (since the default hash uses the memory address, which certainly
        # changes). Rather than doing something drastic like implementing a custom hash function on the 
        # mesh class, we'll just build a separate data structure where vertices are sorted by position,
        # which allows reproducible sampling (as long as positions are distinct).
        sortedVertList = list(mesh.verts)
        sortedVertList.sort(key= lambda x : (x.position[0], x.position[1], x.position[2]))
        random.seed(777)


        # Generate curl-free (ish) component
        curlFreePotentialVerts = random.sample(sortedVertList, max((2,len(mesh.verts)/1000)))
        potential = 1.0
        bVals = {}
        for vert in curlFreePotentialVerts:
            bVals[vert] = potential
            potential *= -1
        smoothPotential = solvePoisson(mesh, bVals)
        mesh.applyVertexValue(smoothPotential, "curlFreePotential")
        gradFromPotential(mesh, "curlFreePotential", "curlFreeVecGen")


        # Generate divergence-free (ish) component
        divFreePotentialVerts = random.sample(sortedVertList, max((2,len(mesh.verts)/1000)))
        potential = 1.0
        bVals = {}
        for vert in divFreePotentialVerts:
            bVals[vert] = potential
            potential *= -1
        smoothPotential = solvePoisson(mesh, bVals)
        mesh.applyVertexValue(smoothPotential, "divFreePotential")
        gradFromPotential(mesh, "divFreePotential", "divFreeVecGen")
        for vert in mesh.verts:
            normEu = eu.Vector3(*vert.normal)
            vecEu = eu.Vector3(*vert.divFreeVecGen)
            vert.divFreeVecGen = vecEu.rotate_around(normEu, pi / 2.0) # Rotate the field by 90 degrees


        # Combine the components
        for face in mesh.faces:
            face.vector = Vector3D(0.0, 0.0, 0.0)
            for vert in face.adjacentVerts():
                face.vector = 1.0 * vert.curlFreeVecGen + 1.0 * vert.divFreeVecGen
            
            face.vector = face.projectToTangentSpace(face.vector)

        
        # clear out leftover attributes to not confuse people
        for vert in mesh.verts:
            del vert.curlFreeVecGen
            del vert.curlFreePotential
            del vert.divFreeVecGen
            del vert.divFreePotential


    # Verify the orientations were defined. Need to do this early, since they are needed for setup
    def checkOrientationDefined(mesh):
        """Verify that edges have oriented halfedges and halfedges have orientation signs"""
    
        for edge in mesh.edges:
            if not hasattr(edge, 'orientedHalfEdge'):
                print("ERROR: Edges do not have orientedHalfEdge defined. Cannot proceed")
                exit()
        for he in mesh.halfEdges:
            if not hasattr(he, 'orientationSign'):
                print("ERROR: halfedges do not have orientationSign defined. Cannot proceed")
                exit()


    # Verify the correct properties are defined after the assignment is run
    def checkResultTypes(mesh):
        
        for edge in mesh.edges:
            # Check exact
            if not hasattr(edge, 'exactComponent'):
                print("ERROR: Edges do not have edge.exactComponent defined. Cannot proceed")
                exit()
            if not isinstance(edge.exactComponent, float):
                print("ERROR: edge.exactComponent is defined, but has the wrong type. Type is " + str(type(edge.exactComponent)) + " when if should be 'float'")
                exit()
        
            # Check cocoexact
            if not hasattr(edge, 'coexactComponent'):
                print("ERROR: Edges do not have edge.coexactComponent defined. Cannot proceed")
                exit()
            if not isinstance(edge.coexactComponent, float):
                print("ERROR: edge.coexactComponent is defined, but has the wrong type. Type is " + str(type(edge.coexactComponent)) + " when if should be 'float'")
                exit()

            # Check harmonic 
            if not hasattr(edge, 'harmonicComponent'):
                print("ERROR: Edges do not have edge.harmonicComponent defined. Cannot proceed")
                exit()
            if not isinstance(edge.harmonicComponent, float):
                print("ERROR: edge.harmonicComponent is defined, but has the wrong type. Type is " + str(type(edge.harmonicComponent)) + " when if should be 'float'")
                exit()



    # Visualization related
    def covectorToFaceVectorWhitney(mesh, covectorName, vectorName):

        for face in mesh.faces:
            pi = face.anyHalfEdge.vertex.position
            pj = face.anyHalfEdge.next.vertex.position
            pk = face.anyHalfEdge.next.next.vertex.position
            eij = pj - pi
            ejk = pk - pj
            eki = pi - pk
            N = cross(eij, -eki)
            A = 0.5 * norm(N)
            N /= 2*A
            wi = getattr(face.anyHalfEdge.edge, covectorName) * face.anyHalfEdge.orientationSign
            wj = getattr(face.anyHalfEdge.next.edge, covectorName) * face.anyHalfEdge.next.orientationSign
            wk = getattr(face.anyHalfEdge.next.next.edge, covectorName) * face.anyHalfEdge.next.next.orientationSign
            # s = (1.0 / (6.0 * A)) * cross(N, wi*(eki-ejk) + wj*(eij-eki) + wk*(ejk-eij))
            s = (1.0 / (6.0 * A)) * cross(N, wi*(ejk-eij) + wj*(eki-ejk) + wk*(eij-eki))

            setattr(face, vectorName, s) 

    def flat(mesh, vectorFieldName, oneFormName):
        """
        Given a vector field defined on faces, compute the corresponding (integrated) 1-form 
        on edges.
        """

        for edge in mesh.edges:

            oe = edge.orientedHalfEdge

            if not oe.isReal:
                val2 = getattr(edge.orientedHalfEdge.twin.face, vectorFieldName)
                meanVal = val2
            elif not oe.twin.isReal:
                val1 = getattr(edge.orientedHalfEdge.face, vectorFieldName)
                meanVal = val1
            else:
                val1 = getattr(edge.orientedHalfEdge.face, vectorFieldName)
                val2 = getattr(edge.orientedHalfEdge.twin.face, vectorFieldName)
                meanVal = 0.5 * (val1 + val2)
    
            setattr(edge, oneFormName, dot(edge.orientedHalfEdge.vector, meanVal))


    ### Actual main method:

    # get ready
    assignEdgeOrientations(mesh)
    checkOrientationDefined(mesh)

    # Generate a vector field on the surface
    if simpleTest:
        generateFieldSimple(mesh)
    else:
        generateInterestingField(mesh)
    
    flat(mesh, 'vector', 'omega')

    # Apply the decomposition from this assignment
    print("\n=== Decomposing field in to components")
    decomposeField(mesh) 
    print("=== Done decomposing field ===\n\n")

    # Verify everything necessary is defined for the output
    checkResultTypes(mesh)

    # Convert the covectors to face vectors for visualization
    covectorToFaceVectorWhitney(mesh, "exactComponent", "omega_exact_component")
    covectorToFaceVectorWhitney(mesh, "coexactComponent", "omega_coexact_component")
    covectorToFaceVectorWhitney(mesh, "harmonicComponent", "omega_harmonic_component")
    covectorToFaceVectorWhitney(mesh, "omega", "omega_original")


    # Register a vector toggle to switch between the vectors we just defined
    vectorList = [  {'vectorAttr':'omega_original', 'key':'1', 'colormap':'Spectral', 'vectorDefinedAt':'face'},
                    {'vectorAttr':'omega_exact_component', 'key':'2', 'colormap':'Blues', 'vectorDefinedAt':'face'},
                    {'vectorAttr':'omega_coexact_component', 'key':'3', 'colormap':'Reds', 'vectorDefinedAt':'face'},
                    {'vectorAttr':'omega_harmonic_component', 'key':'4', 'colormap':'Greens', 'vectorDefinedAt':'face'}
                 ]
    meshDisplay.registerVectorToggleCallbacks(vectorList)

    # Start the GUI
    meshDisplay.startMainLoop()




if __name__ == "__main__": main()
