import numpy as np

from math import acos, pi

from TriSoupMesh import TriSoupMesh
from Utilities import *

# Use euclid for rotations
import euclid as eu


# Iniialize the global counters for id numbers
# NOTE: Global ID numbers mainly exist so that human debugging is easier
# and doesn't require looking at memory addresses. If you want to use them for
# program logic, you're probably wrong.
NEXT_HALFEDGE_ID = 0
NEXT_VERTEX_ID = 0
NEXT_FACE_ID = 0
NEXT_EDGE_ID = 0

# A mesh composed of halfedge elements.
# This class follows a "fast and loose" python design philosophy. Data is stored
# on halfedges/vertices/edges/etc as it's created.
#   - staticGeometry=True means that the structue and positions of this mesh
#     will not be changed after creation. This means that geometry-derived values
#     will be cached internally for performance improvements.
class HalfEdgeMesh(object):

    ### Construct a halfedge mesh from a TriSoupMesh
    def __init__(self, soupMesh, readPosition=True, checkMesh=False, staticGeometry=True):

        ### Members

        # Sets of all of the non-fake objects. Note that these are somewhat sneakily named,
        # the do not include the imaginary halfedges/faces used to close boundaries. Those
        # are only tracked in the 'full' sets below, as the user generally shouldn't mess with
        # them
        self.halfEdges = set()
        self.verts = set()
        self.faces = set()
        self.edges = set()

        # These versions include imaginary objects.
        # TODO these names are lame
        self.halfEdgesFull = set()
        self.facesFull = set()

        # This is operated on by the getters and setters below
        self._staticGeometry = staticGeometry

        print('\nConstructing HalfEdge Mesh...')

        # TODO typecheck to ensure the input is a soupMesh?

        # NOTE assumes faces have proper winding, may fail in bad ways otherwise.
        # TODO Detect bad things (LOTS of ways this could be broken)
        # TODO Recover from bad things


        # There are 3 steps for this process
        #   - Iterate through vertices, create vertex objects
        #   - Iterate through faces, creating face, edge, and halfedge objects
        #     (and connecting where possible)
        #   - Iterate through edges, connecting halfedge.twin

        # Find which vertices are actually used in the mesh
        usedVertInds = set()
        for f in soupMesh.tris:
            usedVertInds.add(f[0])
            usedVertInds.add(f[1])
            usedVertInds.add(f[2])
        nUnused = len(set([i for i in range(len(soupMesh.verts))]) - usedVertInds)
        if nUnused > 0:
            print('  Note: ' + str(nUnused) + ' vertices in the original mesh were not used in any face and are being discarded')

        # Create vertex objects for only the used verts
        verts = []
        for (i, soupVert) in enumerate(soupMesh.verts):
            if i in usedVertInds:
                if readPosition:
                    v = Vertex(soupVert, staticGeometry=staticGeometry)
                else:
                    v = Vertex(staticGeometry=staticGeometry)
                verts.append(v)
                self.verts.add(v)
            else:
                verts.append(None)

        # Iterate over the faces, creating a new face and new edges/halfedges
        # for each. Fill out all properties except the twin & next references
        # for the halfedge, which will be handled below.
        edgeDict = {}           # The edge that connects two verts  [(ind1, ind2) ==> edge]
        edgeHalfEdgeDict = {}   # The two halfedges that border an edge [(ind1, ind2) ==> [halfedge list]]
        edgeSet = set()         # All edges that appear in the mesh, used for a sanity check
        for soupFace in soupMesh.tris:

            face = Face(staticGeometry=staticGeometry)
            self.faces.add(face)
            self.facesFull.add(face)

            theseHalfEdges = []     # The halfedges that make up this face

            # Iterate over the edges that make up the face
            for i in range(3):

                ind1 = soupFace[i]
                ind2 = soupFace[(i+1)%3]
                edgeKey = tuple(sorted((ind1,ind2)))

                # Sanity check that there are no duplicate edges in the input mesh
                if((ind1, ind2) in edgeSet):
                    raise ValueError('Mesh has duplicate edges or inconsistent winding, cannot represent as a half-edge mesh')
                else:
                    edgeSet.add((ind1, ind2))

                # Get an edge object, creating a new one if needed
                if(edgeKey in edgeDict):
                    edge = edgeDict[edgeKey]
                else:
                    edge = Edge(staticGeometry=staticGeometry)
                    edgeDict[edgeKey] = edge
                    self.edges.add(edge)
                    edgeHalfEdgeDict[edgeKey] = []

                # Create a new halfedge, which is always needed
                h = HalfEdge(staticGeometry=staticGeometry)
                self.halfEdges.add(h)
                self.halfEdgesFull.add(h)
                theseHalfEdges.append(h)

                # Set references to the halfedge in the other structures
                # This might be overwriting a previous value, but that's fine
                face.anyHalfEdge = h
                edge.anyHalfEdge = h
                verts[ind1].anyHalfEdge = h

                edgeHalfEdgeDict[edgeKey].append(h)

                # Set references to the other structures in the halfedge
                h.vertex = verts[ind2]
                h.edge = edge
                h.face = face


            # Connect the halfEdge.next reference for each of the halfedges we just created
            # in this face
            for i in range(3):
                theseHalfEdges[i].next = theseHalfEdges[(i+1)%3]

        # Sanity check on the edges we say
        unpairedEdges = 0
        unpairedVerts = set()
        for (v1, v2) in edgeSet:
            if (v2, v1) not in edgeSet:
                unpairedEdges += 1
                unpairedVerts.add(v1)
                unpairedVerts.add(v2)
        print('  Input mesh has ' + str(unpairedEdges) + ' unpaired edges (which only appear in one direction)')
        print('  Input mesh has ' + str(len(unpairedVerts)) + ' unpaired verts (which touch some unpaired edge)')



        # Iterate through the edges to fill out the twin reference for each halfedge
        # This is where we use edgeHalfEdgeDict.
        for (edgeKey, halfEdgeList) in edgeHalfEdgeDict.iteritems():

            # Assuming the mesh is well-formed, this must be a list with two elements
            if(len(halfEdgeList) == 2):
                halfEdgeList[0].twin = halfEdgeList[1]
                halfEdgeList[1].twin = halfEdgeList[0]
            elif(len(halfEdgeList) > 2):
                raise ValueError('Mesh has more than two faces meeting at some edge')

        # Close boundaries by iterating around each hole and creating an imaginary face to cover the hole,
        # along with the associated halfedges. Note that this face will not be a triangle, in general.
        initialHalfEdges = self.halfEdges.copy()
        nHolesFilled = 0
        for initialHE in initialHalfEdges:

            # If this halfedge has no twin, then we have found a new boundary hole. Traverse the outside
            # and create a new faces/new halfedges
            # Note: strange things will happen if the multiples holes touch a single vertex.
            if initialHE.twin is None:
                nHolesFilled += 1

                fakeFace = Face(isReal=False, staticGeometry=staticGeometry)
                self.facesFull.add(fakeFace)

                # Traverse around the outside of the hole
                currRealHE = initialHE
                prevNewHE = None
                while True:

                    # Create a new fake halfedge
                    currNewHE = HalfEdge(isReal=False, staticGeometry=staticGeometry)
                    self.halfEdgesFull.add(currNewHE)
                    currNewHE.twin = currRealHE
                    currRealHE.twin = currNewHE
                    currNewHE.face = fakeFace
                    currNewHE.vertex = currRealHE.next.next.vertex
                    currNewHE.edge = currRealHE.edge
                    currNewHE.next = prevNewHE

                    # Advance to the next border vertex along the loop
                    currRealHE = currRealHE.next
                    while currRealHE != initialHE and currRealHE.twin != None:
                        currRealHE = currRealHE.twin.next

                    prevNewHE = currNewHE

                    # Terminate when we have walked all the way around the loop
                    if currRealHE == initialHE:
                        break


                # Arbitrary point the fakeFace at the last created halfedge
                fakeFace.anyHalfEdge = currNewHE

                # Connect the next ref for the first face edge, which was missed in the above loop
                initialHE.twin.next = prevNewHE

        print('  Filled %d boundary holes in mesh using imaginary halfedges/faces'%(nHolesFilled))


        print("HalfEdge mesh construction completed")

        # Print out statistics about the mesh and check it
        self.printMeshStats(printImaginary=True)
        if checkMesh:
            self.checkMeshReferences()
        #self.checkDegenerateFaces() # a lot of meshes fail this...

    # Perform a basic refence validity check to catch blatant errors
    # Throws and error if it finds something broken about the datastructure
    def checkMeshReferences(self):

        # TODO why does this use AssertionError() instead of just assert statements?
        print('Testing mesh for obvious problems...')

        # Make sure the 'full' sets are a subset of their non-full counterparts
        diff = self.halfEdges - self.halfEdgesFull
        if(diff):
            raise AssertionError('ERROR: Mesh check failed. halfEdges is not a subset of halfEdgesFull')
        diff = self.faces - self.facesFull
        if(diff):
            raise AssertionError('ERROR: Mesh check failed. faces is not a subset of facesFull')

        # Accumulators for things that were referenced somewhere
        allRefHalfEdges = set()
        allRefEdges = set()
        allRefFaces = set()
        allRefVerts= set()

        ## Verify that every object in our sets is referenced by some halfedge, and vice versa
        # Accumulate sets of anything referenced anywhere and ensure no references are None
        for he in self.halfEdgesFull:

            if not he.next:
                raise AssertionError('ERROR: Mesh check failed. he.next is None')
            if not he.twin:
                raise AssertionError('ERROR: Mesh check failed. he.twin is None')
            if not he.edge:
                raise AssertionError('ERROR: Mesh check failed. he.edge is None')
            if not he.face:
                raise AssertionError('ERROR: Mesh check failed. he.face is None')
            if not he.vertex:
                raise AssertionError('ERROR: Mesh check failed. he.vertex is None')

            allRefHalfEdges.add(he.next)
            allRefHalfEdges.add(he.twin)
            allRefEdges.add(he.edge)
            allRefFaces.add(he.face)
            allRefVerts.add(he.vertex)

            if he.twin.twin != he:
                raise AssertionError('ERROR: Mesh check failed. he.twin symmetry broken')

        for edge in self.edges:

            if not edge.anyHalfEdge:
                raise AssertionError('ERROR: Mesh check failed. edge.anyHalfEdge is None')
            allRefHalfEdges.add(edge.anyHalfEdge)

        for vert in self.verts:

            if not vert.anyHalfEdge:
                raise AssertionError('ERROR: Mesh check failed. vert.anyHalfEdge is None')
            allRefHalfEdges.add(vert.anyHalfEdge)

        for face in self.facesFull:

            if not face.anyHalfEdge:
                raise AssertionError('ERROR: Mesh check failed. face.anyHalfEdge is None')
            allRefHalfEdges.add(face.anyHalfEdge)

        # Check the resulting sets for equality
        if allRefHalfEdges != self.halfEdgesFull:
            raise AssertionError('ERROR: Mesh check failed. Referenced halfedges do not match halfedge set')
        if allRefEdges != self.edges:
            raise AssertionError('ERROR: Mesh check failed. Referenced edges do not match edges set')
        if allRefFaces != self.facesFull:
            raise AssertionError('ERROR: Mesh check failed. Referenced faces do not match faces set')
        if allRefVerts != self.verts:
            raise AssertionError('ERROR: Mesh check failed. Referenced verts do not match verts set')

        print('  ...test passed')


    def checkDegenerateFaces(self):
        """
        Checks if the mesh has any degenerate faces, which can mess up many algorithms.
        This is an exact-comparison check, so it won't catch vertices that differ by epsilon.
        """
        print("Checking mesh for degenerate faces...")

        for face in self.faces:

            seenPos = set()
            vList = []
            for v in face.adjacentVerts():
                pos = tuple(v.position.tolist()) # need it as a hashable type
                if pos in seenPos:
                    raise ValueError("ERROR: Degenerate mesh face has repeated vertices at position: " + str(pos))
                else:
                    seenPos.add(pos)
                vList.append(v.pos)

            # Check for triangular faces with colinear vertices (don't catch other such errors for now)
            if(len(vList) == 3):
                v1 = vList[1] - vList[0]
                v2 = vList[2]-vList[0]
                area = norm(cross(v1, v2))
                if area < 0.0000000001*max((norm(v1),norm(v2))):
                    raise ValueError("ERROR: Degenerate mesh face has triangle composed of 3 colinear points: \
                        " + str(vList))


        print("  ...test passed")

    # Print out some summary statistics about the mesh
    def printMeshStats(self, printImaginary=False):

        if printImaginary:
            print('=== HalfEdge mesh statistics:')
            print('    Halfedges = %d  (+ %d imaginary)'%(len(self.halfEdges), (len(self.halfEdgesFull) - len(self.halfEdges))))
            print('    Edges = %d'%(len(self.edges)))
            print('    Faces = %d  (+ %d imaginary)'%(len(self.faces), (len(self.facesFull) - len(self.faces))))
            print('    Verts = %d'%(len(self.verts)))
        else:
            print('=== HalfEdge mesh statistics:')
            print('    Halfedges = %d'%(len(self.halfEdges)))
            print('    Edges = %d'%(len(self.edges)))
            print('    Faces = %d'%(len(self.faces)))
            print('    Verts = %d'%(len(self.verts)))


        maxDegree = max([v.degree for v in self.verts])
        minDegree = min([v.degree for v in self.verts])
        print('    - Max vertex degree = ' + str(maxDegree))
        print('    - Min vertex degree = ' + str(minDegree))

        nBoundaryVerts = sum([v.isBoundary for v in self.verts])
        print('    - n boundary verts = ' + str(nBoundaryVerts))


    # If this mesh has boundaries, close them (make the imaginary faces/halfedges real).
    # The resulting mesh will be manifold.
    # Note: This naively triangulates non-triangular boundary faces, and thus can create
    # low quality meshes
    # TODO implement
    def fillBoundaries(self):
        raise NotImplementedError('fillBoundaries is not yet implemented')


    # Need to clear the caches whenever staticGeometry is made False
    # TODO this all could probably use a bit of testing
    @property
    def staticGeometry(self):
        return self._staticGeometry
    @staticGeometry.setter
    def staticGeometry(self, staticGeometry):

        # Clear the caches (only needed when going from True-->False)
        if staticGeometry == False:
            for v in self.verts: v._cache.clear()
            for e in self.edges: e._cache.clear()
            for f in self.facesFull: f._cache.clear()
            for he in self.halfEdgesFull: he._cache.clear()

        # Update the static settings
        for v in self.verts:
            v.staticGeometry = staticGeometry
            v._pos.flags.writeable = not staticGeometry
        for e in self.edges: e.staticGeometry = staticGeometry
        for f in self.facesFull: f.staticGeometry = staticGeometry
        for he in self.halfEdgesFull: he.staticGeometry = staticGeometry

        self._staticGeometry = staticGeometry


    def enumerateVertices(self, subset=None):
        """
        Assign a unique index from 0 to (N-1) to each vertex in the mesh. Will
        return a dictionary containing mappings {vertex ==> index}.
        """
        if subset is None:
            subset = self.verts

        enum = dict()
        ind = 0
        for vert in subset:
            if vert not in self.verts:
                raise ValueError("ERROR: enumerateVertices(subset) was called with a vertex in subset which is not in the mesh.")

            enum[vert] = ind
            ind += 1

        return enum
    
    
    def enumerateEdges(self):
        """
        Assign a unique index from 0 to (N-1) to each edge in the mesh.
        """
        d = dict()
        for (i, edge) in enumerate(self.edges):
            d[edge] = i
        return d
    
    
    def enumerateFaces(self):
        """
        Assign a unique index from 0 to (N-1) to each face in the mesh.
        """
        d = dict()
        for (i, face) in enumerate(self.faces):
            d[face] = i
        return d


    def assignReferenceDirections(self):
        '''
        For each vertex in the mesh, arbitrarily selects one outgoing halfedge
        as a reference ('referenceEdge').
        '''
        for vert in self.verts:
            vert.referenceEdge = vert.anyHalfEdge


    def applyVertexValue(self, value, attributeName):
        """
        Given a dictionary of {vertex => value}, stores that value on each vertex
        with attributeName
        """

        # Throw an error if there isn't a value for every vertex
        if not set(value.keys()) == self.verts:
            raise ValueError("ERROR: Attempted to apply vertex values from a map whos domain is not the vertex set")

        for v in self.verts:
            setattr(v, attributeName, value[v])

    # Returns a brand new TriSoupMesh corresponding to this mesh
    # 'retainVertAttr' is a list of vertex-valued attributes to carry in to th trisoupmesh
    # TODO do face attributes (and maybe edge?)
    # TODO Maybe implement a 'view' version of this, so that we can modify the HalfEdge mesh
    # without completely recreating a new TriSoup mesh.
    def toTriSoupmesh(self,retainVertAttr=[]):

        # Create a dictionary for the vertex attributes we will retain
        vertAttr = dict()
        for attr in retainVertAttr:
            vertAttr[attr] = []

        # Iterate over the vertices, numbering them and building an array
        vertArr = []
        vertInd = {}
        for (ind, v) in enumerate(self.verts):
            vertArr.append(v.position)
            vertInd[v] = ind

            # Add any vertex attributes to the list
            for attr in retainVertAttr:
                vertAttr[attr].append(getattr(v, attr))

        # Iterate over the faces, building a list of the verts for each
        faces = []
        for face in self.faces:

            # Get the three edges which make up this face
            he1 = face.anyHalfEdge
            he2 = he1.next
            he3 = he2.next

            # Get the three vertices that make up the face
            v1 = vertInd[he1.vertex]
            v2 = vertInd[he2.vertex]
            v3 = vertInd[he3.vertex]

            faceInd = [v1, v2, v3]
            faces.append(faceInd)


        soupMesh = TriSoupMesh(vertArr, faces, vertAttr=vertAttr)

        return soupMesh

### A decorator which automatically caches static geometric values
#
# The invariant for how staticGeometry works is:
#   - The staticGeometry flag should only be changed by setting/unsetting it
#     in the mesh which holds this object (to ensure it gets set/unset everywhere,
#     since the result of cached computation may be stored in a distant object)
#   - If there is anything in the cache, it must be valid to return it. Thus,
#     the cache must be emptied when staticGeometry is set to False.
#
# It would be nice to automatically empty the cache whenever a vertex position is
# changed and forget about the flag. However, this would require recusively updating
# all of the caches which depend on that value. Possible, but a little complex and
# maybe slow.
def cacheGeometry(f):
    name = f.__name__
    def cachedF(self=None):
        if name in self._cache: return self._cache[name]
        res = f(self)
        if self.staticGeometry: self._cache[name] = res
        return res
    return cachedF

class HalfEdge(object):

    ### Construct a halfedge, possibly not real
    def __init__(self, isReal=True, staticGeometry=False):
        self.isReal = isReal  # Is this a real halfedge, or an imaginary one we created to close a boundary?

        ### Members
        self.twin = None
        self.next = None
        self.vertex = None
        self.edge = None
        self.face = None

        self._cache = dict()
        self.staticGeometry = staticGeometry

        # Global id number, mainly for debugging
        global NEXT_HALFEDGE_ID
        self.id = NEXT_HALFEDGE_ID
        NEXT_HALFEDGE_ID += 1

    ## Slightly prettier print function
    # TODO Maybe make repr() more verbose?
    def __str__(self):
        return "<HalfEdge #{}>".format(self.id)
    def __repr__(self):
        return self.__str__()

    # Return a boolean indicating whether this is on the boundary of the mesh
    @property
    def isBoundary(self):
        return not self.twin.isReal

    @property
    @cacheGeometry
    def vector(self):
        """The vector represented by this halfedge"""
        v = self.vertex.position - self.twin.vertex.position
        return v

    @property
    @cacheGeometry
    def cotan(self):
        """
        Return the cotangent of the opposite angle, or 0 if this is an imaginary
        halfedge
        """
        # Validate that this is on a triangle
        if self.next.next.next is not self:
            raise ValueError("ERROR: halfedge.cotan() is only well-defined on a triangle")

        if self.isReal:

            # Relevant vectors
            A = -self.next.vector
            B = self.next.next.vector

            # Nifty vector equivalent of cot(theta)
            val = np.dot(A,B) / norm(cross(A,B))
            return val

        else:
            return 0.0


class Vertex(object):

    ### Construct a vertex, possibly with a known position
    def __init__(self, pos=None, staticGeometry=False):

        if pos is not None:
            self._pos = pos
            if staticGeometry:
                self._pos.flags.writeable = False

        self.anyHalfEdge = None      # Any halfedge exiting this vertex

        self._cache = dict()
        self.staticGeometry = staticGeometry

        # Global id number, mainly for debugging
        global NEXT_VERTEX_ID
        self.id = NEXT_VERTEX_ID
        NEXT_VERTEX_ID += 1

    ## Slightly prettier print function
    # TODO Maybe make repr() more verbose?
    def __str__(self):
        return "<Vertex #{}>".format(self.id)
    def __repr__(self):
        return self.__str__()


    @property
    def position(self):
        return self._pos

    @position.setter
    def position(self, value):
        if self.staticGeometry:
            raise ValueError("ERROR: Cannot write to vertex position with staticGeometry=True. To allow dynamic geometry, set staticGeometry=False when creating vertex (or in the parent mesh constructor)")
        self._pos = value


    # Iterate over the faces adjacent to this vertex (skips imaginary faces by default)
    def adjacentFaces(self, skipImaginary=True):

        # Iterate through the adjacent faces
        first = self.anyHalfEdge
        curr = self.anyHalfEdge
        while True:

            # Yield only real faces
            if curr.isReal or not skipImaginary:
                yield curr.face

            curr = curr.twin.next
            if(curr == first):
                break


    # Iterate over the edges adjacent to this vertex
    def adjacentEdges(self):

        # Iterate through the adjacent edges
        first = self.anyHalfEdge
        curr = self.anyHalfEdge
        while True:

            yield curr.edge

            curr = curr.twin.next
            if(curr == first):
                break


    # Iterate over the halfedges adjacent to this vertex
    def adjacentHalfEdges(self):

        # Iterate through the adjacent edges
        first = self.anyHalfEdge
        curr = self.anyHalfEdge
        while True:

            yield curr

            curr = curr.twin.next
            if(curr == first):
                break

    # Iterate over the verts adjacent to this vertex
    def adjacentVerts(self):

        # Iterate through the adjacent edges
        first = self.anyHalfEdge
        curr = self.anyHalfEdge
        while True:

            yield curr.vertex

            curr = curr.twin.next
            if(curr == first):
                break

    def adjacentEdgeVertexPairs(self):
        """
        Iterate through the neighbors of this vertex, yielding a (edge,vert) tuple
        """
        first = self.anyHalfEdge
        curr = self.anyHalfEdge
        while True:

            yield (curr.edge, curr.vertex)

            curr = curr.twin.next
            if(curr == first):
                break


    # Return a boolean indicating whether this is on the boundary of the mesh
    @property
    def isBoundary(self):

        # Traverse the halfedges adjacent to this, a loop of non-boundary halfedges
        # indicates that this vert is internal
        first = self.anyHalfEdge
        curr = self.anyHalfEdge
        while True:
            if curr.isBoundary:
                return True

            curr = curr.twin.next
            if(curr == first):
                break

        return False

    # Returns the number edges/faces neighboring this vertex
    @property
    @cacheGeometry
    def degree(self):

        d = sum(1 for e in self.adjacentEdges())
        return d



    @property
    @cacheGeometry
    def normal(self):
        """The area-weighted normal vector for this vertex"""

        normalSum = np.array([0.0,0.0,0.0])
        for face in self.adjacentFaces():
            normalSum += face.normal * face.area
        n = normalize(normalSum)

        return n

    def projectToTangentSpace(self, vec):
        """
        Projects a vector in R3 to a new vector in R3, guaranteed
        to lie in the tangent plane of this vertex
        """
        return vec - self.normal * (dot(vec, self.normal))


class Face(object):


    ### Construct a face, possibly not real
    def __init__(self, isReal=True, staticGeometry=False):

        ### Members
        self.anyHalfEdge = None      # Any halfedge bordering this face
        self.isReal = isReal         # Is this an actual face of the mesh, or an artificial face we
                                     # created to close boundaries?

        self._cache = dict()
        self.staticGeometry = staticGeometry

        # Global id number, mainly for debugging
        global NEXT_FACE_ID
        self.id = NEXT_FACE_ID
        NEXT_FACE_ID += 1

    ## Slightly prettier print function
    # TODO Maybe make repr() more verbose?
    def __str__(self):
        return "<Face #{}>".format(self.id)
    def __repr__(self):
        return self.__str__()

    # Return a boolean indicating whether this is on the boundary of the mesh
    @property
    def isBoundary(self):

        first = self.anyHalfEdge
        curr = self.anyHalfEdge
        while True:
            if curr.isBoundary:
                return True

            curr = curr.next
            if(curr == first):
                break

        return False


    # Iterate over the verts that make up this face
    def adjacentVerts(self):

        # Iterate through the adjacent faces
        first = self.anyHalfEdge
        curr = self.anyHalfEdge
        while True:

            # Yield a vertex
            yield curr.vertex

            curr = curr.next
            if(curr == first):
                break

    # Iterate over the halfedges that make up this face
    def adjacentHalfEdges(self):

        # Iterate through the adjacent faces
        first = self.anyHalfEdge
        curr = self.anyHalfEdge
        while True:

            # Yield a halfedge
            yield curr

            curr = curr.next
            if(curr == first):
                break

    # Iterate over the edges that make up this face
    def adjacentEdges(self):

        # Iterate through the adjacent faces
        first = self.anyHalfEdge
        curr = self.anyHalfEdge
        while True:

            # Yield an edge
            yield curr.edge

            curr = curr.next
            if(curr == first):
                break

    @property
    @cacheGeometry
    def normal(self):
        """The normal vector for this face"""

        v = list(self.adjacentVerts())
        n = normalize(cross(v[1].position - v[0].position, v[2].position - v[0].position))

        return n


    @property
    @cacheGeometry
    def area(self):
        """The area of this face"""

        v = list(self.adjacentVerts())
        a = 0.5 * norm(cross(v[1].position - v[0].position, v[2].position - v[0].position))

        return a

    @property
    @cacheGeometry
    def center(self):
        """The center of this face"""

        v1 = self.anyHalfEdge.vertex.position
        v2 = self.anyHalfEdge.next.vertex.position
        v3 = self.anyHalfEdge.next.next.vertex.position

        c = (v1 + v2 + v3) / 3.0

        return c

    def projectToTangentSpace(self, vec):
        """
        Projects a vector in R3 to a new vector in R3, guaranteed
        to lie in the tangent plane of this face
        """
        return vec - self.normal * (dot(vec, self.normal))

class Edge(object):


    def __init__(self, staticGeometry=False):
        ### Members
        anyHalfEdge = None      # Either of the halfedges (if this is a boundary edge,
                                # guaranteed to be the real one)

        self._cache = dict()
        self.staticGeometry = staticGeometry

        # Global id number, mainly for debugging
        global NEXT_EDGE_ID
        self.id = NEXT_EDGE_ID
        NEXT_EDGE_ID += 1

    ## Slightly prettier print function
    # TODO Maybe make repr() more verbose?
    def __str__(self):
        return "<Edge #{}>".format(self.id)
    def __repr__(self):
        return self.__str__()

    # Return a boolean indicating whether this is on the boundary of the mesh
    @property
    def isBoundary(self):
        return self.anyHalfEdge.isBoundary

    # Return true if the edge can be flipped, meaning:
    #   - The edge is not on the boundary
    #   - Neither of the verts that neighbor the edge has degree <= 3
    def canFlip(self):

        if self.isBoundary:
            return False

        # Can only flip if both vertices have degree > 3
        v1 = self.anyHalfEdge.vertex
        v2 = self.anyHalfEdge.twin.vertex

        return v1.degree() > 3 and v2.degree() > 3


    # Flip this edge
    # Does nothing if canFlip() returns false
    def flip(self):

        if self.staticGeometry:
            raise ValueError("ERROR: Cannot flip edge with static geometry")

        if not self.canFlip():
            return

        # Note: This does a complete reassignment of references, which will likely include
        # changing things that don't technically need to be changed (like anyHalfEdge refs
        # that weren't actually invalidated). This is done for simplicity and conciseness.


        # Get references to the relevant objects
        h1 = self.anyHalfEdge
        h2 = h1.twin
        h1n = h1.next
        h1nn = h1n.next
        h2n = h2.next
        h2nn = h2n.next
        e = h1.edge
        f1 = h1.face
        f2 = h2.face
        Vold1 = h1.vertex
        Vold2 = h2.vertex
        Vnew1 = h1.next.vertex
        Vnew2 = h2.next.vertex


        ## Re-assign pointers
        # This edge
        self.anyHalfEdge = h1
        h1.vertex = Vnew2
        h2.vertex = Vnew1

        # Lower face HE loop
        h1.next = h2nn
        h2nn.next = h1n
        h1n.next = h1

        # Upper face HE loop
        h2.next = h1nn
        h1nn.next = h2n
        h2n.next = h2

        # Faces
        f1.anyHalfEdge = h1
        f2.anyHalfEdge = h2
        h2nn.face = f1
        h1nn.face = f2

        # Verts
        Vold1.anyHalfEdge = h1n
        Vold2.anyHalfEdge = h2n
    
    
    @property
    @cacheGeometry
    def cotanWeight(self):
        """
        Return the cotangent weight for an edge.
        """
        val = 0.0
        if self.anyHalfEdge.isReal:
            val += self.anyHalfEdge.cotan
        if self.anyHalfEdge.twin.isReal:
            val += self.anyHalfEdge.twin.cotan
        val *= 0.5
        return val


