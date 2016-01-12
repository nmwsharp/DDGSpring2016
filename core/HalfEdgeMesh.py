import numpy as np

from math import acos, pi

from TriSoupMesh import TriSoupMesh
from Utilities import *

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

            # print(edgeKey)
            # print(halfEdgeList)

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
        self.checkDegenerateFaces() # a lot of meshes fail this...

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
                pos = tuple(v.pos.tolist()) # need it as a hashable type
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

        nBoundaryVerts = sum([v.isBoundary() for v in self.verts])
        print('    - n boundary verts = ' + str(nBoundaryVerts))


    def enumerateVertices(self, subset=None):
        """
        Return a dictionary which assigns a 0-indexed integer to each vertex
        in the mesh. If 'subset' is given (should be a set), only the vertices
        in subset are indexed.
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


    def assignReferenceDirections(self):
        '''
        For each vertex in the mesh, arbitrarily selects one outgoing halfedge
        as a reference ('refEdge').
        '''
        for vert in self.verts:
            vert.refEdge = vert.anyHalfEdge


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
    # NOTE Always saves normals.
    # TODO do face attributes (and maybe edge?)
    # TODO Maybe implement a 'view' version of this, so that we can modify the HalfEdge mesh
    # without completely recreating a new TriSoup mesh.
    def toTriSoupmesh(self,retainVertAttr=[]):

        # Create a dictionary for the vertex attributes we will retain
        vertAttr = dict()
        for attr in retainVertAttr:
            vertAttr[attr] = []
        vertAttr['normal'] = []

        # Iterate over the vertices, numbering them and building an array
        vertArr = []
        vertInd = {}
        for (ind, v) in enumerate(self.verts):
            vertArr.append(v.pos)
            vertInd[v] = ind

            # Add any vertex attributes to the list
            for attr in retainVertAttr:
                vertAttr[attr].append(getattr(v, attr))

            # Always save normals
            vertAttr['normal'].append(v.normal)

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


    # Return a boolean indicating whether this is on the boundary of the mesh
    def isBoundary(self):
        return not self.twin.isReal

    @property
    def vector(self):
        """The vector represented by this halfedge"""
        if 'vector' in self._cache: return self._cache['vector']
        v = self.vertex.pos - self.twin.vertex.pos
        if self.staticGeometry: self._cache['vector'] = v
        return v


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

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
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

    def adjacentHalfEdgeVertexPairs(self):
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
    def isBoundary(self):

        # Traverse the halfedges adjacent to this, a loop of non-boundary halfedges
        # indicates that this vert is internal
        first = self.anyHalfEdge
        curr = self.anyHalfEdge
        while True:
            if curr.isBoundary():
                return True

            curr = curr.twin.next
            if(curr == first):
                break

        return False

    # Returns the number edges/faces neighboring this vertex
    @property
    def degree(self):
        if 'degree' in self._cache: return self._cache['degree']

        d = sum(1 for e in self.adjacentEdges())

        if self.staticGeometry: self._cache['degree'] = d
        return d


    @property
    def normal(self):
        """The area-weighted normal vector for this face"""
        if 'normal' in self._cache: return self._cache['normal']

        # Implement me please!
        n = 0.0

        if self.staticGeometry: self._cache['normal'] = n
        return n

class Face(object):


    ### Construct a face, possibly not real
    def __init__(self, isReal=True, staticGeometry=False):

        ### Members
        self.anyHalfEdge = None      # Any halfedge bordering this face
        self.isReal = isReal         # Is this an actual face of the mesh, or an artificial face we
                                     # created to close boundaries?

        self._cache = dict()
        self.staticGeometry = staticGeometry

    # Return a boolean indicating whether this is on the boundary of the mesh
    def isBoundary(self):

        first = self.anyHalfEdge
        curr = self.anyHalfEdge
        while True:
            if curr.isBoundary():
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
    def normal(self):
        """The normal vector for this face"""
        if 'normal' in self._cache: return self._cache['normal']

        # Implement me please!
        n = np.array([0.0,0.0,0.0])

        if self.staticGeometry: self._cache['normal'] = n
        return n


    @property
    def area(self):
        """The (signed) area of this face"""
        if 'area' in self._cache: return self._cache['area']

        # Implement me please!
        a = np.array([0.0,0.0,0.0])

        if self.staticGeometry: self._cache['area'] = a
        return a

class Edge(object):


    def __init__(self, staticGeometry=False):
        ### Members
        anyHalfEdge = None      # Either of the halfedges (if this is a boundary edge,
                                # guaranteed to be the real one)

        self._cache = dict()
        self.staticGeometry = staticGeometry


    # Return a boolean indicating whether this is on the boundary of the mesh
    def isBoundary(self):
        return self.anyHalfEdge.isBoundary()
