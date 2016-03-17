"""
Generate a set of streamlines on the surface of a mesh corresponding to a vector
field defined in the tangent space of that mesh
"""

import random, cmath
from math import cos, sin, pi

from Utilities import *


def generateStreamlines(mesh, nLines, lineLength, vectorFieldName, nSym = 1, countMax=100,
                        definedOn='vertex', isTangentVector=False, includeNormals=False):
    """
    Return a collection of lines that are streamlines for the vector field.
    The returned object is a list, where each element in that list is a list
    of positions

        - vectorFieldName: String giving the name of the vector field attribute to use.
                           Should be vectors in R3 (will be normalized).

        - includeNormals: if True, also returns a list of normal vectors to go with each point in each line
                          each element of the set is now a (line, normals) tuple of lists

    """

    print("\nTracing " + str(nLines) + " streamlines on the surface of the mesh. Please hold...")

    # Bookkeeping
    valueCache = prepareVectorField(mesh, vectorFieldName, definedOn, isTangentVector=isTangentVector, nSym=nSym)

    lines = []

    # Make sure every mesh has at least one line passing through it before we
    # start repeating
    emptyFaces = set(mesh.faces)

    for i in range(nLines):

        # If every face has 1, start puting 2 in every face (etc...)
        if len(emptyFaces) == 0:
            emptyFaces = set(mesh.faces)
        startingFace = random.sample(emptyFaces, 1)[0]

        # For now, the starting point is just the barycenter of the face
        startingPoint = startingFace.center

        # Traces this line
        if includeNormals:
            line, normals, facesUsed = traceStreamline(startingFace, startingPoint, lineLength, valueCache, countMax=countMax, includeNormals=True)
            lines.append((line, normals))

        else:
            line, facesUsed = traceStreamline(startingFace, startingPoint, lineLength, valueCache, countMax=countMax, includeNormals=False)
            lines.append(line)

        # Remove all faces that were used for this line
        emptyFaces -= facesUsed

    print("   ...done tracing streamlines. Please come again.")

    return lines

def prepareVectorField(mesh, vectorFieldName, definedOn, isTangentVector=False, nSym=1):
    """
    Make sure we have a good vector field defined on faces to compute streamlines.

    Post: Each face will have an attribute _streamVec which is the
          unit-norm constant vector field within that face
    """

    if definedOn == 'vertex':
        for face in mesh.faces:
            if isTangentVector:

                # Extend a vector field defined at vertices to the faces
                
                # First, LC-transport all of the vertex fields to the first to get a uniform representation
                firstVert = None
                for vertex in face.adjacentVerts():
                    if firstVert is None:
                        firstVert = vertex
                        centralVal = cmath.exp(1.0j * getattr(vertex, vectorFieldName) * nSym)
                    else:
                        he = vertex.halfedgeTo(firstVert)
                        centralVal += cmath.exp(1.0j * (getattr(vertex, vectorFieldName) + he.transportAngle)*nSym)
                centralAngle = cmath.phase(centralVal**(1.0/nSym))
                meanVec = firstVert.tangentAngleInR3(centralAngle)
                face._streamVec = normalized(face.projectToTangentSpace(meanVec))
            else:
                if nSym > 1:
                    raise ValueError("ERROR: Symmetric vector fields only supported as tangent angles")
                vecs = [normalized(getattr(vert, vectorFieldName)) for vert in face.adjacentVerts()]
                meanVec = sum(vecs)
                face._streamVec = normalized(face.projectToTangentSpace(meanVec))

    elif definedOn == 'face':

        if isTangentVector:
            raise ValueError("ERROR Don't know how to process tangent vectors on faces")

        for face in mesh.faces:
            face._streamVec = normalized(face.projectToTangentSpace(getattr(face, vectorFieldName)))

    else:
        raise ValueError("Illegal definedOn setting: " + str(definedOn))

    
    # Pre-compute some values that we will be using repeatedly
    delTheta = 2.0*pi / nSym
    rotMat = np.array([[cos(delTheta), -sin(delTheta)],[sin(delTheta), cos(delTheta)]])
    valueCache = {}
    for face in mesh.faces:
        
        xDir = face.anyHalfEdge.vector
        yDir = cross(xDir, face.normal)
        v0 = face.anyHalfEdge.vertex.position

        # Generate a vector for each direction in a symmetric field
        uVecFirst = np.array(( dot(face._streamVec, xDir) , dot(face._streamVec, yDir) ))
        uVecThis = uVecFirst
        uVecs = []
        for i in range(nSym):
            # Project in to 3D
            uVec3 = uVecThis[0] * xDir + uVecThis[1] * yDir
            
            # Save
            uVecs.append((uVecThis.copy(), uVec3))
            
            # Rotate for the next direction
            uVecThis = rotMat.dot(uVecThis)


        valueCache[face] = (xDir, yDir, v0, uVecs)
        
        for he in face.adjacentHalfEdges():

            edgePoint3D = he.vertex.position - v0
            edgePoint = np.array(( dot(edgePoint3D, xDir), dot(edgePoint3D, yDir) ))
            edgeVec3D = -he.vector
            edgeVec = np.array(( dot(edgeVec3D, xDir), dot(edgeVec3D, yDir) ))

            valueCache[(face,he)] = (edgePoint, edgeVec)

    return valueCache

def traceStreamline(startingFace, startingPoint, lineLength, valueCache, countMax = 100, includeNormals=False):
    """
    Traces a single streamline through the mesh, returning the line as a list
    of points.
    """

    line = [startingPoint]
    if(includeNormals):
        normals = [startingFace.normal]
    facesUsed = set()


    length = 0.0

    currFace = startingFace
    currPoint = startingPoint
    currV = None

    while (length < lineLength) and (currFace is not None):
        facesUsed.add(currFace)

        # Trace out to the next point
        nextFace, nextPoint = traceStreamlineThroughFace(currFace, currPoint, currV, valueCache)

        # Measure the velocity and length
        currV = nextPoint - currPoint
        length += norm(currV)

        # Save the new point and continue
        line.append(nextPoint)
        if includeNormals:
            if nextFace is None:
                normals.append(currFace.normal)
            else:
                normals.append(nextFace.normal)


        currFace = nextFace
        currPoint = nextPoint

        # Catch infinte loops that might happen for numerical reasons
        if(len(line) > countMax): 
            break

    if includeNormals:
        return line, normals, facesUsed
    else:
        return line, facesUsed

def traceStreamlineThroughFace(startFace, startPoint, currV, valueCache):
    """
    Trace a point through a triangle, returning (newFace, newPoint)
    If the stream goes off a boundary, return (None, newPoint)

        - currV is the current "velocity" of the line, in 3D. Used to choose which
          direction to follow in symmetric fields

    Pre: startPoint is strictly inside startFace
    Post: newPoint is strictly inside newFace (if not boundary)

    Assumes that the vector field is a constant within the triangle,
    """

    ## Raycast to decide which of the faces of the triangle we pass through
    uMin = float('inf')
    xDir, yDir, v0, uVecsR2R3 = valueCache[startFace]
    startPointLocal = startPoint - v0
    startPoint2D = np.array(( dot(startPointLocal, xDir), dot(startPointLocal, yDir) ))


    # For symmetric fields, choose the rotation direction which is closest to 
    # current "velocity" of the streamline
    if currV is None:
        uVec, uVecR3 = random.sample(uVecsR2R3,1)[0]
    else:
        currDot = -float('inf')
        uVec = None
        for uR2,uR3 in uVecsR2R3:
            if dot(uR3, currV) > currDot:
                currDot = dot(uR3, currV)
                uVec = uR2
                uVecR3 = uR3

    for he in startFace.adjacentHalfEdges():

        edgePoint, edgeVec = valueCache[(startFace,he)]

        # Line/plane intersection
        u = cross2D(startPoint2D - edgePoint, edgeVec) / cross2D(edgeVec, uVec)

        # Check if this is the closest
        if u > 0 and u < uMin:
            uMin = u
            acrossHe = he
            t = cross2D(startPoint2D - edgePoint, uVec) / cross2D(edgeVec, uVec)
            t = clamp(t, 0.005, 0.995)

    # TODO sometimes things can go wrong from numerical errors... just give up if that happens
    if uMin == float('inf'):
        return None, startPoint

    # Compute the new point. Extend the vector just a little so the next numerical problem is well-posed
    # TODO this could be a bug for exceptionally skinny triangles
    newPoint = acrossHe.vertex.position - t * acrossHe.vector + (uMin * 0.00001) * uVecR3

    if acrossHe.isBoundary:
        return None, newPoint
    else:
        newFace = acrossHe.twin.face
        return newFace, newPoint

def cross2D(v1, v2):
    return v1[0]*v2[1] - v1[1]*v2[0]

def constainToFace(point, face):
    """
    Given a point which is supposed to lie inside a
    """
    pass
