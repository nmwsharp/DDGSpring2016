# Visualize a mesh using pyopengl (wrapping openGL)

# System imports
import os
from math import *
import sys

# Library imports
import numpy as np
import matplotlib.cm # used for colormaps
import euclid as eu
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# Local imports
from Utilities import normalize
from HalfEdgeMesh import HalfEdgeMesh
from TriSoupMesh import TriSoupMesh
from Utilities import normalize, norm, cross
from Camera import Camera



# Dictionary of known vertex shaders
vertShaders = {
    'plain' : "shaders/plain.vert",
    'shiny' : "shaders/shiny.vert",
    'edges' : "shaders/edges.vert",
    'surf-draw': "shaders/surf-draw.vert",
}

# Dictionary of known fragment shaders
fragShaders = {
    'plain' : "shaders/plain.frag",
    'shiny' : "shaders/shiny.frag",
    'edges' : "shaders/edges.frag",
    'surf-draw': "shaders/surf-draw.frag",
}


def readFile(relPath):
    """Reads from a file with a path relative to this script location"""

    # Get the path to the shader directory.
    # TODO this is supposedly not the right way to do paths... oh well
    myLoc = os.path.realpath(__file__)
    fullPath = myLoc[:myLoc.rfind(os.sep)] + os.sep + relPath

    s = ""
    for line in open(fullPath).readlines():
        s += line

    return s

class MeshDisplay(object):

    # Useful colors
    colors = {
        'dark_grey':(0.15, 0.15, 0.15),
        'grey':(0.5, 0.5, 0.5),
        'black':(0.0,0.0,0.0),
        'almost_white':(0.95, 0.95, 0.95),
        'white': (1.0, 1.0, 1.0),
        'light_blue': (0.2,0.6,1.0),
        'red': (0.72, 0.0, 0.0),
        'orange': (1.0, 0.45, 0.0)
    }


    def __init__(self, windowTitle='MeshDisplay', mesh=None,
        width=1200, height=800, perfTarget = 'nicest'):

        print("Creating MeshDisplay window")

        ### Members
        # Visual options
        self.shapeVertShader = 'shiny'
        self.shapeFragShader = 'shiny'
        self.edgeVertShader = 'edges'
        self.edgeFragShader = 'edges'
        self.shapeAlpha = 1.0    # NOTE this means order matters, and must be enabled in glInit() below
        self.edgeAlpha = 1.0
        self.lineWidth = 1.0
        self.drawShape = True
        self.drawEdges = False
        self.drawVertices = False
        self.drawVectors = True

        if perfTarget not in ['nicest', 'fastest']:
            raise ValueError("perfTarget must be either 'nicest' or 'fastest'")
        self.perfTarget = perfTarget

        # Data being displayed
        self.mesh = None
        self.nVerts = -1
        self.nFaces = -1
        self.nEdges = -1
        self.dataCenter = np.array([0.0,0.0,0.0])  # A reasonable center for the data
        self.scaleFactor = 1.0         # A scale factor for the magnitude of the data
        self.retainVertAttr = []       # Mesh vertex attributes that are important (= displayed)

        # Display members
        self.meshPrograms = []
        self.shapeProg = None
        self.edgeProg = None
        self.vertProg = None
        self.surfVecProg = None
        self.camera = None

        # Coloring options
        self.colorMethod = 'constant' # other possibilites: 'rgbData','scalarData'
        self.colorAttrName = None
        self.vMinMax = None
        self.cmapName = None
        self.vertexDotColor = 'black'

        # Surface vector drawing options
        self.vectorMethod = 'none'  # other possbilities: 'directionData', 'vectorData'
        self.vectorAttrName = None
        self.vectorRefDirAttrName = None
        self.vectorNsym = None            # symmetry order of the data to be drawn
        self.vectorColor = np.array(self.colors['red'])
        self.nVector = -1           # The number of vectors to be drawn
        self.nVectorVerts = -1      # The size of the vertex buffer needed for
                                    #   specifying vector positions

        # Set window parameters
        self.windowTitle = windowTitle
        self.windowWidth = width
        self.windowHeight = height

        # Initialize the GL/Glut environment
        self.initGLUT()
        print("  init'd GLUT")
        self.initGL()
        print("  init'd GL")


        # Set up the camera
        self.camera = Camera(width, height)

        # Set up callbacks
        glutDisplayFunc(self.redraw)
        glutReshapeFunc(self.resize)
        glutKeyboardFunc(self.keyfunc)
        glutMouseFunc(self.mousefunc)
        glutMotionFunc(self.motionfunc)
        self.userKeyCallbacks = dict() # key => (function, docstring)

        # Set colors
        self.shapeColor = np.array(self.colors['orange'])
        self.edgeColorLight = np.array(self.colors['almost_white'])
        self.edgeColorDark = np.array(self.colors['dark_grey'])

        # Set up the mesh shader program
        self.prepareShapeProgram()
        self.prepareEdgeProgram()
        self.prepareVertexProgram()

        if mesh is not None:
            self.setMesh(mesh)


    def prepareShapeProgram(self):
        """Create an openGL program and all associated buffers to render mesh triangles"""

        self.shapeProg = ShaderProgram(vertShaders[self.shapeVertShader],
                                      fragShaders[self.shapeFragShader])

        # Bind the output location for the fragment shader
        glBindFragDataLocation(self.shapeProg.handle, 0, "outputF");

        # Create uniforms
        self.shapeProg.createUniform('projMatrix', 'u_projMatrix', 'matrix_4')
        self.shapeProg.createUniform('viewMatrix', 'u_viewMatrix', 'matrix_4')
        self.shapeProg.createUniform('alpha', 'u_alpha', 'scalar')
        self.shapeProg.createUniform('eyeLoc', 'u_eye', 'vector_3')
        self.shapeProg.createUniform('lightLoc', 'u_light', 'vector_3')
        self.shapeProg.createUniform('dataCenter', 'u_dataCenter', 'vector_3')

        # Make a VAO for the mesh
        self.shapeProg.createVAO('meshVAO')

        # VBO for positions
        self.shapeProg.createVBO(
            vboName = 'vertPos', varName = 'a_position', vaoName = 'meshVAO', nPerVert = 3)

        # VBO for colors
        self.shapeProg.createVBO(
            vboName = 'vertColor', varName = 'a_color', vaoName = 'meshVAO', nPerVert = 3)

        # VBO for normals
        self.shapeProg.createVBO(
            vboName = 'vertNorm', varName = 'a_normal', vaoName = 'meshVAO', nPerVert = 3)

        # VBO for array indices
        self.shapeProg.createIndexBuffer(vboName = 'triIndex',  vaoName = 'meshVAO')

        # Set a draw function for the mesh program
        def drawMesh():
            # Setup
            glBindVertexArray(self.shapeProg.vaoHandle['meshVAO'])

            # Draw
            glDrawElements(GL_TRIANGLES, 3*self.nFaces, GL_UNSIGNED_INT, None)
            # glDrawElements(GL_LINES, self.nEdges, GL_UNSIGNED_INT, None)
        self.shapeProg.drawFunc = drawMesh

        self.meshPrograms.append(self.shapeProg)


    def prepareEdgeProgram(self):
        """Create an openGL program and all associated buffers to render mesh edges"""

        self.edgeProg = ShaderProgram(vertShaders[self.edgeVertShader],
                                      fragShaders[self.edgeFragShader])

        # Bind the output location for the fragment shader
        glBindFragDataLocation(self.edgeProg.handle, 0, "outputF");

        # Create uniforms
        self.edgeProg.createUniform('projMatrix', 'u_projMatrix', 'matrix_4')
        self.edgeProg.createUniform('viewMatrix', 'u_viewMatrix', 'matrix_4')
        self.edgeProg.createUniform('color', 'u_color', 'vector_3')
        self.edgeProg.createUniform('alpha', 'u_alpha', 'scalar')
        self.edgeProg.createUniform('depthOffset', 'u_depthOffset', 'scalar')
        self.edgeProg.createUniform('eyeLoc', 'u_eye', 'vector_3')
        self.edgeProg.createUniform('lightLoc', 'u_light', 'vector_3')
        self.edgeProg.createUniform('dataCenter', 'u_dataCenter', 'vector_3')

        # Make a VAO for the mesh
        self.edgeProg.createVAO('meshVAO')

        # VBO for positions
        self.edgeProg.createVBO(
            vboName = 'vertPos', varName = 'a_position', vaoName = 'meshVAO', nPerVert = 3)

        # VBO for normals
        self.edgeProg.createVBO(
            vboName = 'vertNorm', varName = 'a_normal', vaoName = 'meshVAO', nPerVert = 3)

        # VBO for array indices
        self.edgeProg.createIndexBuffer(vboName = 'edgeIndex',  vaoName = 'meshVAO')

        # Set a draw function for the mesh program
        def drawMesh():
            # Setup
            glBindVertexArray(self.edgeProg.vaoHandle['meshVAO'])

            # Draw
            glDrawElements(GL_LINES, 2*self.nEdges, GL_UNSIGNED_INT, None)
        self.edgeProg.drawFunc = drawMesh

        self.meshPrograms.append(self.edgeProg)


    def prepareVertexProgram(self):
        """Create an openGL program to draw a dot at every vertex"""

        # Need to make sure we can adjust the size of the points first
        glEnable( GL_PROGRAM_POINT_SIZE )

        self.vertProg = ShaderProgram(vertShaders['surf-draw'],
                                      fragShaders['surf-draw'])

        # Bind the output location for the fragment shader
        glBindFragDataLocation(self.vertProg.handle, 0, "outputF");

        # Create uniforms
        self.vertProg.createUniform('projMatrix', 'u_projMatrix', 'matrix_4')
        self.vertProg.createUniform('viewMatrix', 'u_viewMatrix', 'matrix_4')
        # self.vertProg.createUniform('color', 'u_color', 'vector_3')
        self.vertProg.createUniform('alpha', 'u_alpha', 'scalar')
        self.vertProg.createUniform('depthOffset', 'u_depthOffset', 'scalar')
        self.vertProg.createUniform('eyeLoc', 'u_eye', 'vector_3')
        self.vertProg.createUniform('lightLoc', 'u_light', 'vector_3')
        self.vertProg.createUniform('dataCenter', 'u_dataCenter', 'vector_3')

        # Make a VAO for the mesh
        self.vertProg.createVAO('meshVAO')

        # VBO for positions
        self.vertProg.createVBO(
            vboName = 'vertPos', varName = 'a_position', vaoName = 'meshVAO', nPerVert = 3)

        # VBO for normals
        self.vertProg.createVBO(
            vboName = 'vertNorm', varName = 'a_normal', vaoName = 'meshVAO', nPerVert = 3)

        # VBO for colors
        self.vertProg.createVBO(
            vboName = 'vertColor', varName = 'a_color', vaoName = 'meshVAO', nPerVert = 3)


        # Set a draw function for the mesh program
        def drawMesh():
            # Setup
            glBindVertexArray(self.vertProg.vaoHandle['meshVAO'])

            # Draw
            glDrawArrays(GL_POINTS, 0, self.nVerts)
        self.vertProg.drawFunc = drawMesh

        self.meshPrograms.append(self.vertProg)


    def prepareSurfVecProgram(self):
        """Create an openGL program to draw vectors on the surface of a mesh"""

        self.surfVecProg = ShaderProgram(vertShaders['surf-draw'],
                                      fragShaders['surf-draw'])

        # Bind the output location for the fragment shader
        glBindFragDataLocation(self.surfVecProg.handle, 0, "outputF");

        # Create uniforms
        self.surfVecProg.createUniform('projMatrix', 'u_projMatrix', 'matrix_4')
        self.surfVecProg.createUniform('viewMatrix', 'u_viewMatrix', 'matrix_4')
        # self.surfVecProg.createUniform('color', 'u_color', 'vector_3')
        self.surfVecProg.createUniform('alpha', 'u_alpha', 'scalar')
        self.surfVecProg.createUniform('depthOffset', 'u_depthOffset', 'scalar')
        self.surfVecProg.createUniform('eyeLoc', 'u_eye', 'vector_3')
        self.surfVecProg.createUniform('lightLoc', 'u_light', 'vector_3')
        self.surfVecProg.createUniform('dataCenter', 'u_dataCenter', 'vector_3')

        # Make a VAO for the mesh
        self.surfVecProg.createVAO('meshVAO')

        # VBO for positions
        self.surfVecProg.createVBO(
            vboName = 'vertPos', varName = 'a_position', vaoName = 'meshVAO', nPerVert = 3)

        # VBO for normals
        self.surfVecProg.createVBO(
            vboName = 'vertNorm', varName = 'a_normal', vaoName = 'meshVAO', nPerVert = 3)

        # VBO for colors
        self.surfVecProg.createVBO(
            vboName = 'vertColor', varName = 'a_color', vaoName = 'meshVAO', nPerVert = 3)


        # Set a draw function for the mesh program
        def drawMesh():
            # Setup
            glBindVertexArray(self.surfVecProg.vaoHandle['meshVAO'])

            # Draw
            glDrawArrays(GL_LINES, 0, self.nVectorVerts)
        self.surfVecProg.drawFunc = drawMesh

        self.meshPrograms.append(self.surfVecProg)


    def setMesh(self, mesh):
        """Set a mesh as the current mesh object to be drawn by this viewer"""

        # TODO Properly deallocate buffers to make sure nothing is leaking if this
        # is called many times

        # Save the mesh
        self.mesh = mesh

        ## In general, mesh data is read using readNewMeshValues(). However,
        ## we also read from the mesh to do some initial setup.
        triSoupMesh = self.currMeshAsSoupWithProperties()


        self.nFaces = len(triSoupMesh.tris)
        self.nVerts = len(triSoupMesh.verts)
        self.nEdges = 3*self.nFaces

        # Compute scale and centering factors. These are stored in the uniforms
        # and applied in the shaders
        self.computeScale(triSoupMesh)
        self.camera.zoomDist = self.scaleFactor

        # Initial camera distance
        self.camera.zoomDist = self.scaleFactor


        # Connectivity information is stored here, rather than in readNewMeshValues(),
        # because it should not be changing after the mesh is initialized.
        faceIndData = triSoupMesh.tris.astype(np.uint32)
        edgeIndData = self.generateEdges(triSoupMesh).astype(np.uint32)

        # Face drawing data
        # Store the vertex position, triangle indices, and normals in the buffers
        glBindVertexArray(self.shapeProg.vaoHandle['meshVAO'])
        self.shapeProg.setVBOData('triIndex', faceIndData)

        # Edge drawing data
        # Store the vertex position and triangle indices in the buffers
        glBindVertexArray(self.edgeProg.vaoHandle['meshVAO'])
        self.edgeProg.setVBOData('edgeIndex', edgeIndData)


    def currMeshAsSoupWithProperties(self):
        """
        Returns the current mesh as a triangle soup mesh, converting as necessary
        and preseving any important properties
        """

        # If the input mesh is a halfedge mesh, convert it to a trisoupmesh first
        if type(self.mesh) is HalfEdgeMesh:

            # Perform the conversion
            # Just keep all of the attributes. Some of them might be used for
            # coloring/annotation and we don't want to worry about it
            triSoupMesh = self.mesh.toTriSoupmesh(retainVertAttr=self.retainVertAttr)

        # Typecheck for good luck
        elif type(self.mesh) is not TriSoupMesh:
            raise ValueError("ERROR: setMesh can only accept the types 'TriSoupMesh' and 'HalfEdgeMesh'")
        else:
            triSoupMesh = self.mesh

        return triSoupMesh

    def setMeshColorFromRGB(self, colorAttrName):
        """Sets the mesh face color from RGB data defined on the vertices"""

        self.colorMethod = 'rgbData'
        self.colorAttrName = colorAttrName
        self.retainVertAttr.append(colorAttrName)

    def setMeshColorFromScalar(self, colorScalarAttr, vMinMax=None, cmapName='OrRd'):
        """
        Sets the mesh vertex color from a scalar field defined on the vertices
          - vMinMax: tuple (min, max) giving the bounds for the scalar color scale (the data min/max are used if None)
          - cmapName: colormap to use for scalar colors (any matplotlib colormap name)
                      I recommend 'OrRd' for magnitude data and 'coolwarm' for
                      negative/positive data (classic blue/red)
        """

        self.colorMethod = 'scalarData'
        self.colorAttrName = colorScalarAttr
        self.retainVertAttr.append(colorScalarAttr)
        self.vMinMax = vMinMax
        self.cmapName = cmapName

    def setSurfaceDirections(self, vectorAttrName, vectorRefDirAttrName='referenceDirectionR3', nSym=1):
        """Draws vectors on the surface of the mesh"""

        self.vectorMethod = 'directionData'
        self.vectorAttrName = vectorAttrName
        self.retainVertAttr.append(vectorAttrName)
        self.vectorRefDirAttrName = vectorRefDirAttrName
        self.retainVertAttr.append(vectorRefDirAttrName)
        self.vectorNsym = nSym

        self.prepareSurfVecProgram()


    def readNewMeshValues(self):
        """
        Updates mesh values in the viewer (meaning positions and possibly colors),
        assuming they have been changed on the mesh reference stored herein.
        Normals and colormap things are recalculated internally automatically.

        Note: The structure of the mesh MAY NOT be changed using this method.
        (You may NOT add/remove/modify verts/edges, and this may fail badly)
        """

        # Get the current mesh, converted as needed
        triSoupMesh = self.currMeshAsSoupWithProperties()

        # Compute a general scale factor used to set nice defaults regardless
        # of the scale of the mesh.
        self.computeScale(triSoupMesh)

        # Vertex positions
        vertPosData = triSoupMesh.verts.astype(np.float32)

        # Normals
        triSoupMesh.computeNormals()
        normData = np.array(triSoupMesh.vertAttr['normal'], dtype=np.float32)

        # Get the color using the appropriate color method (generateColor uses
        # one of several options internally based on global state)
        colorData = self.generateColor(triSoupMesh)

        # Face drawing data
        # Store the vertex position, triangle indices, and normals in the buffers
        glBindVertexArray(self.shapeProg.vaoHandle['meshVAO'])
        self.shapeProg.setVBOData('vertPos', vertPosData)
        self.shapeProg.setVBOData('vertNorm', normData)
        self.shapeProg.setVBOData('vertColor', colorData)

        # Edge drawing data
        # Store the vertex position and triangle indices in the buffers
        glBindVertexArray(self.edgeProg.vaoHandle['meshVAO'])
        self.edgeProg.setVBOData('vertPos', vertPosData)
        self.edgeProg.setVBOData('vertNorm', normData)

        # Vertex drawing data
        glBindVertexArray(self.vertProg.vaoHandle['meshVAO'])
        self.vertProg.setVBOData('vertPos', vertPosData)
        self.vertProg.setVBOData('vertNorm', normData)
        vertexDotColors = self.constantVertexColor(self.colors[self.vertexDotColor])
        self.vertProg.setVBOData('vertColor', vertexDotColors)

        # Surface vector drawing data, if applicable
        if self.vectorMethod != 'none':
            (vecPos, vecNorm, vecColor) = self.generateVertexVectorData(triSoupMesh, nSym = self.vectorNsym)
            glBindVertexArray(self.surfVecProg.vaoHandle['meshVAO'])
            self.surfVecProg.setVBOData('vertPos', vecPos)
            self.surfVecProg.setVBOData('vertNorm', vecNorm)
            self.surfVecProg.setVBOData('vertColor', vecColor)


    def generateColor(self, triSoupMesh):
        """
        Generate a color data array for the mesh, either with a constant color,
        specified vertex colors, or specified vertex scalars + colormap
        """

        # Apply the constant default color
        if self.colorMethod == 'constant':
            colorData = np.tile(self.shapeColor[0:3], (self.nVerts,1))

        # Apply colors directly from an [0,1]^3 value defined at each vertex
        elif self.colorMethod == 'rgbData':
            colorData = np.array(triSoupMesh.vertAttr[self.colorAttrName], dtype=np.float32).reshape((self.nVerts,3))

        # Assign colors from a scalar defined over the surface using a colormap
        elif self.colorMethod == 'scalarData':

            scalarData = np.array(triSoupMesh.vertAttr[self.colorAttrName], dtype=np.float32).reshape((self.nVerts))

            # Make sure we have valid bounds for the data
            if self.vMinMax is not None:
                if(self.vMinMax[0] >= self.vMinMax[1]):
                    raise ValueError("ERROR: min bound must be strictly less than max")
                vMinMax = self.vMinMax
            else:
                vMinMax = [scalarData.min(), scalarData.max()]
                # Make sure we don't go crazy if the data is a constant
                scale =  max((abs(vMinMax[0]),abs(vMinMax[1])))
                if (abs(vMinMax[0] - vMinMax[1]) / scale) < 0.0000001:
                    print("WARNING: mesh vertex color scalar was nearly constant, adjusting bounds slightly to draw")
                    vMinMax = [vMinMax[0] - scale*0.01, vMinMax[1] + scale*0.01]

            print("Coloring with scalar data " + str(self.colorAttrName) + " range is " + str(vMinMax[0]) + "  ---  " + str(vMinMax[1]))

            # Remap the data in to [0,1] (if values are outside the range cmap will clamp below)
            span = vMinMax[1] - vMinMax[0]
            normalizedData = (scalarData - vMinMax[0]) / span

            # Get color values from the colormap
            cmap = matplotlib.cm.get_cmap(self.cmapName)
            colorData = cmap(normalizedData)[:,0:3]

        else:
            raise ValueError("ERROR: Unrecognized colorMethod: " + str(self.colorMethod))


        return colorData.astype(np.float32)

    def constantVertexColor(self, color):
        """Helper function to create an array reprsenting a constant color at each vertex"""
        return np.tile(color[0:3], (self.nVerts,1)).astype(np.float32)

    def generateEdges(self, mesh):
        """Generate an index array for mesh edges from the triangle indices"""

        # NOTE: The i'th edge of the n'th triangle is at index i*nTri + n
        e1 = mesh.tris[:,0:2]
        e2 = mesh.tris[:,1:3]
        e3 = np.vstack((mesh.tris[:,2], mesh.tris[:,0])).T
        edges = np.vstack((e1,e2,e3))
        return edges

    def generateVertexVectorData(self, triSoupMesh, nSym=1):
        """Computes data for drawing vectors on the surface, as instructed by setSurfaceDirections()"""

        # TODO the for-loops in here could probably be numpy-ified for a decent speedup

        ## Construct an array filled with the vectors in R3 for each vertex
        vertLocs = triSoupMesh.verts
        tris = triSoupMesh.tris
        normals = triSoupMesh.vertAttr['normal']
        refDirs = triSoupMesh.vertAttr[self.vectorRefDirAttrName]
        angles = triSoupMesh.vertAttr[self.vectorAttrName]

        # Compute a reasonable length for the direction vectors
        # 0.4 * (The 80'th percentile of edge lengths when edges are sorted)
        percentile = 0.80
        edgeLens = []
        for i in range(vertLocs.shape[0]):
            for j in range(3):
                v0 = vertLocs[tris[i,j]]
                v1 = vertLocs[tris[i,(j+1)%3]]
                edgeLens.append(norm(v1-v0))
        edgeLens.sort()
        coef = 0.4 if nSym == 1 else 0.2 # lines should be shorter if we're drawing >1 per vertex
        unitLength = coef*edgeLens[int(round(len(edgeLens)*percentile))]

        # Compute the rotation increment and size to support symmetry
        rotInc = 2.0 * pi / nSym
        nTotalVector = vertLocs.shape[0] * nSym

        # TODO do rotation with euclid for now until I get around to implementing
        # with numpy
        vertVec = np.zeros((nTotalVector,3))
        for i in range(vertLocs.shape[0]):
            for iRot in range(nSym):
                normal = eu.Vector3(*normals[i,:])
                refDir = eu.Vector3(*refDirs[i,:])
                vecDir = refDir.rotate_around(normal, angles[i] + rotInc * iRot)
                vertVec[nSym*i + iRot,:] = vecDir

        # Give all of the vectors length unitLength
        vertVec = normalize(vertVec) * unitLength

        # Interleave the vertices and the (vertices + direction) to get the final
        # answer
        vertVecDraw = np.zeros((2*nTotalVector,3))
        for i in range(vertLocs.shape[0]):
            for iRot in range(nSym):
                vertVecDraw[2*(nSym*i + iRot),:] = vertLocs[i,:]
                vertVecDraw[2*(nSym*i + iRot) + 1,:] = vertLocs[i,:] + vertVec[i*nSym+iRot,:]


        self.nVector = nTotalVector
        self.nVectorVerts = nTotalVector*2

        # Normals need to align
        vertVecDrawNormals = np.zeros((2*nTotalVector,3))
        for i in range(vertLocs.shape[0]):
            for iRot in range(nSym):
                vertVecDrawNormals[2*(nSym*i + iRot),:] = normals[i,:]
                vertVecDrawNormals[2*(nSym*i + iRot) + 1,:] = normals[i,:]

        # Constant colors
        vertVecDrawColors = np.tile(self.vectorColor, (self.nVectorVerts,1))

        return (vertVecDraw.astype(np.float32), vertVecDrawNormals.astype(np.float32), vertVecDrawColors.astype(np.float32))


    def computeScale(self, mesh):
        """
        Compute scale factors and translations so that we get a good default view
        no matter what the scale or translation of the vertex data is
        """

        # Center of the mesh
        bboxMax = np.amax(mesh.verts,0)
        bboxMin = np.amin(mesh.verts,0)
        bboxCent = 0.5 * (bboxMax + bboxMin)
        self.dataCenter = bboxCent

        # Scale factor
        size = max((bboxMax - bboxMin))
        self.scaleFactor = size


    def redraw(self):
        """The almighty draw function"""

        # Clear the disply buffer
        glClearColor(*(np.append(self.colors['dark_grey'],[0.0])).astype(np.float32))
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set the camera view data
        for p in self.meshPrograms: p.setUniform('viewMatrix', self.camera.viewMat())
        for p in self.meshPrograms: p.setUniform('projMatrix', self.camera.projMat())

        # Set a data translation
        for p in self.meshPrograms: p.setUniform('dataCenter', self.dataCenter)

        # Set color and transparency uniforms
        self.shapeProg.setUniform('alpha', self.shapeAlpha)
        if self.drawShape:  # Switch to a dark color as needed to contrast dark background
            self.edgeProg.setUniform('color', self.edgeColorDark)
        else:
            self.edgeProg.setUniform('color', self.edgeColorLight)
        self.edgeProg.setUniform('alpha', self.edgeAlpha)

        # Set up camera and light position in world space, for the shaders that
        # want it
        cameraPos = self.camera.getPos()
        upDir = self.camera.getUp()

        # Light is above the camera
        # lightLoc = (1.7*cameraPos + 0.3*normalize(upDir)*np.linalg.norm(cameraPos)).astype(np.float32)
        lightLoc = 300*cameraPos # TODO this 300 constant is _probably_ not an ideal solution...
        for p in self.meshPrograms: p.setUniform('eyeLoc', cameraPos)
        for p in self.meshPrograms: p.setUniform('lightLoc', lightLoc)

        # Set a depth offset to prevent z-fighting while drawing edges on top
        # of the mesh
        self.edgeProg.setUniform('depthOffset', 0.0001 * self.scaleFactor)
        if self.surfVecProg is not None:
            self.surfVecProg.setUniform('depthOffset', 0.0003 * self.scaleFactor)
        if self.vertProg is not None:
            self.vertProg.setUniform('depthOffset', 0.0005 * self.scaleFactor)


        # Draw the mesh
        if self.drawEdges:
            self.edgeProg.draw()
        if self.drawShape:
            self.shapeProg.draw()
        if self.vectorMethod != 'none' and self.drawVectors:
            self.surfVecProg.draw()
        if self.drawVertices and self.vertProg is not None:
            self.vertProg.draw()

        glutSwapBuffers()

    def initGL(self):
        """Initialize openGL"""

        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_DEPTH_TEST)

        # Transparency related options
        # glEnable(GL_BLEND);
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        # Get some antialiasing enabled
        # glEnable(GL_BLEND);
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        # This incurs a SIGNIFICANT performance hit on large meshes
        # Supposedly line antialiasing is not implemented in hardware on many
        # modern consumer machines, only on workstation cards
        if self.perfTarget == 'nicest':
            glEnable(GL_LINE_SMOOTH)
        # glHint(GL_LINE_SMOOTH_HINT, GL_FASTEST);

        glLineWidth(self.lineWidth)

    def initGLUT(self):
        """Initialize glut"""

        glutInit()

        # Switch statements are definitely platform independent
        print("  Platform is: " + sys.platform)
        if sys.platform == "linux" or sys.platform == "linux2":
            # linux
            glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE)
        elif sys.platform == "darwin":
            # OS X
            glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE)
        elif sys.platform == "win32":
            # Windows...
            glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        else:
            raise OSError("Unrecognized platform: " + str(sys.platform))

        glutInitWindowSize(self.windowWidth, self.windowHeight)
        glutInitWindowPosition(200, 200)
        glutCreateWindow(self.windowTitle)


    def startMainLoop(self):
        """
        Begin running the main loop for the display program. Control flow now
        belongs to openGL until the viewer window is exited. User code will only
        be executed as the result of already-registered callbacks.
        """

        print("Starting MeshDisplay openGL main loop")

        # This reads values from the mesh and stores them in buffers. Do it here
        # so the ordering for setup calls doesn't matter
        self.readNewMeshValues()

        glutMainLoop()


    ### Mouse/keyboard callbacks


    def mousefunc(self, button, state, x, y):
        """Mouse-click callback"""

        # Test if the shift key is being held
        modState = glutGetModifiers()
        shiftHeld = (modState & GLUT_ACTIVE_SHIFT) != 0

        if button == GLUT_RIGHT_BUTTON:
            pass
        else:
            self.camera.processMouse(button, state, x, y, shiftHeld)


        glutPostRedisplay()

    def keyfunc(self, key, x, y):
        """Keyboard callback"""
        if key == chr(27): # escape key
            exit(0)
        if key == 'm':
            self.drawEdges = not self.drawEdges
        elif key == 'k':
            self.drawShape = not self.drawShape
        elif key == 'l':
            self.drawVectors = not self.drawVectors
        elif key == 'n':
            self.drawVertices = not self.drawVertices
        elif key == 'h':
            self.printKeyCallbacks()
        elif key in self.userKeyCallbacks:
            self.userKeyCallbacks[key][0]()
        else:
            # Pass it on to the camera to see if the camera wants to do something
            # with it
            self.camera.processKey(key, x, y)


        glutPostRedisplay()

    def registerKeyCallback(self, key, function, docstring="Application-defined command"):
        """
        Register a new keyboard command with the view. The given function (which
        should take no argument) will be called when the key is pressed. A redraw
        of the of the viewer is automatically triggered when any key is pressed,
        so there is no need to trigger one within your callback function.
        """

        reservedKey = [chr(27),'m','k','h','r','f','w','a','s','d','l','n']
        if(key in reservedKey):
            raise ValueError("ERROR: Cannot register key callback for " + key + ", key is reserved for the viewer application")

        self.userKeyCallbacks[key] = (function, docstring)

    def printKeyCallbacks(self):
        """Print out a list of all keyboard commands for the application"""

        print("\n\n==== Keyboard controls ====\n")
        print("== Viewer commands")
        print("esc   ----  Exit the viewer")
        print("wasd  ----  Pan the current view (also shift-mousedrag)")
        print("r     ----  Zoom the view in")
        print("f     ----  Zoom the view out")
        print("h     ----  Print this help dialog")
        print("m     ----  Toggle drawing the edges of the mesh")
        print("k     ----  Toggle drawing the faces of the mesh")
        print("n     ----  Toggle drawing the vertices of the mesh")
        print("l     ----  Toggle drawing the vector data on the mesh surface (if set)")

        if len(self.userKeyCallbacks) > 0:
            print("\n== Application commands")
            for key in self.userKeyCallbacks:
                print(key+"     ----  "+self.userKeyCallbacks[key][1])

        print("\n")

    def motionfunc(self, x, y):
        """Mouse-move callback"""

        self.camera.processMotion(x, y)
        glutPostRedisplay()


    def resize(self, w, h):
        """Window-resize callback"""

        self.windowWidth = w
        self.windowHeight = h

        glViewport(0, 0, self.windowWidth, self.windowHeight)

        self.camera.updateDimensions(w, h)

        glutPostRedisplay()


class ShaderProgram(object):
    """Convenience class to encapsulate the variables and logic that contribute to an openGL shader program"""

    knownUniformTypes = ['matrix_4', 'vector_3', 'vector_4', 'scalar']

    def __init__(self, vertShaderFile, fragShaderFile, geomShaderFile = None):

        ## Member variables
        self.drawFunc = None # Draw function (should be a function which takes no arguments)

        # VAOs
        self.vaoHandle = dict()

        # VBOs
        self.vboVarName = dict()
        self.vboBuff = dict()
        self.vboAttr = dict()
        self.vboType = dict()

        # Uniforms
        self.uniformLoc = dict()
        self.uniformVarName = dict()
        self.uniformType = dict()


        ## Create the shader program
        vertShader = glCreateShader(GL_VERTEX_SHADER)
        fragShader = glCreateShader(GL_FRAGMENT_SHADER)
        # geomShader = glCreateShader(GL_GEOMETRY_SHADER)

        glShaderSource(vertShader, readFile(vertShaderFile))
        glShaderSource(fragShader, readFile(fragShaderFile))

        # Compile the vertex shader
        glCompileShader(vertShader);
        result = glGetShaderiv(vertShader, GL_COMPILE_STATUS)
        # print("Compile status: " + str(result))
        if not(result):
            raise RuntimeError(glGetShaderInfoLog(vertShader))

        # Compile the fragment shader
        glCompileShader(fragShader);
        result = glGetShaderiv(fragShader, GL_COMPILE_STATUS)
        # print("Compile status: " + str(result))
        if not(result):
            raise RuntimeError(glGetShaderInfoLog(fragShader))

        # Link the program from the vertex and fragment shaders
        self.handle = glCreateProgram();
        glAttachShader(self.handle, vertShader)
        glAttachShader(self.handle, fragShader)
        glLinkProgram(self.handle);


    def activateProgram(self):
        """Sets this program as the active program in the openGL state machine"""

        glUseProgram(self.handle)


    def createUniform(self, name, variableName, dataType):
        """Creates a new uniform value to be used for this program"""

        # Validate the type
        if dataType not in self.knownUniformTypes:
            raise ValueError("I don't know how to create a uniform of type " + str(dataType))

        # Create the uniform
        self.activateProgram()
        self.uniformLoc[name] = glGetUniformLocation(self.handle, variableName)
        self.uniformVarName[name] = variableName
        self.uniformType[name] = dataType


    def setUniform(self, name, value):
        """
        Assign the value of a uniform in the program. Normally called on every
        draw iteration
        """

        self.activateProgram()
        glBindVertexArray(self.vaoHandle['meshVAO'])

        if self.uniformType[name] == 'matrix_4':
            glUniformMatrix4fv(self.uniformLoc[name], 1, True, value)
        elif self.uniformType[name] == 'vector_3':
            glUniform3f(self.uniformLoc[name], value[0], value[1], value[2])
        elif self.uniformType[name] == 'vector_4':
            glUniform4f(self.uniformLoc[name], value[0], value[1], value[2], value[3])
        elif self.uniformType[name] == 'scalar':
            glUniform1f(self.uniformLoc[name], value)


    def createVAO(self, vaoName):
        """Create a new vertex array object"""

        self.activateProgram()
        self.vaoHandle[vaoName] = glGenVertexArrays(1)


    def createVBO(self, vboName, varName, vaoName, nPerVert):
        """Create a new vertex buffer object, attached to the given vertex array"""
        # TODO expose types other than GL_FLOAT
        # TODO implement sharing of VBOs between multiple programs/VAOs

        self.activateProgram()

        # Attach to the variable in the shader program
        self.vboVarName[vboName] = varName
        self.vboAttr[vboName] = glGetAttribLocation(self.handle, varName)

        # Store the type of this VBO
        self.vboType[vboName] = GL_ARRAY_BUFFER

        # Activate the VAO
        glBindVertexArray(self.vaoHandle[vaoName])

        # Create and bind to a buffer
        self.vboBuff[vboName] = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vboBuff[vboName])

        # Associate and enable the attribute in the VAO for vertex-bound data
        glVertexAttribPointer(self.vboAttr[vboName], nPerVert, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(self.vboAttr[vboName]);


    def createIndexBuffer(self, vboName, vaoName):
        """Create a new vertex buffer object which holds index data"""

        self.activateProgram()

        # Activate the VAO
        glBindVertexArray(self.vaoHandle[vaoName])

        # Store the type of this VBO
        self.vboType[vboName] = GL_ELEMENT_ARRAY_BUFFER

        # Create a buffer
        self.vboBuff[vboName] = glGenBuffers(1)


    def setVBOData(self, vboName, data):
        """Set the data in a VBO"""

        self.activateProgram()

        glBindBuffer(self.vboType[vboName], self.vboBuff[vboName])
        glBufferData(self.vboType[vboName], data, GL_STATIC_DRAW)


    def draw(self):
        """Call the user-defined draw function for this program"""

        self.activateProgram()
        self.drawFunc()
