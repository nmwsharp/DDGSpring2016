# Visualize a mesh using openGL

# TODO check out this for drawing edges http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/4884/pdf/imm4884.pdf

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
from HalfEdgeMesh import HalfEdgeMesh, Face, Edge, Vertex, HalfEdge
from TriSoupMesh import TriSoupMesh
from Utilities import normalize, norm, cross
from Camera import Camera


# TODO implement easier to use logic for when to update the data in buffers.
# Don't really want to update every time setXXX() is called, because that might
# be wasteful. However, it would be nice if that didn't need to be tracked externally.
# Maybe set a dirty flag then regenerate data as needed before a draw() call?


# Dictionary of known vertex shaders
vertShaders = {
    'surf-draw': "shaders/surf-draw.vert",
    'flat-draw': "shaders/flat-draw.vert",
}

# Dictionary of known fragment shaders
fragShaders = {
    'surf-draw': "shaders/surf-draw.frag",
    'flat-draw': "shaders/flat-draw.frag",
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
        self.lineWidthCoef = 0.05
        self.lineWidthScaleCoef = 0.05
        self.pointSize = 1.0 # TODO this is currently unused, pointsize is hardcoded in to the shader. Need to expose as a uniform.
        self.nRadialPoints = 12
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
        self.medianEdgeLength = -1

        # Display members
        self.meshPrograms = []
        self.shapeProg = None
        self.edgeProg = None
        self.vertProg = None
        self.meshPickPrograms = []
        self.shapePickProg = None
        self.edgePickProg = None
        self.vertPickProg = None
        self.vectorProg = None
        self.camera = None
        self.nShapeVerts = -1

        # Coloring options
        self.colorMethod = 'constant' # other possibilites: 'rgbData','scalarData'
        self.colorAttrName = None
        self.colorDefinedOn = None # one of 'vertex', 'face', 'edge'
        self.vMinMax = None
        self.cmapName = None

        # Members for picking
        self.pickArray = None
        self.pickInd = dict()
        self.PICK_IND_MAX = 255
        self.pickVertexCallback = None
        self.pickFaceCallback = None
        self.pickEdgeCallback = None

        # Vector drawing options
        self.vectorAttrName = None
        self.vectorDefinedAt = None # One of 'vertex' or 'face'
        self.vectorIsTangent = False
        self.vectorIsUnit = False
        self.vectorRefDirAttrName = None
        self.vectorNsym = 1            # symmetry order of the data to be drawn
        self.nVectorVerts = -1      # The size of the vertex buffer needed for
                                    #   specifying vector positions
        self.kVectorPrism = 8       # Number of sides to use when drawing the vectors

        # Set window parameters
        self.windowTitle = windowTitle
        self.windowWidth = width
        self.windowHeight = height
        self.lastClickPos = None

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
        self.edgeColor = self.edgeColorDark
        # self.vertexDotColor = np.array(self.colors['dark_grey']) # just use edge color
        self.vectorColor = np.array(self.colors['red'])

        # Set up the mesh shader programs for drawing
        self.prepareShapeProgram()
        self.prepareEdgeProgram()
        self.prepareVertexProgram()

        # Set up the mesh shader programs for picking
        self.prepareShapeProgram(pick=True)
        self.prepareEdgePickProgram()
        self.prepareVertexPickProgram()

        if mesh is not None:
            self.setMesh(mesh)


    def preparePrettyShaderProgram(self):
        """Prepare a program which draws using positions/normals/colors with nice shading"""

        prog = ShaderProgram(vertShaders['surf-draw'],
                                      fragShaders['surf-draw'])

        # Bind the output location for the fragment shader
        glBindFragDataLocation(prog.handle, 0, "outputF");

        # Create uniforms
        prog.createUniform('projMatrix', 'u_projMatrix', 'matrix_4')
        prog.createUniform('viewMatrix', 'u_viewMatrix', 'matrix_4')
        prog.createUniform('alpha', 'u_alpha', 'scalar')
        prog.createUniform('eyeLoc', 'u_eye', 'vector_3')
        prog.createUniform('lightLoc', 'u_light', 'vector_3')
        prog.createUniform('dataCenter', 'u_dataCenter', 'vector_3')
        prog.createUniform('depthOffset', 'u_depthOffset', 'scalar')

        # Make a VAO for the mesh
        prog.createVAO('meshVAO')

        # VBO for positions
        prog.createVBO(
            vboName = 'vertPos', varName = 'a_position', vaoName = 'meshVAO', nPerVert = 3)

        # VBO for colors
        prog.createVBO(
            vboName = 'vertColor', varName = 'a_color', vaoName = 'meshVAO', nPerVert = 3)

        # VBO for normals
        prog.createVBO(
            vboName = 'vertNorm', varName = 'a_normal', vaoName = 'meshVAO', nPerVert = 3)

        return prog

    def prepareFlatShaderProgram(self):
        """
        Prepare a program which draws using positions/colors without shading
        (used for picking)
        """

        prog = ShaderProgram(vertShaders['flat-draw'],
                                      fragShaders['flat-draw'])

        # Bind the output location for the fragment shader
        glBindFragDataLocation(prog.handle, 0, "outputF");

        # Create uniforms
        prog.createUniform('projMatrix', 'u_projMatrix', 'matrix_4')
        prog.createUniform('viewMatrix', 'u_viewMatrix', 'matrix_4')
        prog.createUniform('alpha', 'u_alpha', 'scalar')
        prog.createUniform('dataCenter', 'u_dataCenter', 'vector_3')
        prog.createUniform('depthOffset', 'u_depthOffset', 'scalar')

        # Make a VAO for the mesh
        prog.createVAO('meshVAO')

        # VBO for positions
        prog.createVBO(
            vboName = 'vertPos', varName = 'a_position', vaoName = 'meshVAO', nPerVert = 3)

        # VBO for colors
        prog.createVBO(
            vboName = 'vertColor', varName = 'a_color', vaoName = 'meshVAO', nPerVert = 3)

        return prog

    def prepareShapeProgram(self, pick=False):
        """Create an openGL program and all associated buffers to render mesh triangles"""

        if pick:
            prog = self.prepareFlatShaderProgram()
        else:
            prog = self.preparePrettyShaderProgram()

        def drawMesh():
            glBindVertexArray(prog.vaoHandle['meshVAO'])
            glDrawArrays(GL_TRIANGLES, 0, self.nShapeVerts)
        prog.drawFunc = drawMesh

        if pick:
            self.meshPickPrograms.append(prog)
            self.shapePickProg = prog
        else:
            self.meshPrograms.append(prog)
            self.shapeProg = prog


    def prepareEdgeProgram(self):
        """Create an openGL program and all associated buffers to render mesh edges"""

        prog = self.preparePrettyShaderProgram()

        def drawMesh():
            glBindVertexArray(prog.vaoHandle['meshVAO'])
            edgeArrLen = 6*self.nEdges if self.drawShape else 12*self.nEdges
            glDrawArrays(GL_TRIANGLES, 0, edgeArrLen)
        prog.drawFunc = drawMesh

        self.meshPrograms.append(prog)
        self.edgeProg = prog


    def prepareEdgePickProgram(self):
        """Create an openGL program and all associated buffers to render mesh edges"""

        prog = self.prepareFlatShaderProgram()

        def drawMesh():
            glBindVertexArray(prog.vaoHandle['meshVAO'])
            edgeArrLen = 6*self.nEdges if self.drawShape else 12*self.nEdges
            glDrawArrays(GL_TRIANGLES, 0, edgeArrLen)
        prog.drawFunc = drawMesh

        self.meshPickPrograms.append(prog)
        self.edgePickProg = prog


    def prepareVertexProgram(self):
        """Create an openGL program to draw a dot at every vertex"""

        # Need to make sure we can adjust the size of the points first
        glEnable( GL_PROGRAM_POINT_SIZE )

        prog = self.preparePrettyShaderProgram()

        def drawMesh():
            # glPointSize(self.pointSize)
            glBindVertexArray(prog.vaoHandle['meshVAO'])
            vertArrLen = self.nVerts*self.nRadialPoints*3 if self.drawShape else 2*self.nVerts*self.nRadialPoints*3
            glDrawArrays(GL_TRIANGLES, 0, vertArrLen)
        prog.drawFunc = drawMesh

        self.meshPrograms.append(prog)
        self.vertProg = prog


    def prepareVertexPickProgram(self):
        """Create an openGL program to draw a dot at every vertex"""

        # Need to make sure we can adjust the size of the points first
        glEnable( GL_PROGRAM_POINT_SIZE )

        prog = self.prepareFlatShaderProgram()

        def drawMesh():
            # glPointSize(6*self.pointSize)
            glBindVertexArray(prog.vaoHandle['meshVAO'])
            vertArrLen = self.nVerts*self.nRadialPoints*3 if self.drawShape else 2*self.nVerts*self.nRadialPoints*3
            glDrawArrays(GL_TRIANGLES, 0, vertArrLen)
        prog.drawFunc = drawMesh

        self.meshPickPrograms.append(prog)
        self.vertPickProg = prog


    def prepareVectorProgram(self):
        """Create an openGL program to draw vectors on the surface of a mesh"""

        prog = self.preparePrettyShaderProgram()

        def drawMesh():
            glBindVertexArray(prog.vaoHandle['meshVAO'])
            glDrawArrays(GL_TRIANGLES, 0, self.nVectorVerts)
        prog.drawFunc = drawMesh

        self.meshPrograms.append(prog)
        self.vectorProg = prog

    def setMesh(self, mesh):
        """Set a mesh as the current mesh object to be drawn by this viewer"""

        # TODO Properly deallocate buffers to make sure nothing is leaking if this
        # is called many times

        # Save and validate the mesh
        self.mesh = mesh
        self.checkMesh()

        self.nFaces = len(mesh.faces)
        self.nVerts = len(mesh.verts)
        self.nEdges = len(mesh.edges)

        # Compute scale and centering factors. These are stored in the uniforms
        # and applied in the shaders
        self.computeScale()
        self.camera.zoomDist = self.scaleFactor

        # Initial camera distance
        self.camera.zoomDist = self.scaleFactor

    def checkMesh(self):
        """
        Verify that the mesh has triangular faces and valid positions
        """

        # Verify triangle Mesh
        for he in self.mesh.halfEdges:
            if he.next.next.next is not he:
                print("ERROR: Halfedge {} does not have triangular connectivity.".format(str(he)))
                raise ValueError("ERROR: MeshDisplay can only display triangular meshes")


        # Verify non-nan and non-inf positions
        for v in self.mesh.verts:
            if np.any(np.isnan(v.position)) or np.any(np.isinf(v.position)):
                print("ERROR: Invalid position value at vertex {} = {}".format(str(v),str(v.position)))
                raise ValueError("ERROR: Invalid (nan or inf) position value")


    def setShapeColorToDefault(self):
        self.colorMethod = 'constant'
        self.colorDefinedOn = None
        self.colorAttrName = None
        self.vMinMax = None
        self.cmapName = None

    def setShapeColorFromRGB(self, colorAttrName, definedOn='vertex'):
        """Sets the mesh face color from RGB data defined on the vertices"""

        self.colorMethod = 'rgbData'
        self.colorDefinedOn = definedOn
        self.colorAttrName = colorAttrName


    def setShapeColorFromScalar(self, colorScalarAttr, definedOn='vertex', vMinMax=None, cmapName='OrRd'):
        """
        Sets the mesh vertex color from a scalar field defined on the vertices
          - vMinMax: tuple (min, max) giving the bounds for the scalar color scale (the data min/max are used if None)
          - cmapName: colormap to use for scalar colors (any matplotlib colormap name)
                      I recommend 'OrRd' for magnitude data and 'coolwarm' for
                      negative/positive data (classic blue/red)
        """

        self.colorMethod = 'scalarData'
        self.colorDefinedOn = definedOn
        self.colorAttrName = colorScalarAttr
        self.vMinMax = vMinMax
        self.cmapName = cmapName


    def setVectors(self, vectorAttrName, vectorDefinedAt='vertex', isTangentVector=False,
                             vectorRefDirAttrName='referenceDirectionR3', isUnit=False, nSym=1):
        """Draws vectors on the surface of the mesh"""
        # TODO: Right now we can only draw one type of vector at a time

        if vectorAttrName is None:
            self.vectorAttrName = None
            self.vectorDefinedAt = None
            self.vectorIsTangent = None
            self.vectorRefDirAttrName = None
            self.vectorIsUnit = None
            self.vectorNsym = 1
        else:
            self.vectorAttrName = vectorAttrName
            self.vectorDefinedAt = vectorDefinedAt
            self.vectorIsTangent = isTangentVector
            self.vectorRefDirAttrName = vectorRefDirAttrName
            self.vectorIsUnit = isUnit
            self.vectorNsym = nSym

            # Color face and vertex vectors different colors
            if vectorDefinedAt == 'vertex':
                self.vectorColor = np.array(self.colors['red'])
            if vectorDefinedAt == 'face':
                self.vectorColor = np.array(self.colors['light_blue'])

            # Prepare a vector drawing program if we don't already have one
            if self.vectorProg is None:
                self.prepareVectorProgram()

    def prepareToPick(self):
        """Sets up datastructures needed to pick from the current mesh"""

        # Build forward and reverse lookup tables
        self.pickArray = [None] + list(self.mesh.verts) +  list(self.mesh.faces) +  list(self.mesh.edges)
        for i in range(len(self.pickArray)):
            self.pickInd[self.pickArray[i]] = i

        # Valid that we have enough indices to represent this
        if len(self.pickArray) > self.PICK_IND_MAX**3:
            raise("ERROR: Do not have enough indices to support picking on a mesh this large. Pack floats better in picking code")


    def pickIndAsFloats(self, obj):
        """Return the pick index, represented as 3 floats for use as a color"""
        ind = self.pickInd[obj]

        v1 = ind / (self.PICK_IND_MAX**2)
        ind -= (self.PICK_IND_MAX**2) * v1
        v2 = ind / self.PICK_IND_MAX
        ind -= self.PICK_IND_MAX * v2
        v3 = ind

        return np.array([v1,v2,v3], dtype=np.float32)/256.0

    def pickResult(self, vals):
        """Return the object selected by a pick"""
        ind1 = int(vals[0]*256.0)
        ind2 = int(vals[1]*256.0)
        ind3 = int(vals[2]*256.0)
        ind = ind1*self.PICK_IND_MAX*self.PICK_IND_MAX + ind2*self.PICK_IND_MAX + ind3

        return self.pickArray[ind]

    def generateAllMeshValues(self):
        """
        Updates mesh values in the viewer (meaning positions and possibly colors),
        assuming they have been changed on the mesh reference stored herein.
        Normals and colormap things are recalculated internally automatically.

        Note: The structure of the mesh MAY NOT be changed using this method.
        (You may NOT add/remove/modify verts/edges, and this may fail badly)
        """

        ## The data used to generate the image which actually appears onscreen
        self.generateFaceData()
        self.generateEdgeData()
        self.generateVertexData()

        if self.vectorAttrName is not None:
            self.generateVectorData()

        ## Extra buffers/shaders used to support picking
        self.prepareToPick()
        self.generateFaceData(pick=True)
        self.generateEdgeData(pick=True)
        self.generateVertexData(pick=True)

    def generateColorscale(self):
        """
        Create a colormap and bounds for coloring from scalar data
        """

        # Make sure we have valid bounds for the data
        if self.vMinMax is not None:
            if(self.vMinMax[0] >= self.vMinMax[1]):
                raise ValueError("ERROR: min bound must be strictly less than max")
        else:

            vMin = float('inf')
            vMax = -float('inf')
            if self.colorDefinedOn == 'vertex':
                for v in self.mesh.verts:
                    vMin = min((vMin, getattr(v, self.colorAttrName)))
                    vMax = max((vMax, getattr(v, self.colorAttrName)))
            elif self.colorDefinedOn == 'face':
                for f in self.mesh.faces:
                    vMin = min((vMin, getattr(f, self.colorAttrName)))
                    vMax = max((vMax, getattr(f, self.colorAttrName)))
            elif self.colorDefinedOn == 'edge':
                raise NotImplementedError("Edge shape color definitions not implemented yet")
            vMinMax = [vMin, vMax]

            # Make sure we don't go crazy if the data is a constant
            scale =  max((abs(vMinMax[0]),abs(vMinMax[1])))
            if (abs(vMinMax[0] - vMinMax[1]) / scale) < 0.0000001:
                print("WARNING: mesh vertex color scalar was nearly constant, adjusting bounds slightly to draw")
                vMinMax = [vMinMax[0] - scale*0.01, vMinMax[1] + scale*0.01]

            self.vMinMax = vMinMax

        # Get color values from the colormap
        self.scalarDataColorMap = matplotlib.cm.get_cmap(self.cmapName)

        # Make a shorthand function for mapping values to colors
        def mapValueToColor(x):
            return np.array(self.scalarDataColorMap((x - self.vMinMax[0]) / (self.vMinMax[1]-self.vMinMax[0]))[0:3], dtype=np.float32)
        self.scalarColorMapper = mapValueToColor


    def generateFaceData(self, pick=False):
        """Generates the positions, normals, and colors for drawing faces"""

        facePos = np.zeros((3*self.nFaces,3), dtype=np.float32)
        faceNorm = np.zeros((3*self.nFaces,3), dtype=np.float32)
        faceColor = np.zeros((3*self.nFaces,3), dtype=np.float32)

        # If appropriate, make sure we have a colorscale and map.
        # Create a shorthand function to map values to colors
        if self.colorMethod == 'scalarData':
            self.generateColorscale()

        # Iterate through the faces to build arrays
        for (i, face) in enumerate(self.mesh.faces):
            v1 = face.anyHalfEdge.vertex
            v2 = face.anyHalfEdge.next.vertex
            v3 = face.anyHalfEdge.next.next.vertex

            facePos[3*i  ,:] = v1.position
            facePos[3*i+1,:] = v2.position
            facePos[3*i+2,:] = v3.position

            faceNorm[3*i  ,:] = v1.normal
            faceNorm[3*i+1,:] = v2.normal
            faceNorm[3*i+2,:] = v3.normal

            # Select the appropriate color from the many possibile ways the
            # mesh could be colored
            if pick:
                faceIndAsFloats = self.pickIndAsFloats(face)
                faceColor[3*i,  :] = faceIndAsFloats
                faceColor[3*i+1,:] = faceIndAsFloats
                faceColor[3*i+2,:] = faceIndAsFloats
            else:
                if self.colorMethod == 'constant':
                    faceColor[3*i,  :] = self.shapeColor[0:3]
                    faceColor[3*i+1,:] = self.shapeColor[0:3]
                    faceColor[3*i+2,:] = self.shapeColor[0:3]

                elif self.colorMethod == 'rgbData':
                    if self.colorDefinedOn == 'vertex':
                        faceColor[3*i,  :] = getattr(v1, self.colorAttrName)
                        faceColor[3*i+1,:] = getattr(v2, self.colorAttrName)
                        faceColor[3*i+2,:] = getattr(v3, self.colorAttrName)
                    elif self.colorDefinedOn == 'face':
                        faceColor[3*i,  :] = getattr(face, self.colorAttrName)
                        faceColor[3*i+1,:] = getattr(face, self.colorAttrName)
                        faceColor[3*i+2,:] = getattr(face, self.colorAttrName)
                    elif self.colorDefinedOn == 'edge':
                        raise NotImplementedError("Edge shape color definitions not implemented yet")

                elif self.colorMethod == 'scalarData':
                    if self.colorDefinedOn == 'vertex':
                        faceColor[3*i,  :] = self.scalarColorMapper(getattr(v1, self.colorAttrName))
                        faceColor[3*i+1,:] = self.scalarColorMapper(getattr(v2, self.colorAttrName))
                        faceColor[3*i+2,:] = self.scalarColorMapper(getattr(v3, self.colorAttrName))
                    elif self.colorDefinedOn == 'face':
                        faceColor[3*i,  :] = self.scalarColorMapper(getattr(face, self.colorAttrName))
                        faceColor[3*i+1,:] = self.scalarColorMapper(getattr(face, self.colorAttrName))
                        faceColor[3*i+2,:] = self.scalarColorMapper(getattr(face, self.colorAttrName))
                    elif self.colorDefinedOn == 'edge':
                        raise NotImplementedError("Edge shape color definitions not implemented yet")


        # Make edges face both ways
        facePos, faceNorm, faceColor = self.expandTrianglesToFaceBothWays(facePos, faceNorm, faceColor)

        self.nShapeVerts = facePos.shape[0]

        # Store this new data in the buffers
        if pick:
            glBindVertexArray(self.shapePickProg.vaoHandle['meshVAO'])
            self.shapePickProg.setVBOData('vertPos', facePos)
            self.shapePickProg.setVBOData('vertColor', faceColor)
        else:
            glBindVertexArray(self.shapeProg.vaoHandle['meshVAO'])
            self.shapeProg.setVBOData('vertPos', facePos)
            self.shapeProg.setVBOData('vertNorm', faceNorm)
            self.shapeProg.setVBOData('vertColor', faceColor)


    def generateEdgeData(self, pick=False):
        """Generates the positions, normals, and colors for drawing edges"""

        edgePos = np.zeros((2*self.nEdges,3), dtype=np.float32)
        edgeNorm = np.zeros((2*self.nEdges,3), dtype=np.float32)
        edgeColor = np.zeros((2*self.nEdges,3), dtype=np.float32)

        for (i, edge) in enumerate(self.mesh.edges):
            v1 = edge.anyHalfEdge.vertex
            v2 = edge.anyHalfEdge.twin.vertex

            edgePos[2*i,:] = v1.position
            edgePos[2*i+1,:] = v2.position

            edgeNorm[2*i,  :] = v1.normal
            edgeNorm[2*i+1,:] = v2.normal

            if pick:
                edgeIndAsFloats = self.pickIndAsFloats(edge)
                edgeColor[2*i,  :] = edgeIndAsFloats
                edgeColor[2*i+1,:] = edgeIndAsFloats
            else:
                edgeColor[2*i,  :] = self.edgeColor
                edgeColor[2*i+1,:] = self.edgeColor


        # Expand the lines to strips (uncomment alternate verions to scale lines by line length)
        if pick:
            # stripPos, stripNorm, stripColor = self.expandLinesToStrips(edgePos, edgeNorm, edgeColor, lineWidthCoef=3*self.lineWidthCoef)
            stripPos, stripNorm, stripColor = self.expandLinesToStrips(edgePos, edgeNorm, edgeColor,
                                                lineWidth=3*self.lineWidthScaleCoef*self.medianEdgeLength, relativeWidths=False)
        else:
            # stripPos, stripNorm, stripColor = self.expandLinesToStrips(edgePos, edgeNorm, edgeColor, lineWidthCoef=self.lineWidthCoef)
            stripPos, stripNorm, stripColor = self.expandLinesToStrips(edgePos, edgeNorm, edgeColor,
                                                lineWidth=self.lineWidthScaleCoef*self.medianEdgeLength, relativeWidths=False)

        # If we're looking at a wireframe, we need a second set of triangles so
        # we get visibility in both directions
        if not self.drawShape:
            stripPos, stripNorm, stripColor = self.expandTrianglesToFaceBothWays(stripPos, stripNorm, stripColor)

        # Store this new data in the buffers
        if pick:
            glBindVertexArray(self.edgePickProg.vaoHandle['meshVAO'])
            self.edgePickProg.setVBOData('vertPos', stripPos)
            self.edgePickProg.setVBOData('vertColor', stripColor)
        else:
            glBindVertexArray(self.edgeProg.vaoHandle['meshVAO'])
            self.edgeProg.setVBOData('vertPos', stripPos)
            self.edgeProg.setVBOData('vertNorm', stripNorm)
            self.edgeProg.setVBOData('vertColor', stripColor)


    def generateVertexData(self, pick=False):
        """Generates the positions, normals, and colors for drawing verts"""

        vertPos = np.zeros((self.nVerts,3), dtype=np.float32)
        vertNorm = np.zeros((self.nVerts,3), dtype=np.float32)
        vertColor = np.zeros((self.nVerts,3), dtype=np.float32)

        # Iterate through the vertices to fill arrays
        for (i, vert) in enumerate(self.mesh.verts):

            vertPos[i,:] = vert.position
            vertNorm[i,:] = vert.normal
            if pick:
                vertColor[i,:] = self.pickIndAsFloats(vert)
            else:
                vertColor[i,:] = self.edgeColor

        diskPos, diskNorm, diskColor = self.expandDotsToDisks(vertPos, vertNorm, vertColor, dotWidthCoef=self.lineWidthScaleCoef)

        # If we're looking at a wireframe, we need a second set of triangles so
        # we get visibility in both directions
        if not self.drawShape:
            diskPos, diskNorm, diskColor = self.expandTrianglesToFaceBothWays(diskPos, diskNorm, diskColor)

        # Store this new data in the buffers
        if pick:
            glBindVertexArray(self.vertPickProg.vaoHandle['meshVAO'])
            self.vertPickProg.setVBOData('vertPos', diskPos)
            self.vertPickProg.setVBOData('vertColor', diskColor)
        else:
            glBindVertexArray(self.vertProg.vaoHandle['meshVAO'])
            self.vertProg.setVBOData('vertPos', diskPos)
            self.vertProg.setVBOData('vertNorm', diskNorm)
            self.vertProg.setVBOData('vertColor', diskColor)


    def generateVectorData(self):
        """Computes data for drawing vectors on the mesh, as instructed by setSurfaceDirections()"""

        # Clear out if we're not currently looking at anything
        if self.vectorAttrName is None:
            self.nVectorVerts = 0
            return

        # Generate vector starts and ends for vectors in the tangent space
        if self.vectorIsTangent:

            # We only support vertex-defined directions here at the moment (TODO)
            if not self.vectorDefinedAt == 'vertex':
                raise ValueError("Vectors in the tangent space must be defined at vertices")
            if not self.vectorIsUnit:
                raise ValueError("Vectors in the tangent space must be interpreted as unit vectors")

            nSym = self.vectorNsym
            nTotalVector = self.nVerts * nSym
            vecStart = np.zeros((nTotalVector,3), dtype=np.float32)
            vecEnd = np.zeros((nTotalVector,3), dtype=np.float32)

            # Compute a reasonable length for the direction vectors
            # TODO make this vary by vertex
            coef = 0.4 if nSym == 1 else 0.2 # lines should be shorter if we're drawing >1 per vertex
            unitVectorLength = coef*self.medianEdgeLength

            # Rotation increment for symmetric vectors
            rotInc = 2.0 * pi / nSym

            # Iterate over the vertices to fill the arrays
            for (iVert, vert) in enumerate(self.mesh.verts):
                for iRot in range(nSym):

                    # The base point
                    vecStart[(nSym*iVert + iRot),:] = vert.position

                    # The draw-to point
                    theta = getattr(vert, self.vectorAttrName)
                    refDir = eu.Vector3(*getattr(vert, self.vectorRefDirAttrName))
                    vecDir = np.array(refDir.rotate_around(eu.Vector3(*vert.normal), theta + rotInc * iRot), dtype=np.float32)
                    vecEnd[(nSym*iVert + iRot),:] = vert.position + normalize(vecDir) * unitVectorLength

        # Generate vectors in R3
        else:

            # Warn if you try to use nSym other than 1 for now
            if self.vectorNsym != 1:
                raise ValueError("Symmetry not yet supported for vectors in R3")

            # Build an array from data stored on vertices
            if self.vectorDefinedAt == 'vertex':
                nTotalVector = self.nVerts
                vecStart = np.zeros((nTotalVector,3), dtype=np.float32)
                vecEnd = np.zeros((nTotalVector,3), dtype=np.float32)

                for (iVert, vert) in enumerate(self.mesh.verts):
                    vecStart[iVert,:] = vert.position
                    vecEnd[iVert,:] = vert.position + getattr(vert, self.vectorAttrName)

            # Build an array from data stored on faces
            elif self.vectorDefinedAt == 'face':
                nTotalVector = self.nFaces
                vecStart = np.zeros((nTotalVector,3), dtype=np.float32)
                vecEnd = np.zeros((nTotalVector,3), dtype=np.float32)

                for (iFace, face) in enumerate(self.mesh.faces):
                    vecStart[iFace,:] = face.center
                    vecEnd[iFace,:] = face.center + getattr(face, self.vectorAttrName)

            else:
                print("ERROR: Unrecognized value for vectorDefinedAt: " + str(self.vectorDefinedAt))
                print("       Should be one of 'vertex', 'face'")
                raise ValueError("ERROR: Unrecognized value for vectorDefinedAt: " + str(self.vectorDefinedAt))


            # Rescale the vectors to a reasonable length
            vec = vecEnd - vecStart
            maxLen = np.max(norm(vec, axis=1))
            vectorLen = 0.8 * self.medianEdgeLength
            scaleFactor = vectorLen / maxLen
            vecEnd = vecStart + (vecEnd - vecStart)*scaleFactor # kind of redundant way to do this...


        # Expand the vectors in to prisms that look nice
        coef = 0.8 if self.vectorIsTangent else 0.8
        vertVecPos, vertVecNorm, vertVecColor = self.expandVectors(vecStart, vecEnd, coef*self.lineWidthScaleCoef*self.medianEdgeLength)

        # Size of the ultimate buffer containing this vector data
        self.nVectorVerts = vertVecPos.shape[0]

        glBindVertexArray(self.vectorProg.vaoHandle['meshVAO'])
        self.vectorProg.setVBOData('vertPos', vertVecPos)
        self.vectorProg.setVBOData('vertNorm', vertVecNorm)
        self.vectorProg.setVBOData('vertColor', vertVecColor)


    def expandLinesToStrips(self, linePos, lineNorm, lineColor, lineWidth, relativeWidths=True):
        """
        Some platforms (*cough* OSX *cough*) don't support setting linewidth. This function
        takes what would be input to GL_LINES and transforms it to GL_TRIANGLES input as line strips

        If relativeWidths=True, the line widths are set as a fraction of the line's lenght. Otherwise
        the widths are a constant factor of the shape's scale.
        """
        # TODO this is exactly what geometry shaders are meant for

        # Allocate new arrays
        nLines = linePos.shape[0]/2
        stripPos = np.zeros((nLines*6,3), dtype=np.float32)
        stripNorm = np.zeros((nLines*6,3), dtype=np.float32)
        stripColor = np.zeros((nLines*6,3), dtype=np.float32)

        # Directions which will be used in construction
        forwardVec = linePos[1:2*nLines:2,:] - linePos[0:2*nLines:2,:]  # Vector from i-->j along each line
        crossI = np.cross(forwardVec, lineNorm[0:2*nLines:2,:])         # The vector for the width of the strip at i
        crossJ = np.cross(forwardVec, lineNorm[1:2*nLines:2,:])         # The vector for the width of the strip at j

        if relativeWidths:
            # The widths for each line are a factor of the lenght the line
            lineWidth = 0.1 * norm(forwardVec, axis=1)
            crossI = (normalize(crossI).T*lineWidth/2.0).T
            crossJ = (normalize(crossJ).T*lineWidth/2.0).T
        else:
            crossI = normalize(crossI)*lineWidth/2.0
            crossJ = normalize(crossJ)*lineWidth/2.0

        # Generate the 4 points which will make up triangles for each line
        v0 = linePos[0:2*nLines:2,:] + crossI
        v1 = linePos[0:2*nLines:2,:] - crossI
        v2 = linePos[1:2*nLines:2,:] + crossJ
        v3 = linePos[1:2*nLines:2,:] - crossJ

        # The two triangles are (v0,v2,v3) and (v0,v3,v1)
        stripPos[0:6*nLines:6,:] = v0
        stripPos[1:6*nLines:6,:] = v2
        stripPos[2:6*nLines:6,:] = v3
        stripPos[3:6*nLines:6,:] = v0
        stripPos[4:6*nLines:6,:] = v3
        stripPos[5:6*nLines:6,:] = v1

        ## Assign normals and colors to match
        stripNorm[0:6*nLines:6,:] = lineNorm[0:2*nLines:2,:]
        stripNorm[1:6*nLines:6,:] = lineNorm[1:2*nLines:2,:]
        stripNorm[2:6*nLines:6,:] = lineNorm[1:2*nLines:2,:]
        stripNorm[3:6*nLines:6,:] = lineNorm[0:2*nLines:2,:]
        stripNorm[4:6*nLines:6,:] = lineNorm[1:2*nLines:2,:]
        stripNorm[5:6*nLines:6,:] = lineNorm[0:2*nLines:2,:]

        stripColor[0:6*nLines:6,:] = lineColor[0:2*nLines:2,:]
        stripColor[1:6*nLines:6,:] = lineColor[1:2*nLines:2,:]
        stripColor[2:6*nLines:6,:] = lineColor[1:2*nLines:2,:]
        stripColor[3:6*nLines:6,:] = lineColor[0:2*nLines:2,:]
        stripColor[4:6*nLines:6,:] = lineColor[1:2*nLines:2,:]
        stripColor[5:6*nLines:6,:] = lineColor[0:2*nLines:2,:]

        return stripPos, stripNorm, stripColor

    def expandDotsToDisks(self, dotPos, dotNorm, dotColor, dotWidthCoef=None):
        """
        To fit well with expandLinesToStrips(), drawing ugly square points is probably
        not sufficient. Use this to expand points to disks
        """
        # TODO this is exactly what geometry shaders are meant for

        # Size of radial points for each disk
        nRad = self.nRadialPoints
        dotRad = 2 * dotWidthCoef * self.medianEdgeLength

        # Allocate new arrays
        nDots = dotPos.shape[0]
        diskPos = np.zeros((nDots*nRad*3,3), dtype=np.float32)
        diskNorm = np.zeros((nDots*nRad*3,3), dtype=np.float32)
        diskColor = np.zeros((nDots*nRad*3,3), dtype=np.float32)

        # We need to pick an arbitrary x direction in the tangent space of each
        # vertex. However, if the code were to use any fixed direction and that happened
        # to be exactly the normal direction for the vertex, things would break.
        # Using a random vector is almost definitely safe.
        randDir = (2*np.random.rand(3) - np.array([1.0,1.0,1.0])).astype(np.float32)

        # Generate basis vectors at each vertex
        xDir = normalize(np.cross(dotNorm, randDir))*dotRad
        yDir = normalize(np.cross(dotNorm, xDir))*dotRad

        # Walk around a circle placing the points that make up each triangle
        for rotInc in range(nRad):

            theta = (2*pi*rotInc) / nRad
            cTheta = cos(theta)
            sTheta = sin(theta)
            thetaNext = (2*pi*(rotInc+1)) / nRad
            cThetaNext = cos(thetaNext)
            sThetaNext = sin(thetaNext)

            # Centerpoint
            diskPos  [3*rotInc:nDots*nRad*3:nRad*3,:] = dotPos
            diskNorm [3*rotInc:nDots*nRad*3:nRad*3,:] = dotNorm
            diskColor[3*rotInc:nDots*nRad*3:nRad*3,:] = dotColor

            # theta
            diskPos  [3*rotInc+1:nDots*nRad*3:nRad*3,:] = dotPos + cTheta*xDir + sTheta*yDir
            diskNorm [3*rotInc+1:nDots*nRad*3:nRad*3,:] = dotNorm
            diskColor[3*rotInc+1:nDots*nRad*3:nRad*3,:] = dotColor

            # thetaNext
            diskPos  [3*rotInc+2:nDots*nRad*3:nRad*3,:] = dotPos + cThetaNext*xDir + sThetaNext*yDir
            diskNorm [3*rotInc+2:nDots*nRad*3:nRad*3,:] = dotNorm
            diskColor[3*rotInc+2:nDots*nRad*3:nRad*3,:] = dotColor

        return diskPos, diskNorm, diskColor

    def expandTrianglesToFaceBothWays(self, triPos, triNorm, triColor):
        """
        OpenGL only emits a triangle if it wound in the proper direction, which
        means you cannot see a triangle "from beind". This duplicates a triangle,
        creating a new array including dual triangles wound in the opposite direciton.
        Normals for these new triangles are zeroed, which gives a nice darkened
        effect for viewing from behind.
        """

        triPosRev = np.zeros_like(triPos)
        triPosRev[0:triPosRev.shape[0]:3,:] = triPos[1:triPos.shape[0]:3,:]
        triPosRev[1:triPosRev.shape[0]:3,:] = triPos[0:triPos.shape[0]:3,:]
        triPosRev[2:triPosRev.shape[0]:3,:] = triPos[2:triPos.shape[0]:3,:]
        doubleTriPos = np.vstack((triPos, triPosRev))

        triNormRev = np.zeros_like(triNorm)
        triNormRev[0:triNormRev.shape[0]:3,:] = triNorm[1:triNorm.shape[0]:3,:]
        triNormRev[1:triNormRev.shape[0]:3,:] = triNorm[0:triNorm.shape[0]:3,:]
        triNormRev[2:triNormRev.shape[0]:3,:] = triNorm[2:triNorm.shape[0]:3,:]
        doubleTriNorm = np.vstack((triNorm, -triNormRev))

        doubleTriColor = np.vstack((triColor, triColor))

        return doubleTriPos, doubleTriNorm, doubleTriColor


    def expandVectors(self, vecStarts, vecEnds, vectorRadius, drawTips=True):
        """Geneate data for vectors R3 by specifying the start and end of each vector"""

        kPrism = self.kVectorPrism
        nVec = vecStarts.shape[0]
        nLines = kPrism*nVec

        linePos = np.zeros((2*nLines,3), dtype=np.float32)
        lineNorm = np.zeros((2*nLines,3), dtype=np.float32)
        lineColor = np.zeros((2*nLines,3), dtype=np.float32)

        if drawTips:
            tipPos = np.zeros((3*nLines,3), dtype=np.float32)
            tipNorm = np.zeros((3*nLines,3), dtype=np.float32)
            tipColor = np.zeros((3*nLines,3), dtype=np.float32)

        # We are going to build a k-prism around each vector.
        # First, generate the 4 lines that are the axes of the walls of this
        # prism. Then, expand each of those lines to a strip.

        # We need to pick an arbitrary x direction prependicular each
        # vector. However, if the code were to use any fixed direction and that happened
        # to be exactly the normal direction for the vertex, things would break.
        # Using a random vector is almost definitely safe.
        randDir = (2*np.random.rand(3) - np.array([1.0,1.0,1.0])).astype(np.float32)

        # Generate basis vectors at each vector base
        vecs = normalize(vecEnds - vecStarts)
        xDir = normalize(np.cross(vecs, randDir))
        yDir = normalize(np.cross(vecs, xDir))

        if drawTips:
            tipHeight = 2.5 * vectorRadius

        # Walk around a circle placing the walls of the prism
        for rotInc in range(kPrism):

            theta = (2*pi*rotInc) / kPrism
            cTheta = cos(theta)
            sTheta = sin(theta)

            # Vector begin
            linePos[2*rotInc:2*nLines:2*kPrism,:] = vecStarts + (cTheta*xDir + sTheta*yDir) * vectorRadius
            lineNorm[2*rotInc:2*nLines:2*kPrism,:] = cTheta*xDir + sTheta*yDir
            lineColor[2*rotInc:2*nLines:2*kPrism,:] = self.vectorColor

            # Vector end
            linePos[2*rotInc+1:2*nLines:2*kPrism,:] = vecEnds + (cTheta*xDir + sTheta*yDir) * vectorRadius
            lineNorm[2*rotInc+1:2*nLines:2*kPrism,:] = cTheta*xDir + sTheta*yDir
            lineColor[2*rotInc+1:2*nLines:2*kPrism,:] = self.vectorColor


            # Generate triangles in a cone around the tip
            if drawTips:

                # Side length is slightly different than below
                # (see http://mathworld.wolfram.com/RegularPolygon.html)
                tipDist = vectorRadius * 1/cos(pi / kPrism)

                theta = (2*pi*(rotInc+0.5)) / kPrism
                cTheta = cos(theta)
                sTheta = sin(theta)

                thetaNext = (2*pi*(rotInc+1.5)) / kPrism
                cThetaNext = cos(thetaNext)
                sThetaNext = sin(thetaNext)

                # NOTE these normal are actually wrong, not sure it matters though
                # (the tip triangle inherits the normal from the prism strip it attaches to,
                # rather than getting its own correct normal)
                normals = (cTheta + cThetaNext)*xDir + (sTheta + sThetaNext)*yDir

                # First base point
                tipPos[3*rotInc:3*nLines:3*kPrism,:] = vecEnds + (cTheta*xDir + sTheta*yDir) * tipDist
                tipNorm[3*rotInc:3*nLines:3*kPrism,:] = normals
                tipColor[3*rotInc:3*nLines:3*kPrism,:] = self.vectorColor

                # Second base point
                tipPos[3*rotInc+1:3*nLines:3*kPrism,:] = vecEnds + (cThetaNext*xDir + sThetaNext*yDir) * tipDist
                tipNorm[3*rotInc+1:3*nLines:3*kPrism,:] = normals
                tipColor[3*rotInc+1:3*nLines:3*kPrism,:] = self.vectorColor

                # Second base point
                tipPos[3*rotInc+2:3*nLines:3*kPrism,:] = vecEnds + vecs * tipHeight
                tipNorm[3*rotInc+2:3*nLines:3*kPrism,:] = normals
                tipColor[3*rotInc+2:3*nLines:3*kPrism,:] = self.vectorColor


        # Expand the lines we just constructed in to strips which will form the
        # walls of the vector
        # (see http://mathworld.wolfram.com/RegularPolygon.html)
        sideLen = 2*vectorRadius*tan(pi / kPrism) # formula for side length of a regular polygon
        stripPos, stripNorm, stripColor = self.expandLinesToStrips(linePos, lineNorm, lineColor, sideLen, relativeWidths=False)

        # Append the tips
        if drawTips:
            stripPos = np.vstack((stripPos, tipPos))
            stripNorm = np.vstack((stripNorm, tipNorm))
            stripColor = np.vstack((stripColor, tipColor))

        return stripPos, stripNorm, stripColor

    def computeScale(self):
        """
        Compute scale factors and translations so that we get a good default view
        no matter what the scale or translation of the vertex data is
        """

        # Find bounds
        bboxMax = np.array([-float('inf'),-float('inf'),-float('inf')])
        bboxMin = np.array([float('inf'),float('inf'),float('inf')])

        for v in self.mesh.verts:
            for i in range(3):
                bboxMax[i] = max((bboxMax[i], v.position[i]))
                bboxMin[i] = min((bboxMin[i], v.position[i]))

        # Center of the mesh
        bboxCent = 0.5 * (bboxMax + bboxMin)
        self.dataCenter = bboxCent

        # Scale factor
        size = max((bboxMax - bboxMin))
        self.scaleFactor = size


        # Compute the median edge length
        lengths = sorted([norm(e.anyHalfEdge.vector) for e in self.mesh.edges])
        self.medianEdgeLength = lengths[(len(lengths)*9)/10]



    def redraw(self):
        """The almighty draw function"""

        # Clear the disply buffer
        glClearColor(*(np.append(self.colors['dark_grey'],[0.0])).astype(np.float32))
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set the camera view data
        for p in self.meshPrograms: p.setUniform('viewMatrix', self.camera.viewMat())
        for p in self.meshPickPrograms: p.setUniform('viewMatrix', self.camera.viewMat())
        for p in self.meshPrograms: p.setUniform('projMatrix', self.camera.projMat())
        for p in self.meshPickPrograms: p.setUniform('projMatrix', self.camera.projMat())

        # Set a data translation
        for p in self.meshPrograms: p.setUniform('dataCenter', self.dataCenter)
        for p in self.meshPickPrograms: p.setUniform('dataCenter', self.dataCenter)

        # Set color and transparency uniforms

        # The edges of the mesh should be dark if the surface is being drawn,
        # but light otherwise. If the current color setting is wrong, update
        # the setting and fill the buffers with new data.
        if self.drawEdges and self.drawShape and np.all(self.edgeColor == self.edgeColorLight):
            self.edgeColor = self.edgeColorDark
            self.generateEdgeData()
            self.generateVertexData()
        if self.drawEdges and not self.drawShape and np.all(self.edgeColor == self.edgeColorDark):
            self.edgeColor = self.edgeColorLight
            self.generateEdgeData()
            self.generateVertexData()

        self.shapeProg.setUniform('alpha', self.shapeAlpha)
        self.edgeProg.setUniform('alpha', self.edgeAlpha)

        # Set up camera and light position in world space, for the shaders that
        # want it
        cameraPos = self.camera.getPos()

        # Light is above the camera
        lightLoc = 300*cameraPos # TODO this 300 constant is _probably_ not an ideal solution...
        for p in self.meshPrograms: p.setUniform('eyeLoc', cameraPos)
        for p in self.meshPrograms: p.setUniform('lightLoc', lightLoc)

        # Set a depth offset to prevent z-fighting while drawing edges on top
        # of the mesh
        self.shapeProg.setUniform('depthOffset', 0.0)
        self.shapePickProg.setUniform('depthOffset', 0.0)
        self.edgeProg.setUniform('depthOffset', 0.0001 * self.scaleFactor)
        self.edgePickProg.setUniform('depthOffset', 0.0001 * self.scaleFactor)
        self.vertProg.setUniform('depthOffset', 0.0005 * self.scaleFactor)
        self.vertPickProg.setUniform('depthOffset', 0.0005 * self.scaleFactor)
        if self.vectorProg is not None:
            self.vectorProg.setUniform('depthOffset', 0.0003 * self.scaleFactor)


        # Draw the mesh
        if self.drawEdges:
            self.edgeProg.draw()
        if self.drawShape:
            self.shapeProg.draw()
        if self.vectorAttrName is not None and self.drawVectors:
            self.vectorProg.draw()
        if self.drawVertices:
            self.vertProg.draw()

        ## Uncomment to draw the pick buffer for debugging
        # glClearColor(*(np.append(self.colors['black'],[0.0])).astype(np.float32))
        # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # if self.drawEdges:
        #     self.edgePickProg.draw()
        # if self.drawShape:
        #     self.shapePickProg.draw()
        # if self.drawVertices:
        #     self.vertPickProg.draw()

        glutSwapBuffers()

    def pick(self, x, y):
        """
        Pick at the specified coordinates and call the appropriate callback if
        the pick'd location corresponds to a face/edge/vertex. This uses an
        openGL render pass which renders IDs a pixel colors, followed by reading
        the pixel value.
        """

        # Draw to the pick buffer
        glClearColor(*(np.append(self.colors['black'],[0.0])).astype(np.float32))
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.drawShape:
            self.shapePickProg.draw()
        if self.drawEdges:
            self.edgePickProg.draw()
        if self.drawVertices or self.drawEdges:
            self.vertPickProg.draw()
        glFinish() # make sure everything was actually drawn

        # OpenGL uses a different screen coordinate system than GLUT here
        bufferX = x
        bufferY = self.windowHeight - y - 1

        # Read the value of the pixel at the pick location and look up the object
        res = glReadPixelsf(bufferX, bufferY, 1, 1, GL_RGB)[0,0]
        pickObject = self.pickResult(res)

        if pickObject is None:
            return

        ## Take action on the pick
        print("\nPick: " + str(pickObject))

        # Call the appropriate callback
        if type(pickObject) is Vertex and self.pickVertexCallback is not None:
            self.pickVertexCallback(pickObject)
        elif type(pickObject) is Edge and self.pickEdgeCallback is not None:
            self.pickEdgeCallback(pickObject)
        elif type(pickObject) is Face and self.pickFaceCallback is not None:
            self.pickFaceCallback(pickObject)


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

        glLineWidth(1.0)

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
        self.generateAllMeshValues()

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

        # Check for a pick (click without drag using left button)
        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN:
                self.lastClickPos = (x,y)
            elif state == GLUT_UP and self.lastClickPos == (x,y):
                self.pick(x,y)

        glutPostRedisplay()

    def keyfunc(self, key, x, y):
        """Keyboard callback"""
        if key == chr(27) or key == 'q': # escape key
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
        print("esc/q ----  Exit the viewer")
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
            for key in sorted(self.userKeyCallbacks):
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

        glShaderSource(vertShader, readFile(vertShaderFile))
        glShaderSource(fragShader, readFile(fragShaderFile))

        # Compile the vertex shader
        glCompileShader(vertShader);
        result = glGetShaderiv(vertShader, GL_COMPILE_STATUS)
        if not(result):
            raise RuntimeError(glGetShaderInfoLog(vertShader))

        # Compile the fragment shader
        glCompileShader(fragShader);
        result = glGetShaderiv(fragShader, GL_COMPILE_STATUS)
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
        glBindVertexArray(self.vaoHandle['meshVAO']) # TODO

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
