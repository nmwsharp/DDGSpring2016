# A class encapsulating the basic functionality commonly used for an OpenGL view

# System imports
import numpy as np
from math import pi, sin, cos, tan

# OpenGL imports
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# Use euclid for rotations
import euclid as eu

# Local imports
from Utilities import normalize

class Camera(object):

    def __init__(self, windowWidth, windowHeight):


        # Window variables
        self.updateDimensions(windowWidth, windowHeight)

        # View variables
        self.viewTarget = np.array((0.0,0.0,0.0))
        self.cameraDir = eu.Vector3(0,0,1).normalize()
        self.upDir = eu.Vector3(0,1,0).normalize()
        self.cameraTrans = np.array((0.0,0.0,0.0))
        self.fov = 65 # degrees
        self.nearClipRat = 0.01
        self.farClipRat = 100
        self.zoomDist = 3.0          # Camera distance from target
        self.minZoom = 0.01           # Cannot zoom closer than this

        # Mouse drag variables
        self.mouseDragging = False
        self.mouseDragState = None
        self.lastPos = None
        self.shiftHeld = False


    ### OpenGL model and projection
    def projMat(self):

        # Build a projection matrix
        fVal = 1.0 / tan(self.fov * (pi / 360.0))
        farClip = self.farClipRat * self.zoomDist
        nearClip = self.nearClipRat * self.zoomDist

        projMat = np.eye(4)
        projMat[0,0] = fVal / self.aspectRatio
        projMat[1,1] = fVal
        projMat[2,2] = (farClip + nearClip) / (nearClip - farClip)
        projMat[2,3] = (2.0 * farClip * nearClip) / (nearClip - farClip)
        projMat[3,2] = -1.0
        projMat[3,3] = 0.0

        return projMat.astype(np.float32)


    def viewMat(self):

        # First make sure we know all relevant positions and directions
        E = self.viewTarget + np.array(self.cameraDir) * self.zoomDist
        C = self.viewTarget
        U = np.array(self.upDir)

        # Rotation matrix to put the camera in the right direction
        rotMat = np.zeros((4,4))
        rotMat[0,0:3] = np.cross(self.upDir, self.cameraDir)
        rotMat[1,0:3] = self.upDir
        rotMat[2,0:3] = self.cameraDir
        rotMat[3,3] = 1.0

        # Translation matrix, which mostly just pushes it out to the -z Axis
        # where the camera looks
        # If we want to make the camera translate, should probably add it here
        transMat = np.eye(4)
        transMat[0,3] = 0.0 + self.cameraTrans[0]
        transMat[1,3] = 0.0 + self.cameraTrans[1]
        transMat[2,3] = -self.zoomDist + self.cameraTrans[2]
        transMat[3,3] = 1.0

        viewMat = np.dot(transMat, rotMat)

        return viewMat.astype(np.float32)

    def getPos(self):
        return (self.viewTarget + np.array(self.cameraDir) * self.zoomDist).astype(np.float32)

    def getUp(self):
        return np.array(self.upDir).astype(np.float32)

    def updateDimensions(self, windowWidth, windowHeight):
        self.aspectRatio = float(windowWidth) / windowHeight
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        glViewport(0, 0, windowWidth, windowHeight);

    ### Mouse and keyboard callbacks to reposition

    def processMouse(self, button, state, x, y, shiftHeld):
        # print("ProcessMouse   button = " + str(button) + "   state = " + str(state))

        # Scroll wheel for zoom
        if button == 3 or button == 4:
            if state == GLUT_UP:
                return
            elif button == 3:
                self.zoomIn()
            elif button == 4:
                self.zoomOut()

        # Left click activates dragging
        elif button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN:
                self.mouseDragging = True
                self.lastPos = (x,y)

                # Holding shift gives translation instead of rotation
                if(shiftHeld):
                    self.mouseDragState = 'translate'
                else:
                    self.mouseDragState = 'rotate'

            else:   # (state == GLUT_UP)
                self.mouseDragging = False
                self.lastPos = None
                self.mouseDragState = None

    def processMotion(self, x, y):

        if self.mouseDragging:

            # The vector representing this drag, scaled so the dimensions
            # of the window correspond to 1.0
            delX = (float(x) - self.lastPos[0]) / self.windowWidth
            delY = (float(y) - self.lastPos[1]) / self.windowWidth

            if(self.mouseDragState == 'rotate'):

                # Scale the rotations relative to the screen size
                delTheta = -2*pi * delX
                delPhi = -pi * delY

                # Rotate by theta around 'up' (rotating up is unneeded since it
                # would do nothing)
                oldCamDir = self.cameraDir.copy();
                self.cameraDir = self.cameraDir.rotate_around(self.upDir, delTheta)

                # # Rotate by phi around 'left'
                leftDir = self.upDir.cross(oldCamDir)
                self.cameraDir = self.cameraDir.rotate_around(leftDir, delPhi)
                self.upDir = self.upDir.rotate_around(leftDir, delPhi)

            elif(self.mouseDragState == 'translate'):

                moveDist = self.zoomDist * 5.0
                self.cameraTrans[0] += delX*moveDist
                self.cameraTrans[1] -= delY*moveDist

            self.lastPos = (x,y)

    def processKey(self, key, x, y):
        # print("ProcessKey   key = " + str(key))
        moveDist = self.zoomDist * 0.02

        # Use 'r' and 'f' to zoom (OSX doesn't give mouse scroll events)
        if key == 'r':
            self.zoomIn()
        elif key == 'f':
            self.zoomOut()

        # Use 'wasd' to translate view window
        elif key == 'd':
            self.cameraTrans[0] += moveDist
        elif key == 'a':
            self.cameraTrans[0] -= moveDist
        elif key == 'w':
            self.cameraTrans[1] += moveDist
        elif key == 's':
            self.cameraTrans[1] -= moveDist



    def zoomIn(self):
        self.zoomDist = max(self.minZoom, self.zoomDist * 0.9)

    def zoomOut(self):
        self.zoomDist = self.zoomDist * 1.1
