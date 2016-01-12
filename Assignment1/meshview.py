# Basic application to load a mesh from file and view it in a window

# Python imports
import sys, os

## Imports from this project
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from InputOutput import *
from MeshDisplay import MeshDisplay
from HalfEdgeMesh import *

def main():

    # Get the path for the mesh to load, either from the program argument if
    # one was given, or a dialog otherwise
    if(len(sys.argv) > 1):
        filename = sys.argv[1]
    else:
        print("ERROR: No file name specified. Proper syntax is 'python meshview.py path/to/your/mesh.obj'.")
        exit()

    # Read in the mesh
    mesh = HalfEdgeMesh(readMesh(filename))

    # Toss up a viewer window
    winName = 'meshview -- ' + os.path.basename(filename)
    meshDisplay = MeshDisplay(windowTitle=winName)
    meshDisplay.setMesh(mesh)
    meshDisplay.startMainLoop()


if __name__ == "__main__": main()
