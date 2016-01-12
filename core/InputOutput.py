import numpy as np
from plyfile import PlyData, PlyElement, make2d

from TriSoupMesh import TriSoupMesh

# Multipurpose mesh reader
def readMesh(filename, filetype=None, output='soup'):

    print('Reading mesh from: ' + filename)

    # TODO implement this
    if(output != 'soup'):
        raise Exception('Mesh types other than soup not yet supported')

    # Attempt to detect the filetype, if necessary
    if not filetype:
        if(filename.endswith('.ply')):
            filetype = 'ply'
        elif(filename.endswith('.obj')):
            filetype = 'obj'
        else:
            raise Exception('Could not detect filetype for input file: ' + filename)

    # Call the appropriate input function
    if filetype == 'ply':
        mesh = readMesh_PLY(filename, output)
    elif filetype == 'obj':
        mesh = readMesh_OBJ(filename, output)
    else:
        raise Exception('No reader for filetype: ' + filetype)

    if(mesh.nVerts == 0):
        raise Exception("ERROR: Loaded mesh has no vertices")

    print('   Loaded mesh with ' + "{:,}".format(mesh.nVerts) + ' vertices and ' + "{:,}".format(mesh.nTris) + ' faces')
    return mesh

# Read a .ply mesh
def readMesh_PLY(filename, output="soup"):

    if(output != 'soup'):
        raise Exception('Mesh types other than soup not yet supported')

    # Read the actual file
    # TODO This takes a long time, maybe try to replace with something faster of my own?
    plydata = PlyData.read(filename)

    # Read vertices
    # If the mesh has more than three columns of vertex data, ignore the later columns
    # (for instance, Stanford Mesh Repo meshes store intensity and confidence here)
    nVerts = plydata['vertex'].count
    verts = np.zeros((nVerts,3))
    verts[:,0] = np.array(plydata['vertex'].data['x'])
    verts[:,1] = np.array(plydata['vertex'].data['y'])
    verts[:,2] = np.array(plydata['vertex'].data['z'])


    # Read faces
    faces = make2d(plydata['face'].data['vertex_indices'])

    # Build a mesh from these vertices and faces
    mesh = TriSoupMesh(verts, faces)

    return mesh


# Read a .ply mesh
# TODO Ignores everything execpt for vertex position and face indices
# TODO Will only process triangular vertices
def readMesh_OBJ(filename, output="soup"):

    if(output != 'soup'):
        raise Exception('Mesh types other than soup not yet supported')

    verts = []
    tris = []

    # Process the file line by line
    lineInd = -1
    for line in open(filename).readlines():
        lineInd += 1
        if line[0] == '#':
            continue

        items = line.strip().split(' ')

        # Process vertex
        if items[0] == 'v':
            verts.append([float(s) for s in items[1:]])

        # Process tex-coord
        elif items[0] == 'vt':
            pass

        # Process normal vector
        elif items[0] == 'vn' : #normal vector
            pass

        # Process face indices
        elif items[0] == 'f' :
            face = items[1:]
            if len(face) != 3 :
                print("Line " + str(lineInd) + " is not a triangular face: " + line)
                continue

            # Take index before slash, which is the vertex index (if slash is given).
            # Conversion from 1-based to 0-based indexing happens at end
            tris.append([int(s.split("/")[0]) for s in face])

    # Convert to numpy arrays
    verts = np.array(verts)
    tris = np.array(tris)

    # Convert to 0-based indexing
    tris = tris - 1

    # Make a tri-soup mesh
    mesh = TriSoupMesh(verts, tris)

    return mesh
