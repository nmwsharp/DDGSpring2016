# DDGSpring2016

Welcome to the code repository for 15-869 Discrete Differential Geometry at CMU in Spring 2016.

* Course website: http://www.brickisland.net/DDGSpring2016/
* Instructor: [Keenan Crane](http://www.keenan.is/here)
* TA: [Nick Sharp](http://www.nmwsharp.com) (direct questions about this code to me, my email is on the course website)

##Getting started (OSX and Linux)

1. Make sure you have an installation of Python 2.7, including the `pip` package manager.
  * **Mac** On recent OSX versions, Python is installed by default, and you may already have `pip` as well. Executing `sudo easy_install pip` in a terminal should install `pip` if not.  Note that you may have to run `python2.7` from the command line, as just plain `python` may run a different version.
  * **Linux** Get Python and `pip` from your package manager, if you don't already have them. On Ubuntu, try `sudo apt-get install python-pip`. To satisfy some other requirements below as well, try `sudo apt-get install python-numpy python-scipy python-matplotlib`.

2. Install dependencies using `pip`. Execute these commands at the terminal (you may need to preface them with `sudo`).
  * `pip install numpy` (Numerical computing library)
  * `pip install scipy` (more scientific/numerical computing things)
  * `pip install matplotlib` (Plotting utilities)
  * `pip install euclid` (concise 3D vector rotations)
  * `pip install plyfile` (read & write the `.ply` file format)
     **Note:** On recent OSX machines, this may fail because System Integrity Protection makes it hard to upgrade numpy. In that case, try `pip install --ignore-installed plyfile`. 
  * `pip install pyopengl` (bindings for 3D graphics & windowing)

3. Download the code in this repository. Either use the _Download Zip_ button on the [repository page](https://github.com/nmwsharp/DDGSpring2016), or (if you are familiar with git) clone the repository.

4. To verify the code works, open a terminal in the Assignment0 directory, and execute `python testview.py ../meshes/bunny.obj` (on some systems, you may have to use `python2.7` rather than just `python`). You should get a window with bunny in it! Try clicking and dragging around in the window. For now the bunny will just look like a silhouette, after completing a future assignment it will look much nicer.

##Getting started (Windows)
Unfortunately, like most programming tasks, this will be a little trickier in Windows.

1. Anaconda is Python distribution which automatically includes many useful components for scientific computing (and also sets things up quite nicely on Windows). Download and install the Python 2.7 version of Anaconda from https://www.continuum.io/downloads (keeping the default configuation options).

2. We use a few additional packages beyond those included with Anaconda, get the first two by running the following commands in a command prompt
  * `pip install euclid` (concise 3D vector rotations)
  * `pip install plyfile` (read & write the `.ply` file format)

3. The third package we need is called `pyopengl`, which contains bindings for 3D graphics and windowing. Unfortunately, there is an error involving the 64-bit version of `pyopengl` in the `pip` repository. If you are running 32-bit Windows (unlikely), you can just execute `pip install pyopengl` as above.

  If you are running 64-bit Windows (most likely), you will need to download a proper version of the package here http://www.lfd.uci.edu/~gohlke/pythonlibs/, you want the file called `PyOpenGL-3.1.1b1-cp27-none-win_amd64.whl`. Then navigate to the file in the command prompt and execute `pip install PyOpenGL-3.1.1b1-cp27-none-win_amd64.whl`. This will install the package directly with the fixed version you just downloaded.

4. Download the code in this repository. Either use the _Download Zip_ button on the [repository page](https://github.com/nmwsharp/DDGSpring2016), or (if you are familiar with git) clone the repository.

5. To verify the code works, open a command prompt in the Assignment0 directory, and execute `python testview.py ..\meshes\bunny.obj`. You should get a window with bunny in it! Try clicking and dragging around in the window. For now the bunny will just look like a silhouette, after completing a future assignment it will look much nicer.


##About Python

Clearly, the code for this class is written in Python. Pure Python code is typically an order of magnitude slower than the equivalent C/C++ code, but nonetheless it sees widespread use for numerical/scientific computing within some communities. We will be using Python for this course primarily because it very accessible to someone unfamiliar with the language, because sharing Python code across different platforms is relatively painless, and because it is free/open-source.

If you are unfamiliar with Python, don't be afraid! We will be doing everything possible to make this a painless experience. You might want to devote some time to basic tutorials on Python, but hopefully that will not be necessary. In this class you won't necessarily be creating complex programs from scratch -- just implementing algorithms.

The [wiki](https://github.com/nmwsharp/DDGSpring2016/wiki) for this repository contains (or will contain) some notes about Python as well as a series of examples demonstrating the usage of the language and this library.


##Codebase Outline

```
/DDGSpring2016
    /core
        /HalfEdgeMesh.py -- the main mesh datastructure
        /TriSoupMesh.py  -- an alternate mesh datastructure, used for input and output
        /InputOutput.py  -- read meshes from file
        /Utilities.py    -- miscellaneous helper functions
        /MeshDisplay.py  -- create viewers to visualize meshes in 3D
        /Camera.py       -- additional logic for MeshDisplay.py
        /shaders         -- used for visualization
          ...
    /Assignment1
        ...
        The files for the first assignment
    /Assignment2
        ...
        The files for the second assignment (etc)
    /meshes
        ...
        A collection of sample meshes to run your code on
```

During the course, we will be adding new /Assignment directories, as well as adding new functionality to the core libraries as it is needed.

You may notice some methods in `HalfEdgeMesh.py` that are only skeletons, the body of the method is not implemented. You will be implementing these methods as homework assignments. After each assignment is submitted, I will update this repository with reference implementations for these methods, so that you may use them on the subsequent assignments.
