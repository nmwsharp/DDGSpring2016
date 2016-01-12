# DDGSpring2016

Welcome to the code repository for 15-869 Discrete Differential Geometry at CMU in Spring 2016.

* Course website: http://brickisland.net/DDGSpring2016/
* Instructor: [Keenan Crane](keenan.is/here)
* TA: [Nick Sharp](nmwsharp.com) (direct questions about this code to me, my email is on the course website)

##Getting started

1. Make sure you have an installation of Python 2.7, including the `pip` package manager.
  * **Mac** On recent OSX versions, Python is installed by default. Executing `sudo easy_install pip` in a terminal should install `pip`.
  * **Windows** [Anaconda](https://www.continuum.io/downloads) provides a convenient installer which includes Python, `pip`, and many popular scientific packages for Python. (**Note:** it might be a little more complex than this)
  * **Linux** Get Python and `pip` from your package manager, if you don't already have them. On Ubuntu, try `sudo apt-get install python-pip`.

2. Install dependencies using `pip`. Execute these commands at the terminal or command line.
  * `pip install numpy` (Numerical computing library)
  * `pip install matplotlib` (Plotting utilities)
  * `pip install euclid` (concise 3D vector rotations)
  * `pip install plyfile` (read & write the `.ply` file format)
  * `pip install pyopengl` (bindings for 3D graphics & windowing)

3. Download the code in this repository. Either use the _Download Zip_ button on the [repository page](https://github.com/nmwsharp/DDGSpring2016), or (if you are familiar with git) clone the repository.

4. To verify the code works, open a terminal/command line in the Assignment1 directory, and execute `python meshview.py ../meshes/bunny.obj`. You should get a window with bunny in it! Try clicking and dragging around in the window. For now the bunny will just look like a silhouette, after completing Assignment 1 it will look much nicer.

##About Python

Clearly, the code for this class is written in Python. Pure Python code is typically an order of magnitude slower than the equivalent C/C++ code, but nonetheless it sees widespread use for numerical/scientific computing within some communities. We will be using Python for this course primarily because it very accessible to someone unfamiliar with the language, because sharing Python code across different platforms is relatively painless, and because it is free/open-source.

If you are unfamiliar with Python, don't be afraid! We will be doing everything possible to make this a painless experience. You might want to devote some time to basic tutorials on Python, but hopefully that will not be necessary. In this class you won't necessarily be creating complex programs from scratch -- just implementing algorithms.

I'm not going to try to teach you Python in a README, but here are a few quick pointers about how it differs from other languages you might know:

* Python is an _interpreted language_. This means programs are _interpreted_ each time you run them, they do not need to be compiled. You may notice some `.pyc` files floating around your directories, these are automatically generated/managed by Python, and you can safely ignore them.

* In Python code, unlike nearly every other modern language, **whitespace is significant** -- changing the indentation of a program changes its meaning. Colons are used to denote the beginning of a loop or if-statement, and the subsequent indented region makes up the body of the loop. Brackets are never used.

	```python    
	for i in range(10):
	  print("Hello from inside the loop")
	  if i == 3:
	    print("Iteration 3 is my favorite iteration")
	  else:
	    print("I wish this was iteration 3...")
	print("Hello from outside the loop")
	```
* Python is a _dynamically typed_ language. You never need to give your variables a type (simply `x = 3` instead of something like `int x = 3`), and you are free to later reassign a variable to a new value of a different type (`x = "cat"` is valid).

* This codebase is designed using a simple form _object-oriented_ programming, where the code is organized in to _classes_ and each class contains the variables and logic for some component of the program. For instance, there is a `Vertex` class representing a vertex in a mesh, which contains a variable for the vertex position, as well as defining methods for operating on a vertex. If you don't already know object-oriented programming, don't worry -- you shouldn't need to learn it to complete the homework assignments.

  However, you should know that the `self` keyword is crucial for writing object-oriented code in Python. When writing code inside a class, that class' variables are accessed using `self.variableName`, and similarly class methods are called using `self.methodName(parameter1, parameter2)`. When defining a class method, it must take `self` as its first parameter `def myMethod(self, parameter1):`.

* In Python, variables defined inside loops and if-statements are still valid after the end of the loop/if-statement.

  ```python
  for i in range(10):
  		x = i*i
  print(x) #(valid code)
  ```

  This will irritate C++ programmers immensely.

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

##Examples

Here are some code snippets which demonstrate the basic functionality of this codebase.

(In progress)
