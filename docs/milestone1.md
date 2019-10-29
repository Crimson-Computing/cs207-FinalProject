# Milestone 1 Document
CS207 Final Project, Group 22
Alex Spiride, Maja Garbulinska, Matthew Finney, Zhiying Xu

## Introduction
With the evolution of science from empirical to theoretical and now computational approaches, differentiation plays a critical role in a wide range of scientific and industrial applications of computer science. However, the precise computation of symbolic derivatives is computationally expensive, and not even possible in all situations, whereas the finite differencing method is not always accurate or stable. Automatic Differentiation, however, provides a computationally efficient way to calculate derivatives, particularly of complex functions, for applications where accuracy and performance at scale are important.

## Background


Our software will make it easier for the user to compute derivatives using automatic differentiation. Autodifferentiation provides efficiency and numerical stability as opposed to other methods such as finite differences. A computer can perform elementary operations quickly. If we apply the chain rule to these elementary operations we can compute derivatives of functions with working precision.

To work, the package makes use of the following concepts:

### Forward Mode

The forward mode uses the chain rule described below to compute derivatives of nested functions. The chain rule is applied to elementary operations step by step starting with the most inner operation. 
Our software will make it easier for the user to compute derivatives using automatic differentiation. Our software will be able to compute the values of a function as well as its derivatives at specified points. Autodifferentiation provides efficiency and numerical stability as opposed to other methods such as finate differences. To work, the functions make use of the following concepts:

### The Chain Rule

According to the chain rule the derivative of f(g(x)) is f'(g(x))⋅g'(x).

See the following example: 

<img width="252" alt="Screen Shot 2019-10-27 at 16 35 05" src="https://user-images.githubusercontent.com/43005886/67641296-c3507d80-f8d7-11e9-8a03-2e80da87e26b.png">

If we have a function such that z = f (x(t), y(t)) then the derivative is f(x(t), y(t)) * f'(x) + f(x(t), y(t)) * f'(y).

See the following example: 


<img width="304" alt="Screen Shot 2019-10-27 at 16 38 26" src="https://user-images.githubusercontent.com/43005886/67641393-41148900-f8d8-11e9-90f3-45ca94f2a37c.png">

### The Jacobian Matrix 

The Jacobian Matrix is a matrix of all first order partial derivatives of an equation. Let's say we are given a complicated set of non-linear equations. If we graphed it, and zoomed closely enough, the function will be locally linear. The Jacobian Matrix can tell us more about how the function looks like locally. For functions with only 1D output the Jacobian is only the gradient. 



### Computational Graphs

Computational graph make its easier to think about mathematical operations. 


Consider the following function: 

<img width="160" alt="Screen Shot 2019-10-27 at 16 17 54" src="https://user-images.githubusercontent.com/43005886/67641073-6653c800-f8d5-11e9-9aca-fca591ff2473.png">

It consists of four operations: 
- cos()
- sin() 
- multiplication 
- addition 

We can therefore construct the following graph to visualize the operations. 

<img width="759" alt="Screen Shot 2019-10-27 at 16 34 09" src="https://user-images.githubusercontent.com/43005886/67641280-a1ef9180-f8d7-11e9-83d8-029c4e4c3b92.png">

Now, let's say we would like to evaluate the function a some specific values of x and y we just have to apply the applicable operations following the graph. 


### Elementry Functions

An elementary function is a function that is a finite combination of constant functions, field operations, algebraic, exponential and logarithmic functions and their inverses. The derivative of any elementary function is itself elementary. You can read more about it [here](http://mathworld.wolfram.com/ElementaryFunction.html) .  

Field operations: 
- addition 
- substruction 
- multiplication
- devision 

See the examples of elementary functions below: 


<img width="84" alt="Screen Shot 2019-10-27 at 16 57 28" src="https://user-images.githubusercontent.com/43005886/67641740-e6c8f780-f8da-11e9-9a9f-a731639fe798.png">


Click [here](https://en.wikipedia.org/wiki/Riemann_zeta_function) to see an example of a non-elementary function. 

### More information

If you would like to know more about this topic, you should have a look [this book](https://arxiv.org/pdf/1411.0583.pdf).

## How to use the package
The user will install the package with pip and them import it in Python. As we include some dependencies in our module(e.g. Numpy), importing the AD package will also implicity import those dependencies. 

In order to use the package, the user will instantiate an object of the automatic-differentiation class, which includes methods for different functions. The new object will keep in its internal state the value and derivative. 

A simple example is listed as below. The user could install with command ``pip install AutoDiff`` and import the package. If the user wants to evaluate ``f = x * x`` at ``x = 2``, he / she could first instantiate an AD object for ``x`` with ``x = AutoDiff.AD(2.0, 1.0)``, where ``2`` is the value and ``1`` is the derrivative. Then the user could input the function ``f = x * x`` and print its value and derrivative. The user could also instantiate dual value ``2.0 + 1.0e`` by ``x = AutoDiff.Dual(2.0, 1.0)``. As for the output, the real component ``4.0`` is the function value and the dual component ``4.0e`` is the derivative.
```
# Install at command line
$ pip install AutoDiff

# Python
$ Python
>>> import Autodiff 
>>> import numpy as np

# Example of normal numbers
# Expect value of 4.0, derivative of 4.0
>>> x = AutoDiff.AD(2.0, 1.0) 
>>> f = x * x
>>> print(f.val, f.der)
4.0 4.0

# Example of dual numbers
# Expect value of 4.0, derivative of 4.0
>>> x = AutoDiff.Dual(2.0, 1.0)
>>> f = x * x
>>> print(f.val)
4.0 + 4.0e
```

## Software organization
### Proposed Directory Structure
```
	cs207-FinalProject/
			README.md
			LICENSE
			autodiff/
				dual.py
				math.py
				autodiff.py
				advanced_feature.py [Placeholder]
			docs/
				milestone1.md
			tests/
				test_autodiff.py
			...
```

### Modules
The `autodiff` package will have four modules:

|Module|Basic Functionality|
|-|-|
|autodiff| This is the main module, which will contain the autodiff class and methods for operator overloading (e.g., add, mult, etc.).|
|dual| This module will contain the dual numbers class and methods.|
|math| This module will contain elementary functions, (e.g. sin, cos, sqrt, log, exp,etc.) for the autodiff class. |
|advanced_feature| This module is a placeholder for the advanced feature. Once we've finalized the decision on the advancved feature to build, we will determine whether this needs to be its own module.|

### Test Suite
Our test suite will be in the directory `cs207-FinalProject/tests`. We will use TravisCI to perform continuous integration, running these tests with each build pushed to GitHub. We will also use CodeCov to ensure that our software implementation has sufficient code covered by our test suite. Badges indicating test compliance and code coverage are included in `README.md`.


### Distribution and Packaging
We will distribute our library using the Python Package Index (PyPI). We will distribute the software in the pythonic formats of an sdist and as a wheel to facilitate installation via Python's package installer `pip`. Given that our aim is to deliver software that another developer can use for automatic differentiation in their own applications, we are not planning to deliver a standalone application. As such we are not planning to use a distribution framework beyond Python's native packaging.

## Implementation
Our autodiff class will be used to create an AutoDiff object, including custom methods, that will be able to work on scalars and numpy arrays. The output of this object will be another type of object: a Dual object, which represents a dual number. The AutoDiff class will then be used in our extension class, which will be an object of its own (RootFinder). Each of the math methods will be callable from an imported library of math functions.

The Dual object will have several methods, including an init, add, subtract, multiply, division, positive, negative, and comparison (<, ≥) dunder methods. Dual objects will have a value and a derivative. The math functions will include functions such as log, exp, tan, power, trigonometric functions, and more. To deal with elementary functions like sin, sqrt, log, and exp and all the others, we will write methods to extend general implementations (e.g. numpy) updating the derivative at each step. Finally, the AutoDiff class will have attributes to get the derivative and value of the object.

We want to make our class compatible with numpy arrays, so we will need to use NumPy, as well as math. For testing, we will need doctest and pytest, and we might use scipy for the rootfinder. 
