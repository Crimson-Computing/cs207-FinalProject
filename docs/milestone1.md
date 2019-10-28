# Milestone 1 Document
CS207 Final Project, Group 22
Alex Spiride, Maja Garbulinska, Matthew Finney, Zhiying Xu

## Introduction

## Background

Our software will make it easier for the user to compute derivatives using automatic differentiation. Our software will be able to compute the values of a function as well as its derivatives at specified points. Autodifferentiation provides efficiency and numerical stability as opposed to other methods such as finate differences. To work, the functions make use of the following concepts:

### The Chain Rule

According to the chain rule the derivative of f(g(x)) is f'(g(x))⋅g'(x).

See the following example: 

<img width="278" alt="Screen Shot 2019-10-27 at 15 43 09" src="https://user-images.githubusercontent.com/43005886/67640363-a5cbe580-f8d0-11e9-907f-bea69360198e.png">


### Computational Grapth
### Elementry Functions

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
# Expect value of , derivative of 
>>> x = AutoDiff.Dual(2.0, 1.0)
>>> f = x * x
>>> print(f.val)
4.0 + 4.0e
```

## Software organization

## Implementation
Our autodiff class will be used to create an AutoDiff object, including custom methods, that will be able to work on scalars and numpy arrays. The output of this object will be another type of object: a Dual object, which represents a dual number. The AutoDiff class will then be used in our extension class, which will be an object of its own (RootFinder). Each of the math methods will be callable from an imported library of math functions.

The Dual object will have several methods, including an init, add, subtract, multiply, division, positive, negative, and comparison (<, ≥) dunder methods. Dual objects will have a value and a derivative. The math functions will include functions such as log, exp, tan, power, trigonometric functions, and more. To deal with elementary functions like sin, sqrt, log, and exp and all the others, we will write methods to extend general implementations (e.g. numpy) updating the derivative at each step. Finally, the AutoDiff class will have attributes to get the derivative and value of the object.

We want to make our class compatible with numpy arrays, so we will need to use NumPy, as well as math. For testing, we will need doctest and pytest, and we might use scipy for the rootfinder. 
