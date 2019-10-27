# Milestone 1 Document
CS207 Final Project, Group 22
Alex Spiride, Maja Garbulinska, Matthew Finney, Zhiying Xu

## Introduction

## Background

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
