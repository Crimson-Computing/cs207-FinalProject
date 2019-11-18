# AutoDiffCC - Milestone 2 Document
CS207 Final Project, Group 22
Alex Spiride, Maja Garbulinska, Matthew Finney, Zhiying Xu

## Introduction
With the evolution of science from empirical to theoretical and now computational approaches, differentiation plays a critical role in a wide range of scientific and industrial applications of computer science. However, the precise computation of symbolic derivatives is computationally expensive, and not even possible in all situations, whereas the finite differencing method is not always accurate or stable. Automatic Differentiation, however, provides a computationally efficient way to calculate derivatives, particularly of complex functions, for applications where accuracy and performance at scale are important.

## Background

Our software will make it easier for the user to compute derivatives using automatic differentiation, which provides efficiency and numerical stability as opposed to other methods such as finite differences. A computer can perform elementary operations quickly. If we apply the chain rule to these elementary operations we can compute derivatives of functions efficiently and with working precision.

To work, the package makes use of the following concepts:

### Elementary functions

An elementary function is a function that is a finite combination of constant functions, field operations, algebraic, exponential and logarithmic functions and their inverses. The derivative of any elementary function is itself elementary. You can read more about it [here](http://mathworld.wolfram.com/ElementaryFunction.html) .  

Field operations: 
- addition 
- subtraction 
- multiplication
- division 

See the examples of elementary functions below: 


<img width="84" alt="Screen Shot 2019-10-27 at 16 57 28" src="https://user-images.githubusercontent.com/43005886/67641740-e6c8f780-f8da-11e9-9a9f-a731639fe798.png">


Click [here](https://en.wikipedia.org/wiki/Riemann_zeta_function) to see an example of a non-elementary function. 


### Forward mode

The forward mode uses the chain rule described below to compute derivatives of nested functions. The chain rule is applied to elementary operations step by step starting with the most inner operation. We then obtain a derivative trace. At every step derivative and the value of the expression is evaluated. 

To work, the package makes use of the following concepts:

### The chain rule

According to the chain rule the derivative of f(g(x)) is f'(g(x))⋅g'(x).

See the following example: 

<img width="252" alt="Screen Shot 2019-10-27 at 16 35 05" src="https://user-images.githubusercontent.com/43005886/67641296-c3507d80-f8d7-11e9-8a03-2e80da87e26b.png">

If we have a function such that z = f (x(t), y(t)) then the derivative is f(x(t), y(t)) * f'(x) + f(x(t), y(t)) * f'(y).

See the following example: 


<img width="304" alt="Screen Shot 2019-10-27 at 16 38 26" src="https://user-images.githubusercontent.com/43005886/67641393-41148900-f8d8-11e9-90f3-45ca94f2a37c.png">


### Computational graphs

Computational graph makes it easier to think about the mathematical operations that our package performs.  

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

### More information

If you would like to know more about this topic, you should have a look at [this book](https://arxiv.org/pdf/1411.0583.pdf).

## How to use the package

### Installation
AutoDiffCC supports package installation via `pip`. Users can install the package in the command line with the following command.

```buildoutcfg
pip install autodiffcc
```

### Automatic differentiation
In order to use the package, the user will instantiate an object of the automatic-differentiation class, which includes methods for different functions. The new object will keep in its internal state the value and derivative. 

A simple example using overloaded operators is described below. A user wants to evaluate ``f = x * x`` at ``x = 2``. They first instantiate an AD object  ``x`` with ``x = ad.AD(2.0, 1.0)``, where ``2`` is the value and ``1`` is the derivative. Then the user defines ``f = x * x`` and prints its value and derivative.

``` AD Overloaded Operators Example
# Import the autodiffcc package
>>> import autodiffcc as ad 

# Overload basic arithmetic operations
>>> x = ad.AD(2.0, 1.0) 
>>> f = x * x
>>> print(f.val, f.der)
4.0 4.0
```

A more complex example using custom math methods is described below. A user wants to evaluate ``g = exp(x)`` at ``x = 3``. They first instantiate an AD object  ``x`` with with ``x = ad.AD(val = 3, der = 1)``, where ``3`` is the value and ``1`` is the derivative. Then the user defines ``g = ad.exp(x)`` and prints its value and derivative.

``` AD Custom Math Methods Example
import autodiffcc as ad

# Find the derivative of e^x
x = ad.AD(val = 3, der = 1)
g = ad.exp(x)
print(g.val, g.der)
20.085536923187668 20.085536923187668
```

### Distribution and packaging
The AutoDiffCC library using the Python Package Index (PyPI). It is distributed in the pythonic formats of an sdist and as a wheel to facilitate installation via Python's package installer `pip`. The user will install the package with `pip` and them import it in Python. Importing the package via `pip` will also ensure that the user installs required dependencies.

Given that our aim is to deliver software that another developer can use for automatic differentiation in their own applications, we are not planning to deliver a standalone application. As such we are not planning to use a distribution framework beyond Python's native packaging.

## Software organization
### Directory structure
```
	cs207-FinalProject/
			README.md
			LICENSE
			autodiffcc/
                __init__.py
				ADmath.py
				autodiffcc.py
				rootfindercc.py [Placeholder]
			docs/
				milestone1.md
				milestone2.md
			tests/
				test_autodiff.py
                test_admath.py
			...
```

### Modules
The `autodiffcc` package will have four modules:

|Module|Basic Functionality|
|-|-|
|autodiffcc| This is the main module, which will contain the `AD` class and methods for operator overloading (e.g., add, mult, etc.).|
|ADmath| This module contains elementary functions, (e.g. sin, cos, sqrt, log, exp,etc.) for the `AD` class. |
|RootFinder| This module is a placeholder for the advanced feature.|

### Test suite
Our test suite is in the directory `cs207-FinalProject/tests`. We  use TravisCI to perform continuous integration, running these tests with each build pushed to GitHub. We use CodeCov to ensure that our software implementation has sufficient code covered by our test suite. Badges indicating test compliance and code coverage are included in `README.md`.

## Implementation
Our `AD` class is used to create an AD object, including custom methods, that work on scalars and numpy arrays. The AD class will then be used in our extension class, which will be an object of its own (RootFinder). Each of the math methods will be callable from an imported library of math functions.

The AD class has several methods, including an init, add, subtract, multiply, division, positive, negative, and comparison (<, ≥) dunder methods. AD objects will have a value and a derivative. The math functions will include functions such as log, exp, tan, power, trigonometric functions, and more. To deal with elementary functions like sin, sqrt, log, and exp and all the others, we will write methods to extend general implementations (e.g. numpy) updating the derivative at each step. Finally, the AD class will have attributes to get the derivative and value of the object.

We want to make our class compatible with numpy arrays, so we will need to use NumPy, as well as math. For testing, we will need doctest and pytest, and we might use scipy for the RootFinder. 


## Future features

### Continuing development of autodiffcc
Our forward mode implementation is mostly complete, however, we still need to implement support for derivatives of vector functions. Vectorizing the existing modules could be a technical challenge, but we experimenting with using array data structures in the current class and function definitions, and we foresee this approach being promising. Given that this current distribution of our package uses arrays, we don't foresee adding new classes, modules, or data structures to implement full support for derivatives of vector functions. 

### Advanced feature: RootFinder
We will develop a RootFinder for our advanced feature. Our RootFinder will implement Newton's method to approximate the roots of a real-valued function within a given tolerance. This will be in its own module, RootFinder. At this time, we don't foresee any additional modules, or data structures, but we may implement a Root class that can support real and possibly even complex roots.

We select Newton's method for our RootFinder because it leverages differentiation and generalizes to high-dimensional problems and complex functions.  Our RootFinder, provided a function, will start by using `autodiffcc` to find the derivative of the function at an initial guess for a root. It will iterate through successively better approximations of the root along the function, taking the derivative with `autodiffcc` at each step, until it finds the root(s) within a given tolerance. An example of the potential use of the RootFinder is shown below. The user interaction is subject to change pending final implementation. 

``` RootFinder example
import autodiffcc as ad

# Find the root of x^6 - 6x^5 + 5x^4 - 4
x = ad.AD(val = 1, der = 1)
h = x ** 6 - 6 * (x ** 5) + 5 * (x ** 4) - 4
roots = ad.roots(h)
print(roots)
-0.78842 5.0016
```
