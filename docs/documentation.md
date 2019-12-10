# AutoDiffCC Final Documentation
CS207 Final Project, Group 22
Alex Spiride, Maja Garbulinska, Matthew Finney, Zhiying Xu

## Introduction
With the evolution of science and the growing computational possibilities, differentiation plays a critical role in a wide range of scientific and industrial applications of computer science. However, the precise computation of symbolic derivatives is computationally expensive, and not even possible in all situations, whereas the finite differencing method is not always accurate or stable. Automatic Differentiation, however, provides a computationally efficient way to calculate derivatives, particularly of complex functions, for applications where accuracy and performance at scale are important.

Our main extension is a module which can be used to find a real- or vector-valued function's root. Root finding, or finding the values of a function's arguments for which the function's value is 0, useful in many practical applications including optimization tasks or solving systems of equations.

## Background

Automatic Differentiation (AD) is also known as computational differentiation, algorithmic differentiation and differentiation of algorithms. Automatic differentiation does not use symbolic expressions but rather exact formulas and floating-point values. It provides a great way to avoid approximation errors. It easier for the user to compute derivatives using automatic differentiation, which provides efficiency and numerical stability as opposed to other methods such as finite differences. 

The most important idea that AD benefits from is the chain rule and the fact that it can be implemented in a numerical program. Differentiation is applied to elementary functions step by step to get the final results. A computer can perform elementary operations quickly. If we apply the chain rule to these elementary operations we can compute derivatives of functions efficiently and with working precision.

To work, the package makes use of the following concepts:

### Elementary functions

An elementary function is a function that is a finite combination of constant functions, field operations, algebraic, exponential and logarithmic functions and their inverses. The derivative of any elementary function is itself elementary. You can read more about it [here](http://mathworld.wolfram.com/ElementaryFunction.html).  

Field operations: 
- addition 
- subtraction 
- multiplication
- division 

See the examples of elementary functions below: 


<img width="84" alt="Screen Shot 2019-10-27 at 16 57 28" src="https://user-images.githubusercontent.com/43005886/67641740-e6c8f780-f8da-11e9-9a9f-a731639fe798.png">

We are also considering the power function as an elementary function as well.


Click [here](https://en.wikipedia.org/wiki/Riemann_zeta_function) to see an example of a non-elementary function. 


### Forward mode

The forward mode uses the chain rule described below to compute derivatives of nested functions. The chain rule is applied to elementary operations step by step starting with the most inner operation changing the values for derivatives. Our final function is composed of derivatives of elementary functions which are very easy to compute. 

The forward mode requires a function, a seed vector and a vector at which the function should be evaluated.
What the forward mode is really computing, mathematically, is the product of the gradient with a seed vector chosen for the derivatives. This is called the Jacobian-vector product. 


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

### Evaluation Table

As in the computational graphs, we can represent the elementary operations in an evaluation table. 

<img width="772" alt="Screen Shot 2019-11-18 at 11 48 47" src="https://user-images.githubusercontent.com/43005886/69072311-6e48e880-09f9-11ea-9ba5-fd32f71fc6aa.png">

### More information

If you would like to know more about Automatic Differentiation, you should have a look at [this book](https://arxiv.org/pdf/1411.0583.pdf).

### Background on our root finding extension
Our main extension is a module which can be used to find a real- or vector-valued function's root. Root finding, or finding the values of a function's arguments for which the function's value is 0, useful in many practical applications including optimization tasks or solving systems of equations. For full details on root-finding and the mathematical background on the algorithms implemented, see the *Extension: Root finder* section below.



## How to use the package

### Installation
AutoDiffCC supports package installation via `pip`. Users can install the package in the command line with the following command.

``` buildoutcfg
pip install autodiffcc
```

### Automatic differentiation
In order to use the package, the user will instantiate an object of the automatic-differentiation class, which includes methods for different functions. The new object will keep in its internal state the value and derivative. 

A simple example using overloaded operators is described below. A user wants to evaluate ``f = x * x`` at ``x = 2``. They first instantiate an AD object  ``x`` with ``x = ad.AD(val=2.0, der=1.0)``, where ``2`` is the value and ``1`` is the derivative. Then the user defines ``f = x * x`` and prints its value and derivative.

``` python AD Overloaded Operators Example
# Import the autodiffcc package
>>> import autodiffcc as ad 

# Overload basic arithmetic operations
>>> x = ad.AD(val=2.0, der=1.0) 
>>> f = x * x
>>> print(f.val, f.der)
4.0 4.0
```

A more complex example using custom math methods is described below. A user wants to evaluate ``g = exp(x)`` at ``x = 3``. They first instantiate an AD object  ``x`` with with ``x = ad.AD(val = 3, der = 1)``, where ``3`` is the value and ``1`` is the derivative. Then the user defines ``g = ad.exp(x)`` and prints its value and derivative.

``` python AD Custom Math Methods Example
>>> import autodiffcc as ad

# Find the derivative of e^x
>>> x = ad.AD(val = 3, der = 1)
>>> g = ad.exp(x)
>>> print(g.val, g.der)
20.085536923187668 20.085536923187668
```

#### Differentiation closure
The `differentiate` function is a closure which takes a user defined base function as an input, and returns another function which can evaluate the base function's derivative.


For example, consider the task of repeatedly differentiating the function `f(x) = 3*x^2` at different values. The user defines their function `f(x)` in python and calls `diffentiate(f)` to get the function `dfdx` which can be used to differentiate `f` at various values.

``` python 
>>> import autodiffcc as ad

# Find the numeric derivative of 3*x^2
>>> def f(x):
>>>     return 3*(x**2)

>>> dfdx = differentiate(f)
>>> print(dfdx(x=5))
[30.]

>>> print(dfdx(x=[1,1,2,3,5,8]))
[ 6.  6. 12. 18. 30. 48.]
```

### Root finding
Our Root Finder implementation requires that the user first define the function for which they would like to find the root, and an `interval` in which to look or `start_values` that are arbitrarily close to the real root.


The user then passes the `function`, `method` and `interval` or `start_values` arguments to the `find_root` function. The user may also provide the optional `max_iter` and `threshold` arguments. `max_iter` specifies the maximum number of iterations the algorithm should attempt to find a converging solution, and `threshold` sets the minimum threshold to declare convergence, such that lower thresholds return finer approximations.

Note that the user-defined functions do not need to explicitly use the `AD` basic or comparison operators. However, for elemental and trigonometric functions like `sin` or `log` the user would need to define their function with the `ADmath` methods `ad.sin` and `ad.log`.

#### Root finding algorithms implemented
 
| Root finding algorithm | Accepted `method` strings | Required arguments |
| - | - | - |
| Bisection | \['bisect', 'bisection', 'b'\] | `function`, `interval`, `method` |
| Newton-Fourier | \['newton-fourier', 'n-f'\] | `function`, `interval`, `method`|
| Newton-Raphson | \['newton', 'newton-raphson', 'n-r'\] |`function`, `start_values`, `method` |


See below for an example of how to find a root with `find_root` using the 'bisection' method.

``` python RootFinder example for the bisection method 
# Import the autodiffcc package
>>> import autodiffcc as ad

# Find the foot of a function with two variables using the bisection method

>>> def f(x, y):
>>>    return x + y - 100
>>> interval  = [[1, 2], [3, 100]]
>>> my_root = ad.find_root(function=f, method='bisection', interval=interval)
>>> print(my_root)
[1.999999999999993, 98.0]
```

The next example shows how to find a root with `find_root` using the 'newton-fourier' method and a vector function of two variables.

``` python
# Import the autodiffcc package
>>> import autodiffcc as ad
    >>> interval = [[3, -3], [3, -3]]
    >>> my_root = ad.find_root(lambda x, y: (2 * x + y - 2, y + 2), interval=interval, method='newton-fourier', max_iter=150)
    >>> print(my_root)
    [ 2. -2.]
```

A final example demonstrates how a user can find a root with `find_root` using the 'newton-raphson' method and a function of one variables.

``` python
# Import the autodiffcc package
>>> import autodiffcc as ad
    >>> def f1var(x):
    >>>     return (x + 2) * (x - 3)

    >>> my_root = ad.find_root(function=f1var, method='newton', start_values=1, threshold=1e-8)
    >>> print(my_root)
    3.
```

### Expression parsing

The below are two examples of parsing string expressions to function objects `fn` corresponding to the expressions. 

``` python 
# Import the autodiffcc package 
>>> import autodiffcc as ad
>>> from autodiffcc.parser import expressioncc

>>> x = ad.AD(2, der = [1, 0])
>>> y = ad.AD(3, der = [0, 1])

# Use expressioncc to parse a normal expression
>>> fn = expressioncc('x+y+1', ['x', 'y']).get_fn()
>>> print(fn(x,y).val)
6.0
>>> print(fn(x,y).der)
[1. 1.]

# Use expressioncc to parse an equation (left - right)
>>> fn = expressioncc('x = -y-1', ['x', 'y']).get_fn()
>>> print(fn(x,y).val)
6.0
>>> print(fn(x,y).der)
[1. 1.]
```

### Distribution and packaging
The AutoDiffCC library can be installed using the Python Package Index (PyPI). It is distributed in the pythonic formats of an sdist and as a wheel to facilitate installation via Python's package installer `pip`. The user will install the package with `pip` and them import it in Python. Importing the package via `pip` will also ensure that the user installs required dependencies.

Given that our aim is to deliver software that another developer can use for automatic differentiation in their own applications, we are not planning to deliver a standalone application. As such we are not planning to use a distribution framework beyond Python's native packaging.

## Software organization
### Directory structure
```
cs207-FinalProject/
    README.md
    LICENSE
    requirements.txt
    autodiffcc/
        __init__.py
        ADmath.py
        core.py
        parser.py
        root.py
        Equation/
            __init__.py
            core.py
            equation_base.py
            equation_scipy.py
            similar.py
            util.py
    docs/
        milestone1.md
        milestone2.md
        documentation.md
    tests/
        test_ADmath.py
        test_core.py
        test_parser.py
        test_root.py
    ...

```

### Modules
The `autodiffcc` package has four modules:

|Module|Basic Functionality|
|-|-|
|core| This is the main module, which contains the `AD` class and methods for operator overloading (e.g., add, mult, etc.).|
|ADmath| This module contains elementary functions, (e.g. sin, cos, sqrt, log, exp,etc.) for the `AD` class. |
|parser| This module contains our expression extension, which parses an expression string into the function object by extending the [Equation](https://github.com/glenfletcher/Equation) library for AD objects and more methods.\*|
|root| This module contains our root finding extension, which leverages out AD class and methods to find roots of vector equations using the Newton-Raphson, Newton-Fourier, and Bisection algorithms.|\

\* Given that we extended an existing [Equation](https://github.com/glenfletcher/Equation) library, content from that library which we forked and adapted for our extension is located in the `/autodiffcc/Equation/` directory

### Test suite
Our test suite is in the directory `cs207-FinalProject/tests`. We  use [TravisCI](https://travis-ci.org/Crimson-Computing/cs207-FinalProject) to perform continuous integration, running these tests with each build pushed to GitHub. We use [CodeCov](https://codecov.io/gh/Crimson-Computing/cs207-FinalProject) to ensure that our software implementation has sufficient code covered by our test suite. Badges indicating test compliance and code coverage are included in `README.md`. You can also view reports here: [TravisCI](https://travis-ci.org/Crimson-Computing/cs207-FinalProject) and [CodeCov](https://codecov.io/gh/Crimson-Computing/cs207-FinalProject).

### Installation
AutoDiffCC supports package installation via `pip`. Consumers and developers can install the package in the command line with the following command.

``` buildoutcfg
pip install autodiffcc
```

## Implementation (Base Automatic Differentiation Object and Methods)

### `AD` class
Our core module contains an `AD` class, including custom methods, that work on scalars and numpy arrays. These methods include an init, add, subtract, multiply, division, positive, negative, and comparison (<, ≥) dunder methods. 

Our `AD` class is used to create an `AD` object, which has a value and a derivative. These are stored as attributes of the object in its internal state, and can be represented as a tuple.

As operations or functions are applied to an `AD` object, we update the value and the derivative of the `AD` objects by using dual numbers in tuples, where the first element is the value and the second element (the "imaginary" part) is the derivative.

#### Differentiate
To facilitate use by the user and in other applications that leverage the `AD` class, such as our root finder, we developed a `differentiate` function. `differentiate` is a closure which takes a user defined base function as an input, and returns another function which can evaluate the base function's derivative at for a provided value.

#### Support for vector-valued functions
For vector-valued functions, we override the priority of numpy basic operators to ensure our `AD` objects are compatible with numpy arrays using our operators. Then, since the value and derivative in the `AD` object is stored as a numpy array, we use leverage vectorized numpy operations together with the chain rule to update the derivative for vector-valued functions.

### Elementary functions
We've developed custom elementary functions that support objects in our `AD` class and their structure, which stores the value and derivative as a dual number. Each of these is callable from the `ADmath` module, but a user who imports `autodiffcc` module does not need to import `ADmath` separately in order to use the `ADmath` elementary functions.

Implemented elementary math functions include log, exp, tan, power, trigonometric functions, and more.These are implemented by methods in the `ADmath` module which specifically extend the `numpy` implementations to apply the chain rule to update the derivative at each step.

### External Dependencies
Our core `AD` class, `ADmath` methods, and `find_root` function are dependent on `NumPy` for use of the `ndarray` class as a data structure and the elementary functions like `sin` and `log` from which we've constructed the `ADmath` methods.

Additionally, our `find_root` function's `bisection` method is dependent on the `matplotlib` library, for its feature which plots the user-defined function on the interval on which it search for a root.

Our testing suite is dependent on the `pytest` and `coverage` libraries for testing and reporting.

## Extension: Root finder
Our first extension is a root finding module, which leverages the `AD` class and methods to find a function or vector function's root. To find the root of a function means to find the values of its arguments for which the function's value is zero, This is, for example, useful in optimization tasks or in solving systems of equations. Over the years, a variety of methods have been proposed for this very common task. We have implemented three numerical root finding algorithms which leverage our `AD` object and `differentiate` methods: Newton-Raphson, Newton-Fourier, or Bisection algorithms.

### Newton-Raphson Method
Newton's method, also known as the Newton–Raphson method, named after Isaac Newton and Joseph Raphson. The Newton-Raphson method is an algorithm that when applied to real-valued functions produces successively better approximations to the roots.  

The most basic version of the algorithm for functions of x starts with a random value. It then evaluates the function f(x) at that value and checks if it is the root. Chances that the first guess is the root are very small. What the algorithm does if x is not the root is to find a line tangent to f(x) at x and see what for other value of x it intersects with 0. This is a new point to be evaluated. The more iterations executed, the closer we should be to the root. 
This iteration proceeds until root is approximated to desired precision. 

If all the assumptions requires are satisfied: 

x<sub>1</sub> = x<sub>1</sub> - (f(t<sub>0</sub>)/f't<sub>0</sub>) is a better approximation to the root. 





///OLD TEXT BUT CAN BE USEFUL: 
////We will develop a RootFinder for our advanced feature. Our RootFinder will implement Newton's method to approximate the roots of a real-valued function within a given tolerance. This will be in its own module, RootFinder. At this time, we don't foresee any additional modules, or data structures, but we may implement a Root class that can support real and possibly even complex roots.

///We select Newton's method for our RootFinder because it leverages differentiation and generalizes to high-dimensional problems and complex functions.  Our RootFinder, provided a function, will start by using `autodiffcc` to find the derivative of the function at an initial guess for a root. It will iterate through successively better approximations of the root along the function, taking the derivative with `autodiffcc` at each step, until it finds the root(s) within a given tolerance. An example of the potential use of the RootFinder is shown below. The user interaction is subject to change pending final implementation. 

### Newton-Fourier method
The Newton Fourier method, developed by Joseph Fourier, is an generalization of the Newton-Raphson method. Like the Newton-Raphson method, it iterates to produce successively better approximations of a function's root until it reaches quadratic convergence, but differs in that it provides a bound on the absolute error of the approximations.

Starting with a function `f` and an initial guess of the interval on which a root lies `[s,t]`, the function iteratively updates the end points of the interval such that iterations of `s` increase towards the root and iterations of `t` decrease towards the root.

The algorithm finds a root this in the following steps:
1. Calculate t<sub>n+1</sub> = t<sub>n</sub> - (f(t<sub>n</sub>)/f't<sub>n</sub>)
2. Calculate s<sub>n+1</sub> = s<sub>n</sub> - (f(s<sub>n</sub>)/f't<sub>n</sub>)
3. Calculate the distance between t<sub>n+1</sub> and s<sub>n+1</sub> scaled to the quadratic distance between t<sub>n</sub> and s<sub>n</sub>
4. Loop through steps 1, 2, and 3 until the distance measure is less than the `threshold` or if the number of iterations reaches `max_iter`
5. If the threshold criteria is reached, return the mean of t<sub>n+1</sub> and s<sub>n+1</sub> as the value of the root, otherwise if `max_iter` is reached raise an `Exception` for failure to converge. 

Where t<sub>n</sub> is the value of `t` at iteration `n`, s<sub>n</sub> is the value of `s` at iteration `n`, `f(t)` is the function `f` evaluated at value `t`, and `f'(t)` is the derivative of the function `f` evaluated at `t`.

As the number of iterations increases, this algorithm converges to the root of the function, with accuracy bounded within the user-specified threshold.

#### Support for vector-valued functions
We've adapted steps one and two of this general algorithm to support vector-valued functions. In place of the quotient of the function and its derivative evaluated at a value, we implement the product of the pseudoinverse of the Jacobian matrix and the function evaluated at the vector-values.

### Bisection method with interval halving and binary search
The bisection method is a root-finding algorithm that can be applied to any continuous function for which there exist values with opposite signs. If, for example, f(a)<0 and f(b)>0 there must exist at least one point c between a and b such that f(c)=0 The method is implemented by splitting an interval in half and then checking in which of the two halves the sign changes. This method finds the approximations of roots instead of the real value of the root. It is also relatively slow but very robust. 

For a function with only one argument for example f(x)=2+x the pseudocode is pretty straight forward: 
1) Choose an interval starting at a and ending at b.
2) Calculate point c that is placed in the middle, between a and b. 
3) Calculate f(x) 
4) If f(x) is close to zero (precision to be defined depending on the application) return c as the root and stop the iteration, otherwise, choose the new interval to be from a to c or from c to b depending on where the sign changes
5) Repeat 2, 3, 4 until convergence. 

Our method works not only on functions with one variable, but it returns the root for multivariate functions. If the dimension is higher, the process is implemented in the same way but dividing n-D spaces into smaller parts instead of dividing an interval. 

**Note:** The `autodiffcc.find_root` methods only return the function argument values for one root at a time. To find additional roots, if any, rerun `find_root` by initializing at a different `interval` or `start_values`. 


### Expression parsing

We also developed expression extension, which parses a string of expression into the function object corresponding to the expression. The returned function object takes AD objects as inputs and outputs. For example, the result for string expression ``'cosh(x,2) + 3 * arctan(y)'`` will be ``f(x,y) = cosh(x,2) + 3 * arctan(y)``.

Our implementation is build on an existing parser named [Equation](https://github.com/glenfletcher/Equation) on GitHub. We extend it by
- Returning function object with our AD objects as inputs/outputs.  
- Adding other mathematical operations defined in AD objects, such as ``arcsin`` and ``log``.

The input string for the parser can be either normal expressions such as ``'cosh(x,2) + 3 * arctan(y)'`` or equations such as ``'log(x,2) = sin(y)'``. The output of the equation will be the left side of it minus the right side of it (``'log(x,2) - sin(y)'`` for the example equation), on which we can apply rooting finding for the solutions of the equation. 

## Future work/possible extensions	

### LaTeX format

The parser of the package should be able to support LaTeX format apart from the normal format. Since people in academia may use LaTeX for writing papers and documentations, supporting LaTeX format will increase the flexibility of our package.

### Graphical User Interface 

We hope that people in other areas as well as computer engineers will be able use our package. However, people in other areas may not even know how to writing code with Python. Building a graphical user interface allows users to interact with packages through graphical icons, which makes this package easier to learn and use.

### Visualization

Although we can get the numeric results of values and derivatives for functions, we have no knowledge about how the function looks like. Simple visualization in 2 dimensions or 3 dimensions provides users a more vivid view of the function. For example, people in economic area will be able to analyze economic models with a big picture in mind.

### Extensibility

People in other areas, such as mathematics and statistics, may need more operations than we have thought about. Thus, another extension can be making the package extensible so that they could define new operations by themselves.

### Reverse Mode

We also suggest reverse mode be a future extension. Forward mode can be quite inefficient when the number of independent variables increases. On the other hand, the reverse mode computes the transpose of the Jacobian and is independent of the number of variables. When people deal with a huge amount of variables, such as machine learning or data mining, reverse mode is a more efficient choice compared to forward mode.



