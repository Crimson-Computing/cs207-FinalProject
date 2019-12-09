# AutoDiffCC Final Documentation
CS207 Final Project, Group 22
Alex Spiride, Maja Garbulinska, Matthew Finney, Zhiying Xu

## Introduction
With the evolution of science and the growing computational possibilities, differentiation plays a critical role in a wide range of scientific and industrial applications of computer science. However, the precise computation of symbolic derivatives is computationally expensive, and not even possible in all situations, whereas the finite differencing method is not always accurate or stable. Automatic Differentiation, however, provides a computationally efficient way to calculate derivatives, particularly of complex functions, for applications where accuracy and performance at scale are important.

## Background

Authomatic differentiation(AD) is also known as computational differentiation, algorithmic differentation and differentiation of algorithms. Authomatic differentiation does not use symbolic expressions but rather exact formulas and floating-point values. It provides a great way to avoid approximation errors. It easier for the user to compute derivatives using automatic differentiation, which provides efficiency and numerical stability as opposed to other methods such as finite differences. 

The most important idea that AD benefits from is the chain rule and the fact that it can be implemented in a numerical program. Deferentiation is applied to elementary function step by step to get the final results. A computer can perform elementary operations quickly. If we apply the chain rule to these elementary operations we can compute derivatives of functions efficiently and with working precision.

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
>>> import autodiffcc as ad

# Find the derivative of e^x
>>> x = ad.AD(val = 3, der = 1)
>>> g = ad.exp(x)
>>> print(g.val, g.der)
20.085536923187668 20.085536923187668
```

### Root finding

``` RootFinder example for the bisection method 
# find the foot of a function with two variables
>>> def f(x, y):
>>>    return(x+y-100)
>>>    interval  = [[1, 2], [3, 100]]
>>> find_root(function=f, method='bisection', interval=interval)
[1.999999999999993, 98.0]
```
### Expression
``` 
>>> x = AD(2, der = [1, 0])
>>> y = AD(3, der = [0, 1])

# Use expressioncc to parse a normal expression
>>> fn = ad.expressioncc('x+y+1', ['x', 'y']).get_fn()
>>> print(fn(x,y).val)
6.0
>>> print(fn(x,y).der)
[1. 1.]

# Use expressioncc to parse an equation
>>> fn = ad.expressioncc('x=-y-1', ['x', 'y']).get_fn()
>>> print(fn(x,y).val)
6.0
>>> print(fn(x,y).der)
[1. 1.]
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
            requirements.txt
			autodiffcc/
				Equation/
                __init__.py
				ADmath.py
				core.py
				parser.py
				rootfindercc.py [Placeholder]
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
The `autodiffcc` package will have four modules:

|Module|Basic Functionality|
|-|-|
|core| This is the main module, which will contain the `AD` class and methods for operator overloading (e.g., add, mult, etc.).|
|ADmath| This module contains elementary functions, (e.g. sin, cos, sqrt, log, exp,etc.) for the `AD` class. |
|RootFinder| This module is a placeholder for the advanced feature.|
|Expression| This module parses a string of expression into the function object corresponding to the expression.|

### Test suite
Our test suite is in the directory `cs207-FinalProject/tests`. We  use TravisCI to perform continuous integration, running these tests with each build pushed to GitHub. We use CodeCov to ensure that our software implementation has sufficient code covered by our test suite. Badges indicating test compliance and code coverage are included in `README.md`.

## Implementation
// TODO  # Streamline/simplify existing content
// TODO  # Discuss implementation of root finder

Our `AD` class is used to create an AD object, including custom methods, that work on scalars and numpy arrays. The AD class will then be used in our extension class, which will be an object of its own (RootFinder). Each of the math methods will be callable from an imported library of math functions.

The AD class has several methods, including an init, add, subtract, multiply, division, positive, negative, and comparison (<, ≥) dunder methods. AD objects will have a value and a derivative. The math functions will include functions such as log, exp, tan, power, trigonometric functions, and more. To deal with elementary functions like sin, sqrt, log, and exp and all the others, we will write methods to extend general implementations (e.g. numpy) updating the derivative at each step. Finally, the AD class will have attributes to get the derivative and value of the object. We are updating the derivative of our AD objects by using dual numbers in tuples, where the first element is the value and the second element (the "imaginary" part) is the derivative. 

For vector-valued functions, we will override the priority of numpy basic operators to ensure our AD objects are compatible with numpy arrays using our operators. Then, since the value and derivative in the AD object is stored as a numpy array, we will be using the numpy operations in our derivative updating together with the chain rule to update the derivative for vector-valued functions.

We want to make our class compatible with numpy arrays, so we will need to use NumPy, as well as math. For testing, we will need doctest and pytest, and we might use scipy for the RootFinder. 


## Extension: Root finder
// TODO  # Update tense
// TODO  # Write about each of the three algorithms

To find a root of a function means to find the values for it's parameters that make the function's value be 0. A variety of methods have been proposed over the years. We have decided to implement three of them. 

### Newton
//// TO DO ALEX

///OLD TEXT BUT CAN BE USEFUL: 
////We will develop a RootFinder for our advanced feature. Our RootFinder will implement Newton's method to approximate the roots of a real-valued function within a given tolerance. This will be in its own module, RootFinder. At this time, we don't foresee any additional modules, or data structures, but we may implement a Root class that can support real and possibly even complex roots.

///We select Newton's method for our RootFinder because it leverages differentiation and generalizes to high-dimensional problems and complex functions.  Our RootFinder, provided a function, will start by using `autodiffcc` to find the derivative of the function at an initial guess for a root. It will iterate through successively better approximations of the root along the function, taking the derivative with `autodiffcc` at each step, until it finds the root(s) within a given tolerance. An example of the potential use of the RootFinder is shown below. The user interaction is subject to change pending final implementation. 


### Bisection (interval halving, binary search method)

The bisection method is a root-finding algorithm that can be applied to any continous function for which there exist values with opposite signs. If, for example, f(a)<0 and f(b)>0 there must exist at least one point c between a and b such that f(c)=0 The method is implemented by splitting an interval in half and then checking in which of the two halfs the sign changes. This method finds the approximations of roots instead of the real value of the root. It is also relatively slow but very robust. 

For a function with only one argument for example f(x)=2+x the pseudocode is pretty straight forward: 
1) Choose an interval starting at a and ending at b.
2) Calculate point c that is placed in the middle, between a and b. 
3) Calculate f(x) 
4) If f(x) is close to zero (precision to be defined depending on the application) return c as the root and stop the iteration, otherwise, choose the new interval to be from a to c or from c to b depending on where the sign changes
5) Repeat 2, 3, 4 until convergance. 

Our method works not only on functions with one variable, but it returns the root for multivariable functions. If the dimention is higher, the process is implemented in the same way but dividing n-D spaces into smaller parts instead of dividing an interval. 

### Fourier - Matt 
/// TO DO MATT



### Expression

We also developed expression extension, which parses a string of expression into the function object corresponding to the expression. The returned function object takes AD objects as inputs and outputs. For example, the result for string expression ``'cosh(x,2) + 3 * arctan(y)'`` will be ``f(x,y) = cosh(x,2) + 3 * arctan(y)``.

Our implementation is build on a previous parser named [Equation](https://github.com/glenfletcher/Equation) on GitHub. We extend it by
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



