
<img align="right" width="100" height="100" src="https://user-images.githubusercontent.com/43005886/70481100-8919f480-1aaf-11ea-8b0e-f8a8bde5c6ef.png">


[![Build Status](https://travis-ci.org/Crimson-Computing/cs207-FinalProject.svg?branch=master)](https://travis-ci.org/Crimson-Computing/cs207-FinalProject)

[![codecov](https://codecov.io/gh/Crimson-Computing/cs207-FinalProject/branch/master/graph/badge.svg)](https://codecov.io/gh/Crimson-Computing/cs207-FinalProject)




Click [here](https://github.com/Crimson-Computing/cs207-FinalProject/blob/master/docs/documentation.md) to see full documentation


## Final Project - AutoDiffCC Python Package
## CS207: Systems Development for Computational Science in Fall 2019 
#### Group 22
- Alex Spiride
- Maja Garbulinska
- Matthew Finney
- Zhiying Xu

### Overview 

With the evolution of science and the growing computational possibilities, differentiation plays a critical role in a wide range of scientific and industrial applications of computer science. However, the precise computation of symbolic derivatives is computationally expensive, and not even possible in all situations, whereas the finite differencing method is not always accurate or stable. Automatic differentiation, however, provides a computationally efficient way to calculate derivatives, particularly of complex functions, for applications where accuracy and performance at scale are important.

Our package **AutoDiffCC** provides is an easy to use package that computes derivates of scalar and vector functions using the concept of automatic differentation. 

We invite you to take a look at our repo and use **AutoDiffCC**!

### Installation Guide

AutoDiffCC supports package installation via `pip`. Users can install the package in the command line with the following command.

```buildoutcfg
pip install autodiffcc
```

### How To Use 
To use **AutoDiffCC** you first have to import it. If you already have it installed, you can do it by just running:

``` python 
# Import the autodiffcc package
>>> import autodiffcc as ad 
```

#### Basic Applications
There are several ways in which you can take advantage of **AutoDiffCC**. Below we present some examples.

###### Example 1  
A simple example using overloaded operators is described below. If you would like to evaluate ``f = x * x`` at ``x = 2``, first initiate an AD object ``x`` with ``x = ad.AD(2.0, 1.0)``, where ``2`` is the value and ``1`` is the derivative. Then simply define your function ``f = x * x`` and enjoy the results. You can see this example implemented below. 

``` python 
# Overload basic arithmetic operations
>>> x = ad.AD(val = 2.0, 1.0) 
>>> f = x * x
>>> print(f.val, f.der)
4.0 4.0
```

Alternatively, you can just proceed as follows: 

``` python 
>>> def f(x):
>>>   return x*x
>>> dfdx = differentiate(f)
>>> dfdx(x= 2.0)
4.0 # this is the derivative value at x=2 
```

###### Example 2

To use more complex function like cos(x) follow this example using our built-in module ADmath: 

``` python 
>>> x = AD(val = 3.0, der = 1.0)
>>> ADmath.cos(x) 
(array(-0.9899924966004454), array(-0.1411200080598672))
 ```    
 
 Again, you can also do: 
 
``` python 
>>> def f(x):
>>>   return ADmath.cos(x) 
>>> dfdx = differentiate(f)
>>> dfdx( x = 3.0)
-0.1411200080598672 # this is the derivative value evaluated at 3.0.
```
 

#### Offered Extentions
##### Root Finding
Our package offers three root finding methods. The bisection method, the newton-fourier method and the newton-raphson method.

###### Example 1 

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

###### Example 2

``` python
# Import the autodiffcc package
>>> import autodiffcc as ad
    >>> interval = [[3, -3], [3, -3]]
    >>> my_root = ad.find_root(lambda x, y: (2 * x + y - 2, y + 2), interval=interval, method='newton-fourier', max_iter=150)
    >>> print(my_root)
    [ 2. -2.]
```

###### Example 3

``` python
# Import the autodiffcc package
>>> import autodiffcc as ad
    >>> def f1var(x):
    >>>     return (x + 2) * (x - 3)

    >>> my_root = ad.find_root(function=f1var, method='newton', start_values=1, threshold=1e-8)
    >>> print(my_root)
    3.
```

##### Expression parsing

###### Example 1

Another extension we offer is expression parsing. The below are two examples of parsing string expressions to function objects `fn` corresponding to the expressions. 

``` python 
>>> x = AD(2, der = [1, 0])
>>> y = AD(3, der = [0, 1])

# Use expressioncc to parse a normal expression
>>> fn = ad.expressioncc('x+y+1', ['x', 'y']).get_fn()
>>> print(fn(x,y).val)
6.0
>>> print(fn(x,y).der)
[1. 1.]

# Use expressioncc to parse an equation (left - right)
>>> fn = ad.expressioncc('x = -y-1', ['x', 'y']).get_fn()
>>> print(fn(x,y).val)
6.0
>>> print(fn(x,y).der)
[1. 1.]
```


