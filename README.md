[![Build Status](https://travis-ci.org/Crimson-Computing/cs207-FinalProject.svg?branch=master)](https://travis-ci.org/Crimson-Computing/cs207-FinalProject)

[![codecov](https://codecov.io/gh/Crimson-Computing/cs207-FinalProject/branch/master/graph/badge.svg)](https://codecov.io/gh/Crimson-Computing/cs207-FinalProject)

![56698097](https://user-images.githubusercontent.com/43005886/70481100-8919f480-1aaf-11ea-8b0e-f8a8bde5c6ef.png)



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

```
# Import the autodiffcc package
>>> import autodiffcc as ad 
```

#### Basic Applications
There are several ways in which you can take advantage of **AutoDiffCC**. Below we present some examples.

##### Example 1  
A simple example using overloaded operators is described below. If you would like to evaluate ``f = x * x`` at ``x = 2``, first initiate an AD object ``x`` with ``x = ad.AD(2.0, 1.0)``, where ``2`` is the value and ``1`` is the derivative. Then simply define your function ``f = x * x`` and enjoy the results. You can see this example implemented below. 

``` 
# Overload basic arithmetic operations
>>> x = ad.AD(val = 2.0, 1.0) 
>>> f = x * x
>>> print(f.val, f.der)
4.0 4.0
```

Alternatively, you can just proceed as follows: 

``` 
>>> def f(x):
>>>   return x*x
>>> dfdx = differentiate(f)
>>> dfdx(x= 2.0)
4.0 # this is the derivative value at x=2 
```

##### Example 2

To use more complex function like cos(x) follow this example using our built-in module ADmath: 

``` 
>>> x = AD(val = 3.0, der = 1.0)
>>> ADmath.cos(x) 
(array(-0.9899924966004454), array(-0.1411200080598672))
 ```    
 
 Again, you can also do: 
 
 ``` 
>>> def f(x):
>>>   return ADmath.cos(x) 
>>> dfdx = differentiate(f)
>>> dfdx( x = 3.0)
-0.1411200080598672 # this is the derivative value evaluated at 3.0.
```
 

#### Offered Extentions

