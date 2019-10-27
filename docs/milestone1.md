# Milestone 1 Document
CS207 Final Project, Group 22
Alex Spiride, Maja Garbulinska, Matthew Finney, Zhiying Xu

## Introduction

## Background


Our software will make it easier for the user to compute derivatives using automatic differentiation. Autodifferentiation provides efficiency and numerical stability as opposed to other methods such as finite differences. A computer can perform elementary operations quickly. If we apply the chain rule to these elementary operations we can compute derivatives of functions with working precision.

To work, the package makes use of the following concepts:

### Forward Mode

The forward mode uses the chain rule described below to compute derivatives of nested functions. The chain rule is applied to elementary operations step by step starting with the most inner operation. 

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

See the examples below: 


<img width="84" alt="Screen Shot 2019-10-27 at 16 57 28" src="https://user-images.githubusercontent.com/43005886/67641740-e6c8f780-f8da-11e9-9a9f-a731639fe798.png">


Click [here](https://en.wikipedia.org/wiki/Riemann_zeta_function) to see an example of a non-elementary function. 

### More information

If you would like to know more about this topic, you should have a look [this book](https://arxiv.org/pdf/1411.0583.pdf).

## How to use the package

## Software organization

## Implementation
