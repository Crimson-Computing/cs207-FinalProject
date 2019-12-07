# driver.py:  can make use of the package by simple imports.
import pytest
import sys
sys.path.append('autodiffcc')
import autodiffcc as ad
import numpy as np
import inspect

differentiate = ad.differentiate

def f(x):
    f1 = ad.sin(3*(x**2)) + x * ad.tan(ad.sqrt(x*7))
    # f2 = y**(3*x) - ad.sin(x)
    return f1
dfdx = differentiate(f)
print(type(dfdx(x=4)))

# from Equation import Expression

# function = "sin(x+x^2)"
# fn_vars = ['x']
# x=3


# print(function)
# print(f'x={x}')
# print()
# def f(x):
# 	return ad.sin(x + x**2)

# x = ad.AD(x, n_vars=len(fn_vars))

# print("Hardcoded function (def f):\n  ", f(x))
# print()

# fn = Expression(function,fn_vars)
# x = ad.AD(3, n_vars=1)
# print("Equation parser:\n  ", fn(x))