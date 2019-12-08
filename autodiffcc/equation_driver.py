from core import *
from ADmath import *
from Equation import Expression
function = "-sinh(x + x**2)"
fn_vars = ['x']
x=3
print(function)
print(f'x={x}')
print()
def f(x):
  return -sinh(x + x**2)
x = AD(x, n_vars=len(fn_vars))
print("Hardcoded function (def f):\n  ", f(x))
print()
fn = Expression(function,fn_vars)
x = AD(3, n_vars=1)
print("Equation parser:\n  ", fn(x))
