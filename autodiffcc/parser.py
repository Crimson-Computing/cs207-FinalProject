from autodiffcc.core import AD
from autodiffcc.ADmath import *
from autodiffcc.Equation import Expression

class expressioncc():
  """
  Return an AD object after calculations.
    
  ATTRIBUTES
  ==========
  line : A line of the calculation expression
  fn_vars: Number of varriables in the line
  fn: Function build based on the line
  
  METHODS
  =======
  Calculate the value and derrivative based on a string
  
  EXAMPLES
  ========
  # >>> fn = expressioncc('3 * log(x,2) + sin(7)', ['x']).get_fn()
  # (array(6.6569866), array([1.08202128]))
  # >>> fn(AD(4, n_vars=1))
  # (array(2.69120231), array([2.7050532]))
  # >>> fn = expressioncc('3 * log(x,2) + sin(7)', ['x']).get_fn()
  #  >>> fn(AD(4, n_vars=1)) 
  #  (array(2.69120231), array([2.7050532]))
  """
  def __init__(self, line, fn_vars):
    self.line = line
    self.log_parsing()
    self.equation_parsing()
    self.fn_vars = fn_vars
    self.fn = Expression(self.line,self.fn_vars)

  def equation_parsing(self):
    """Returns left - right if string is equation
    
    RETURNS
    ========
    Returns the function
    
    EXAMPLES
    =========
    >>> fn = expressioncc('3 * log(x,2) = - sin(7)', ['x']).get_fn()
    >>> fn(AD(4, n_vars=1)) 
    (array(2.69120231), array([2.7050532]))
    """
    if '=' in self.line:
      self.line = self.line.replace('=','-(') + ')'

  def log_parsing(self):
    """parsing results of lof 
    
    EXAMPLES
    =========
    >>> self.line = log(x,2) + 5
    # after parsing
    >>> self.line = (x log 2) + 5
    """
    self.line = self.line.replace('log', '')
    self.line = self.line.replace(',', ' log ')

  def get_fn(self):
    """Returns function of the expression 
    
    RETURNS
    ========
    Returns the function
    
    EXAMPLES
    =========
    >>> fn = expressioncc('3 * log(x,2) + sin(7)', ['x']).get_fn()
    >>> fn(AD(4, n_vars=1)) 
    (array(2.69120231), array([2.7050532]))
    """
    return self.fn
