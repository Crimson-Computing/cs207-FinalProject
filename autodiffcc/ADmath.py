import numpy as np
from core import AD

def cos(obj):
    """Returns the cos of a scalar or an AD object 

    INPUTS
    =======
    obj: AD object or a scalar
    
    RETURNS
    ========
    Returns the cos of obj
    
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.cos(x) 
    (array(-0.9899924966004454), array(-0.1411200080598672))
    """
    # if scalar 
    if np.isscalar(obj):
        return(np.cos(obj))
     
    # if AD object 
    if isinstance(obj, AD):
        return AD(val=np.cos(obj.val), der=-np.sin(obj.val) * obj.der)

def sin(obj):
    """Returns the sin of a scalar or an AD object 

    INPUTS
    =======
    obj: AD object or a scalar
    
    RETURNS
    ========
    Returns the sin of obj
    
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.sin(x) 
    (array(0.1411200080598672), array(-0.9899924966004454))
    """

    # if scalar 
    if np.isscalar(obj):
        return(np.sin(obj))
        
    # if AD object 
    if isinstance(obj, AD):
        return AD(val = np.sin(obj.val), der = np.cos(obj.val) * obj.der)

def tan(obj):
    """Returns the tan of a scalar or an AD object 

    INPUTS
    =======
    obj: AD object or a scalar
    
    RETURNS
    ========
    Returns the tan of obj
    
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.tan(x) 
    (array(-0.1425465430742778), array(1.020319516942427))
    """
    # if scalar 
    if np.isscalar(obj):
        return(np.tan(obj))
        
    # if AD object 
    if isinstance(obj, AD):
        return AD(val = np.tan(obj.val), der = (1 / (np.cos(obj.val) ** 2)) * obj.der) 

def exp(obj):
    """Returns the exp of a scalar or an AD object 

    INPUTS
    =======
    obj: AD object or a scalar
    
    RETURNS
    ========
    Returns the exp of obj
    
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.exp(x) 
    (array(20.085536923187668), array(20.085536923187668))
    """
    # if scalar 
    if np.isscalar(obj):
        return(np.exp(obj))

    # if AD object 
    if isinstance(obj, AD):
        return AD(val = np.exp(obj.val), der = np.exp(obj.val) * obj.der)

def sqrt(obj):
    """Returns the sqrt of a scalar or an AD object 

    INPUTS
    =======
    obj: AD object or a scalar
    
    RETURNS
    ========
    Returns the sqrt of obj
    
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.sqrt(x) 
    (array(1.7320508075688772), array(0.28867513459481287))
    """
    # if scalar 
    if np.isscalar(obj):
        if obj <= 0 :
            raise TypeError('Sqrt has a positive domain, input negative')
        else: 
            return(np.sqrt(obj))
     
    # if AD object 
    if isinstance(obj, AD):
        val = obj.val
        if (val <= 0).any(): 
            raise TypeError('Sqrt has a positive domain, input negative')
        else:
            return obj.__pow__(0.5)

def arcsin(obj):
    """Returns the arcsin of a scalar or an AD object 

    INPUTS
    =======
    obj: AD object or a scalar
    
    RETURNS
    ========
    Returns the arcsin of obj
    
    EXAMPLES
    =========
    >>> x = AD(val=0.5, der=1)
    >>> print(ADmath.arcsin(x)) 
    (array(0.52359878), array(1.15470054))
    """

    if np.isscalar(obj):
        if obj < (-1) or obj> (1):
            raise ValueError('Values are not in the domain of arcsin [-1, 1].')
        else: 
            return(np.arcsin(obj))
            
    if isinstance(obj, AD):
        values = obj.val 
        if (values < (-1) or values > (1)).any():
            raise ValueError('Values are not in the domain of arcsin [-1, 1].')

        val = np.arcsin(obj.val)
        der = 1 / np.sqrt(1 - (obj.val ** 2) * obj.der)

        return AD(val = val, der = der)

def arccos(obj):
    """Returns the arccos of a scalar or an AD object 

    INPUTS
    =======
    obj: AD object or a scalar
    
    RETURNS
    ========
    Returns the arccos of obj
    
    EXAMPLES
    =========
    >>> x = AD(val=0.5, der=1)
    >>> print(ADmath.arcos(x)) 
    (array(1.04719755), array(-1.15470054))
    """
    if np.isscalar(obj):
        if obj < (-1) or obj> (1):
            raise ValueError('Values are not in the domain of arcsin [-1, 1].')
        else: 
            return(np.arccos(obj))
    

    if isinstance(obj, AD):
        values = obj.val 
        if (values < (-1) or values > (1)).any():
            raise ValueError('Values are not in the domain of arcsin [-1, 1].')
        
        val = np.arccos(obj.val)
        # this is the negative of the derivative of the arcsin 
        der = -1 * (1 / np.sqrt(1 - (obj.val ** 2) * obj.der))
        return AD(val = val, der = der)

def arctan(obj):
    """Returns the arctan of a scalar or an AD object 

    INPUTS
    =======
    obj: AD object or a scalar
    
    RETURNS
    ========
    Returns the arctan of obj
    
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.arctan(x) 
    (array(1.24904577), array(0.1))
    """
    if np.isscalar(obj):
        return(np.arctan(obj))
    
    if isinstance(obj, AD):
        val = np.arctan(obj.val)
        der = 1/((obj.val**2) +1)* obj.der
        return AD(val = val, der = der)

def sinh(obj):
    """Returns the sinh of a scalar or an AD object 

    INPUTS
    =======
    obj: AD object or a scalar
    
    RETURNS
    ========
    Returns the sinh of obj
    
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.sinh(x) 
    (array(10.01787493), array(10.067662))
    
    """
    if np.isscalar(obj):
        return(np.sinh(obj))
    
    if isinstance(obj, AD):
        val = np.sinh(obj.val)
        der = np.cosh(obj.val) * obj.der
        return AD(val = val, der = der)

def cosh(obj):
    """Returns the cosh of a scalar or an AD object 

    INPUTS
    =======
    obj: AD object or a scalar
    
    RETURNS
    ========
    Returns the cosh of obj
    
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.cosh(x) 
    (array(10.067662), array(10.01787493))
    """
    
    if np.isscalar(obj):
        return(np.cosh(obj))
    
    if isinstance(obj, AD):
        val = np.cosh(obj.val)
        der = np.sinh(obj.val) * obj.der
        return AD(val = val, der = der)

def tanh(obj):
    """Returns the tanh of a scalar or an AD object 

    INPUTS
    =======
    obj: AD object or a scalar
    
    RETURNS
    ========
    Returns the tanh of obj
    
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.tanh(x) 
    (array(0.99505475), array(0.00986604))
    """
    if np.isscalar(obj):
        return(np.tanh(obj))
        
    # the derivative of tanh is sech^2(x) and sech(x) can be defined as 1/cosh(x)
        
    if isinstance(obj, AD):
        val = np.tanh(obj.val)
        der = ((1/np.cosh(obj.val))**2) * obj.der
        return AD(val = val, der = der)

def logistic(obj):
    """Returns the logit of a scalar or an AD object 

    INPUTS
    =======
    obj: AD object or a scalar
    
    RETURNS
    ========
    Returns the logit of obj
    
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.logistic(x) 
    (array(0.95257413), array(0.04517666))
    """
    
    if np.isscalar(obj):
        return(1 / (1+ np.exp(-obj)))
    
    if isinstance(obj, AD):
    
        val = (1 / (1+ np.exp(-obj.val)))
        der = ((np.exp(-obj.val))/(1+np.exp(-obj.val))**2)*obj.der
        
        return AD(val = val, der = der)

def log(obj, base=None):
    """Returns the log of a scalar or an AD object with any base

    INPUTS
    =======
    obj: AD object or a scalar
    base: the base of the log, if None defaults to natural log
    
    RETURNS
    ========
    Returns the log of obj
    
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.log(x) 
    (array(0.47712125), array(0.14476483))
    """
    
    if not base:
        base = np.exp(1)

    if np.isscalar(obj):
        return np.log(obj) / np.log(base)
    
    if isinstance(obj, AD):
        if obj <= 0:
            raise ValueError('Log accepts only positive numbers')
        val = np.log(obj.val) / np.log(base)
        der = (1 / (obj.val * np.log(base))) * obj.der
        return AD(val = val, der = der)
