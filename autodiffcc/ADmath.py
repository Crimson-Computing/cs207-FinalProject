import numpy as np
from autodiffcc.core import AD
import math as math 


def cos(ad_object):
    """Returns the cos of ad_object 
    INPUTS
    =======
    AD object
    RETURNS
    ========
    Returns the cos of ad_object 
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.cos(x) 
    (array(-0.9899924966004454), array(-0.1411200080598672))
    """
    if not isinstance(ad_object, AD):
        raise TypeError('This function can only take AD objects as inputs. Input of %s is not an AD object.' % type(
            ad_object).__name__)
    return AD(val=np.cos(ad_object.val), der=-np.sin(ad_object.val) * ad_object.der)


def sin(ad_object):
    """Returns the sin of ad_object 
    INPUTS
    =======
    AD object
    RETURNS
    ========
    Returns the sin of ad_object 
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.sin(x) 
    (array(0.1411200080598672), array(-0.9899924966004454))
    """
    if not isinstance(ad_object, AD):
        raise TypeError('This function can only take AD objects as inputs. Input of %s is not an AD object.' % type(
            ad_object).__name__)
    return AD(val = np.sin(ad_object.val), der = np.cos(ad_object.val) * ad_object.der)


def tan(ad_object):
    """Returns the tan of ad_object 
    INPUTS
    =======
    AD object
    RETURNS
    ========
    Returns the tan of ad_object
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.tan(x) 
    (array(-0.1425465430742778), array(1.020319516942427))
    """
    if not isinstance(ad_object, AD):
        raise TypeError('This function can only take AD objects as inputs. Input of %s is not an AD object.' % type(
            ad_object).__name__)
    return AD(val = np.tan(ad_object.val), der = (1 / (np.cos(ad_object.val) ** 2)) * ad_object.der) 


def exp(ad_object):
    """Returns the exponent of ad_object 
    INPUTS
    =======
    AD object
    RETURNS
    ========
    e to the power of ad_object 
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.exp(x) 
    (array(20.085536923187668), array(20.085536923187668))
    """
    if not isinstance(ad_object, AD):
        raise TypeError('This function can only take AD objects as inputs. Input of %s is not an AD object.' % type(
            ad_object).__name__)
    return AD(val = np.exp(ad_object.val), der = np.exp(ad_object.val) * ad_object.der)


def sqrt(ad_object):
    """Returns the sqrt of ad_object 
    INPUTS
    =======
    AD object
    RETURNS
    ========
    ad_object to the exponent of 1/2
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.sqrt(x) 
    (array(1.7320508075688772), array(0.28867513459481287))
    """
    if not isinstance(ad_object, AD):
        raise TypeError('This function can only take AD objects as inputs. Input of %s is not an AD object.' % type(
            ad_object).__name__)
    return ad_object.__pow__(0.5)


def arcsin(ad_object):
    '''Returns the arcsin of ad_object 
    INPUTS
    =======
    AD object
    RETURNS
    ========
    arcsin of an AD object
    EXAMPLES
    =========
    >>> x = AD(val=0.5, der=1)
    >>> print(ADmath.arcsin(x)) 
    (array(0.52359878), array(1.15470054))
    '''

    if not isinstance(ad_object, AD):
        raise TypeError('This function can only take AD objects as inputs. Input of %s is not an AD object.' % type(
            ad_object).__name__)
    values = ad_object.val 
    if values < (-1) or values> (1):
        raise ValueError('Values are not in the domain of arcsin [-1, 1].')

    val = np.arcsin(ad_object.val)
    der = 1 / np.sqrt(1 - (ad_object.val ** 2) * ad_object.der)

    return AD(val = val, der = der)


def arccos(ad_object):
    '''Returns the arccos of ad_object 
    INPUTS
    =======
    AD object
    RETURNS
    ========
    arcsos of an AD object
    EXAMPLES
    =========
    >>> x = AD(val=0.5, der=1)
    >>> print(ADmath.arcos(x)) 
    (array(1.04719755), array(-1.15470054))
    '''
    
    if not isinstance(ad_object, AD):
        raise TypeError('This function can only take AD objects as inputs. Input of %s is not an AD object.' % type(
            ad_object).__name__)
    values = ad_object.val 
    if values < (-1) or values> (1):
        raise ValueError('Values are not in the domain of arcsin [-1, 1].')
        
    val = np.arccos(ad_object.val)
    # this is the negative of the derivative of the arcsin 
    der = -1 * (1 / np.sqrt(1 - (ad_object.val ** 2) * ad_object.der))

    return AD(val = val, der = der)



def arctan(ad_object):
    """Returns the arctan of ad_object 
    INPUTS
    =======
    AD object
    RETURNS
    ========
    Returns the cos of ad_object 
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.arctan(x) 
    (array(1.24904577), array(0.1))
    """
    
    if not isinstance(ad_object, AD):
        raise TypeError('This function can only take AD objects as inputs. Input of %s is not an AD object.' % type(
            ad_object).__name__)
    
    val = np.arctan(ad_object.val)
    der = 1/((ad_object.val**2) +1)* ad_object.der
    return AD(val = val, der = der)



def sinh(ad_object):
    """Returns the sinh of ad_object 
    INPUTS
    =======
    AD object
    RETURNS
    ========
    Returns the cos of ad_object 
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.sinh(x) 
    (array(10.01787493), array(10.067662))
    """
    
    if not isinstance(ad_object, AD):
        raise TypeError('This function can only take AD objects as inputs. Input of %s is not an AD object.' % type(
            ad_object).__name__)
        
    val = np.sinh(ad_object.val)
    der = np.cosh(ad_object.val) * ad_object.der
    return AD(val = val, der = der)



def cosh(ad_object):
    """Returns the cosh of ad_object 
    INPUTS
    =======
    AD object
    RETURNS
    ========
    Returns the cos of ad_object 
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.cosh(x) 
    (array(10.067662), array(10.01787493))
    """
    
    if not isinstance(ad_object, AD):
        raise TypeError('This function can only take AD objects as inputs. Input of %s is not an AD object.' % type(
            ad_object).__name__)
        
    val = np.cosh(ad_object.val)

    der = np.sinh(ad_object.val) * ad_object.der
    return AD(val = val, der = der)



def tanh(ad_object):
    """Returns the tan of ad_object 
    INPUTS
    =======
    AD object
    RETURNS
    ========
    Returns the cos of ad_object 
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.tan(x) 
    (array(-0.14254654), array(1.02031952))
    """
    
    if not isinstance(ad_object, AD):
        raise TypeError('This function can only take AD objects as inputs. Input of %s is not an AD object.' % type(
            ad_object).__name__)
    # the derivative of tanh is sech^2(x) and sech(x) can be defined as 1/cosh(x)
    
   
    val = np.tanh(ad_object.val)
    der = ((1/np.cosh(ad_object.val))**2) * ad_object.der
    return AD(val = val, der = der)




def logistic(ad_object):
    """Returns the logit of ad_object 
    INPUTS
    =======
    AD object
    RETURNS
    ========
    Returns the cos of ad_object 
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.logistic(x) 
    (array(0.95257413), array(0.04517666))
    """
    
    if not isinstance(ad_object, AD):
        raise TypeError('This function can only take AD objects as inputs. Input of %s is not an AD object.' % type(
            ad_object).__name__)
    
    val = (1 / (1+ np.exp(-ad_object.val)))
    der = ((np.exp(-ad_object.val))/(1+np.exp(-ad_object.val))**2)*ad_object.der
    
    return AD(val = val, der = der)

# this is log base 10 
def log(ad_object):
    """Returns the log base 10 of ad_object 
    INPUTS
    =======
    AD object
    RETURNS
    ========
    Returns the cos of ad_object 
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.log(x) 
    (array(0.47712125), array(0.14476483))
    """
    
    if not isinstance(ad_object, AD):
        raise TypeError('This function can only take AD objects as inputs. Input of %s is not an AD object.' % type(
            ad_object).__name__)
    values = ad_object.val 
    if values <= 0:
        raise ValueError('Log accepts only positive numbers')
        
    val = math.log(ad_object.val, 10)
    der = (1/(ad_object.val * np.log(10)))*ad_object.der
    return AD(val = val, der = der)
    

# this is natural log 
def ln(ad_object):
    """Returns the natural log of ad_object 
    INPUTS
    =======
    AD object
    RETURNS
    ========
    Returns the cos of ad_object 
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.cos(x) 
    (array(1.09861229), array(0.33333333))
    """
    
    if not isinstance(ad_object, AD):
        raise TypeError('This function can only take AD objects as inputs. Input of %s is not an AD object.' % type(
            ad_object).__name__)
    values = ad_object.val 
    if values <= 0:
        raise ValueError('Ln accepts only positive numbers')
       
    val = np.log(ad_object.val) 
    der = (1/(ad_object.val)) *ad_object.der
    return AD(val = val, der = der)

