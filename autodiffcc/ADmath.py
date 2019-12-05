import numpy as np
from autodiffcc.core import AD


def cos(ad_object):
    if not isinstance(ad_object, AD):
        raise TypeError(
            f'This function can only take AD objects as inputs. Input of {type(ad_object).__name__} is not an AD object.')

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

    return AD(val=np.cos(ad_object.val), der=-np.sin(ad_object.val)) * ad_object.der


def sin(ad_object):
    if not isinstance(ad_object, AD):
        raise TypeError(
            f'This function can only take AD objects as inputs. Input of {type(ad_object).__name__} is not an AD object.')


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

    return AD(val=np.sin(ad_object.val), der=np.cos(ad_object.val)) * ad_object.der


def tan(ad_object):
    if not isinstance(ad_object, AD):
        raise TypeError(
            f'This function can only take AD objects as inputs. Input of {type(ad_object).__name__} is not an AD object.')


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

    return AD(val=np.tan(ad_object.val), der=(1 / (np.cos(ad_object.val) ** 2))) * ad_object.der


def exp(ad_object):
    if not isinstance(ad_object, AD):
        raise TypeError(
            f'This function can only take AD objects as inputs. Input of {type(ad_object).__name__} is not an AD object.')


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

    return AD(val=np.exp(ad_object.val), der=np.exp(ad_object.val))*ad_object.der


def sqrt(ad_object):
    if not isinstance(ad_object, AD):
        raise TypeError(
            f'This function can only take AD objects as inputs. Input of {type(ad_object).__name__} is not an AD object.')


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
    return ad_object.__pow__(0.5)


def arcsin(ad_object):
    if not isinstance(ad_object, AD):
        raise TypeError(
            f'This function can only take AD objects as inputs. Input of {type(ad_object).__name__} is not an AD object.')


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

    # Check that values are in the domain of arcsin
    # TODO: Will we need to do this when vectorized?
    # values = map(lambda x: -1 <= x <= 1, ad_object.val)
    # if not all(values):
    values = ad_object.val
    if not values:
        raise ValueError('Values are not in the domain of arcsin [-1, 1].')

    val = np.arcsin(ad_object.val)
    der = 1 / np.sqrt(1 - (ad_object.val ** 2)) * ad_object.der

    return AD(val=val, der=der)


def arccos(ad_object):
    if not isinstance(ad_object, AD):
        raise TypeError(
            f'This function can only take AD objects as inputs. Input of {type(ad_object).__name__} is not an AD object.')


    # TODO
    pass


def arctan(ad_object):
    if not isinstance(ad_object, AD):
        raise TypeError(
            f'This function can only take AD objects as inputs. Input of {type(ad_object).__name__} is not an AD object.')


    # TODO
    pass


def sinh(ad_object):
    if not isinstance(ad_object, AD):
        raise TypeError(
            f'This function can only take AD objects as inputs. Input of {type(ad_object).__name__} is not an AD object.')


    # TODO
    pass


def cosh(ad_object):
    if not isinstance(ad_object, AD):
        raise TypeError(
            f'This function can only take AD objects as inputs. Input of {type(ad_object).__name__} is not an AD object.')


    # TODO
    pass


def tanh(ad_object):
    if not isinstance(ad_object, AD):
        raise TypeError(
            f'This function can only take AD objects as inputs. Input of {type(ad_object).__name__} is not an AD object.')


    # TODO
    pass


def logistic(ad_object):
    if not isinstance(ad_object, AD):
        raise TypeError(
            f'This function can only take AD objects as inputs. Input of {type(ad_object).__name__} is not an AD object.')


    # TODO
    pass


def log(ad_object):
    if not isinstance(ad_object, AD):
        raise TypeError(
            f'This function can only take AD objects as inputs. Input of {type(ad_object).__name__} is not an AD object.')


    # TODO
    pass


def ln(ad_object):
    if not isinstance(ad_object, AD):
        raise TypeError(
            f'This function can only take AD objects as inputs. Input of {type(ad_object).__name__} is not an AD object.')


    # TODO
    pass
