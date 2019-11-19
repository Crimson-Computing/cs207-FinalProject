import numpy as np
from autodiffcc.core import AD


def cos(self):
    
    if not isinstance(self, AD):
            raise TypeError("ADmath cos function takes only AD objects in")
    
    """Returns the cos of self 
        
    INPUTS
    =======
    AD object
        
    RETURNS
    ========
    Returns the cos of self 
        
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.cos(x) 
    (array(-0.9899924966004454), array(-0.1411200080598672))
    """
        
    return AD(val = np.cos(self.val), der = -np.sin(self.val))
    
    
def sin(self):
    
    if not isinstance(self, AD):
        raise TypeError("ADmath sin function takes only AD objects in")
    
    
    """Returns the sin of self 
        
    INPUTS
    =======
    AD object
        
    RETURNS
    ========
    Returns the sin of self 
        
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.sin(x) 
    (array(0.1411200080598672), array(-0.9899924966004454))
    """

    return AD(val = np.sin(self.val), der = np.cos(self.val))
    
    
    
def tan(self):
    
    if not isinstance(self, AD):
        raise TypeError("ADmath sin function takes only AD objects in")
    
    
    """Returns the tan of self 
        
    INPUTS
    =======
    AD object
        
    RETURNS
    ========
     Returns the tan of self
        
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.tan(x) 
    (array(-0.1425465430742778), array(1.020319516942427))
    """

    return AD(val = np.tan(self.val), der = 1 / (np.cos(self.val) ** 2))
    
    
def exp(self):

        
    if not isinstance(self, AD):
        raise TypeError("ADmath exp function takes only AD objects in")
    
    """Returns the exponent of self 
        
    INPUTS
    =======
    AD object
        
    RETURNS
    ========
    self to the exponent of self 
        
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.exp(x) 
    (array(20.085536923187668), array(20.085536923187668))
    """
    
    return AD(val = np.exp(self.val), der = self.der * np.exp(self.val))
    
    
def sqrt(self):

        
    if not isinstance(self, AD):
        raise TypeError("ADmath exp function takes only AD objects in")
    
    """Returns the sqrt of self 
        
    INPUTS
    =======
    AD object
        
    RETURNS
    ========
    self to the exponent of 1/2
        
    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> ADmath.sqrt(x) 
    (array(1.7320508075688772), array(0.28867513459481287))
    """
    return AD(self.val, self.der).__pow__(0.5)
