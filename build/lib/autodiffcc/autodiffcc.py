import numpy as np

class AD():
    """Create AD objects that allow for scalars and arrays with differentiation.
    
    ATTRIBUTES
    ==========
    val : the value of the AD object, can be scalar or array
    der : the derivative of the AD object, type should match val
    
    METHODS
    =======
    Overloads basic arithmetic operations. 
    
    EXAMPLES
    ========
    # >>> x = AD(val = 3, der = 1)
    # >>> f = 3 * x - 4
    # >>> print(f.val)
    # 5.0
    # >>> print(f.der)
    # 3.0
    """

    def __init__(self, val, der):
        self.val = np.array(val)
        self.der = np.array(der)

    def __pos__(self):
        """Returns the unary positive operator on self
        
        INPUTS
        =======
        self: AD object
        
        RETURNS
        ========
        self with unary positive operation applied
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> +x
        (3.0, 1.0)
        """
        return self

    def __neg__(self):
        """Returns the unary negative operator on self
        
        INPUTS
        =======
        self: AD object
        
        RETURNS
        ========
        self with unary negative operation applied
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> -x
        (-3.0, -1.0)
        """
        return AD(val = -self.val, der = -self.der)

    def __add__(self, other):
        """Returns the sum of self and other
        
        INPUTS
        =======
        self: AD object
        other: AD object or regular number/numpy array
        
        RETURNS
        ========
        self + other
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> x + 3
        (6.0, 1.0)
        """
        try:
            return AD(val = self.val + other.val, der = self.der + other.der)
        except AttributeError:
            return self + AD(val = other, der = np.zeros(self.der.shape))

    def __radd__(self, other):
        """Returns the reflected sum of self and other

        INPUTS
        =======
        self: AD object
        other: AD object or regular number/numpy array
        
        RETURNS
        ========
        other + self
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> 3 + x
        (6.0, 1.0)
        """
        return self.__add__(other)

    def __sub__(self, other):
        """Returns the difference of self and other
        
        INPUTS
        =======
        self: AD object
        other: AD object or regular number/numpy array
        
        RETURNS
        ========
        self - other
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> x - 3
        (0.0, 1.0)
        """
        try:
            return AD(val = self.val - other.val, der = self.der - other.der)
        except AttributeError:
            return self - AD(val = other, der = np.zeros(self.der.shape))

    def __rsub__(self, other):
        """Returns the reflected difference of self and other
        
        INPUTS
        =======
        self: AD object
        other: AD object or regular number/numpy array
        
        RETURNS
        ========
        other - self
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> 3 - x
        (0.0, -1.0)
        """
        try:
            return AD(val = other.val - self.val, der = other.der - self.der)
        except AttributeError:
            return AD(val = other, der = np.zeros(self.der.shape)) - self

    def __mul__(self, other):
        """Returns the product of self and other
        
        INPUTS
        =======
        self: AD object
        other: AD object or regular number/numpy array
        
        RETURNS
        ========
        self * other
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> x * 3
        (9.0, 3.0)
        """
        try:
            return AD(val = self.val * other.val, der = self.val*other.der + other.val*self.der)
        except AttributeError:
            return self * AD(val = other, der = np.zeros(self.der.shape))

    def __rmul__(self, other):
        """Returns the reflected product of self and other
        
        INPUTS
        =======
        self: AD object
        other: AD object or regular number/numpy array

        RETURNS
        ========
        other * self
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> 3 * x
        (9.0, 3.0)
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """Returns the quotient of self and other
        
        INPUTS
        =======
        self: AD object
        other: AD object or regular number/numpy array
        
        RETURNS
        ========
        self / other
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> x / 3
        (1.0, 0.3333333333333333)
        """
        try:
            return AD(val = self.val / other.val, 
                der = (other.val*self.der - self.val*other.der)/(other.val*other.val))
        except AttributeError:
            return self / AD(val = other, der = np.zeros(self.der.shape))

    def __rtruediv__(self, other):
        """Returns the reflected quotient of self and other

        INPUTS
        =======
        self: AD object
        other: AD object or regular number/numpy array
        
        RETURNS
        ========
        other / self
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> 3 / x
        (1.0, -0.3333333333333333)
        """
        try:
            return AD(val = other.val/self.val, 
                der = (self.val*other.der - other.val*self.der)/(self.val*self.val))
        except AttributeError:
            return AD(val = other, der = np.zeros(self.der.shape)) / self

    def __pow__(self, other):
        """Returns self to the power of other
        
        INPUTS
        =======
        self: AD object
        other: AD object or regular number/numpy array
        
        RETURNS
        ========
        self ** other
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> x ** 2
        (9.0, 6.0)
        """
        try:
            return AD(val = self.val ** other.val, 
                der = self.val**(other.val-1)*(self.val*other.der*np.log(self.val)+other.val*self.der))
        except AttributeError:
            return self ** AD(val = other, der = np.zeros(self.der.shape))

    def __rpow__(self, other):
        """Returns other to the power of self
        
        INPUTS
        =======
        self: AD object
        other: AD object or regular number/numpy array
        
        RETURNS
        ========
        other ** self
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> x ** 2
        (1.0, 0.3333333333333333)
        """
        try:
            return AD(val = other.val ** self.val, 
                der = other.val**(self.val-1)*(other.val*self.der*np.log(other.val)+self.val*other.der))
        except AttributeError:
            return AD(val = other, der = np.zeros(self.der.shape)) ** self

    def __eq__(self, other):
        # TODO:
        pass

    def __gt__(self, other):
        # TODO:
        pass

    def __lt__(self, other):
        # TODO:
        pass

    def __repr__(self):
        return str((self.val, self.der))