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

    # gives our operators priority over numpy's methods
    __array_priority__ = 2

    def __init__(self, val, der):
        self.val = np.array(val)
        self.der = np.array(der)
        if self.val.shape != self.der.shape:
            raise(Exception('The shape of val and der do not match: val is {}, der is {}.'.format(self.val.shape, self.der.shape)))

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
        >>> 2 ** x
        (8.0, 5.54517744)
        """
        try:
            return AD(val = other.val ** self.val, 
                der = other.val**(self.val-1)*(other.val*self.der*np.log(other.val)+self.val*other.der))
        except AttributeError:
            return AD(val = other, der = np.zeros(self.der.shape)) ** self

    def __eq__(self, other):
        """Returns True if self and other have the same value
        
        INPUTS
        =======
        self: AD object
        other: AD object or regular number/numpy array
        
        RETURNS
        ========
        self == other
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> x == 3
        True
        >>> y = AD(val = 3, der = 2)
        >>> x == y
        True
        """
        try:
            return self.val == other.val
        except AttributeError:
            return self.val == other

    def __gt__(self, other):
        """Returns True if self's value is greater than other's value
        
        INPUTS
        =======
        self: AD object
        other: AD object or regular number/numpy array
        
        RETURNS
        ========
        self > other
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> x > 2
        True
        >>> y = AD(val = 3, der = 2)
        >>> x > y
        False
        """
        try:
            return self.val > other.val
        except AttributeError:
            return self.val > other

    def __ge__(self, other):
        """Returns True if self's value is greater than or equal to other's value
        
        INPUTS
        =======
        self: AD object
        other: AD object or regular number/numpy array
        
        RETURNS
        ========
        self >= other
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> x >= 2
        True
        >>> y = AD(val = 3, der = 2)
        >>> x >= y
        True
        """
        return (self > other) or (self == other)

    def __lt__(self, other):
        """Returns True if self's value is less than other's value
        
        INPUTS
        =======
        self: AD object
        other: AD object or regular number/numpy array
        
        RETURNS
        ========
        self < other
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> x < 2
        False
        >>> y = AD(val = 4, der = 2)
        >>> x < y
        True
        """
        try:
            return self.val < other.val
        except AttributeError:
            return self.val < other

    def __le__(self, other):
        """Returns True if self's value is less than or equal to other's value
        
        INPUTS
        =======
        self: AD object
        other: AD object or regular number/numpy array
        
        RETURNS
        ========
        self <= other
        
        EXAMPLES
        =========
        >>> x = AD(val = 3, der = 1)
        >>> x <= 3
        True
        >>> y = AD(val = 2, der = 2)
        >>> x <= y
        False
        """
        return (self > other) or (self == other)

    def __repr__(self):
        return str((self.val, self.der))