import inspect
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

    def __init__(self, val, **kwargs):
        val = np.array(val).astype(float)
        # raise ValueError if val is a matrix
        if len(val.shape) > 1:
            raise ValueError("val must :be a scalar or vector, cannot be a matrix")
        if 'der' in kwargs:
            self.der = np.array(kwargs['der']).astype(float)
            # check if user specifies n_vars and der, they should match
            if 'n_vars' in kwargs:
                if len(self.der.shape) == 0:
                    if kwargs['n_vars'] > 1:
                        raise ValueError('n_vars does not match shape of der')
                elif kwargs['n_vars'] != self.der.shape[0]:
                    raise ValueError('n_vars does not match shape of der')
        else:
            # if number of variables is 1
            if kwargs['n_vars'] == 1:
                idx = 0
            elif 'n_vars' not in kwargs or 'idx' not in kwargs:
                raise KeyError("If der isn't specified, need to specify n_vars and idx")
            else:
                idx = kwargs['idx']
            n_vars = kwargs['n_vars']
            if len(val.shape) == 0:
                self.der = np.zeros(n_vars)
                self.der[idx] = 1.0
            else:
                self.der = np.zeros((n_vars, val.shape[0]))
                self.der[idx,:] = 1.0
        self.val = val
        
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
            if (self.val).all() == 0:
                return AD(val = self.val ** other.val, 
                    der = self.val ** (other.val - 1) * (self.val * other.der + other.val * self.der))
            else:
                return AD(val = self.val ** other.val, 
                    der = self.val ** (other.val - 1) * (self.val * other.der * np.log(np.abs(self.val)) + other.val * self.der))
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
            if (self.val).all() == 0:
                return AD(val = other.val ** self.val, 
                    der = other.val ** (self.val - 1) * (other.val * self.der + self.val * other.der))
            else:
                return AD(val = other.val ** self.val, 
                    der = other.val ** (self.val - 1) * (other.val * self.der * np.log(np.abs(other.val)) + self.val * other.der))
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
        return (self < other) or (self == other)

    def __repr__(self):
        return str((self.val, self.der))

def differentiate(base_func):
    """Returns a function that takes as input a value and returns the derivative of 
    base_func evaluated at the value
    
    INPUTS
    =======
    base_func: a function that uses autodiffcc math functions to create an output
    
    RETURNS
    ========
    a function that takes as input a value and returns the derivative of base_func
        evaluated at the value
    
    NOTES
    =====
    PRE: 
         - if a scalar function, base_func returns a scalar
         - if a vector function, base_func returns tuple, list, or numpy array

    EXAMPLES
    =========
    >>> def f(x):
    ...     return 3*(x**2)
    >>> dfdx = differentiate(f)
    >>> dfdx(x=5)
    30.0
    """
    def base_func_der(**kwargs):
        signature = inspect.signature(base_func).parameters
        n_vars = len(signature)
        n_keys = len(kwargs.keys())
        # check that kwargs and function signature have same values
        if n_keys != n_vars:
            raise KeyError("Length of **kwargs and base function signature do not match.")
        var_to_AD_obj = {}
        for i, key in enumerate(kwargs.keys()):
            if key not in signature:
                raise KeyError("**kwargs key {} missing from base function signature.".format(key))
            # add key to variable
            var_to_AD_obj[key] = AD(kwargs[key], n_vars = n_vars, idx = i)
        
        # run base_func on input values now keeping track of derivative
        result = base_func(**var_to_AD_obj)

        # if base_func is a scalar function, return 1-D flat derivative (combining multiple vector-valued inputs)
        if type(result) == AD:
            return result.der.flatten()
        
        # if base_func is vector function, return 2-D Jacobian where each row is f1, f2, ...
        n_fn_dim = len(result)
        final_der = []
        for ad_obj in result:
            final_der.append(ad_obj.der.flatten())

        return np.array(final_der)

    return base_func_der
