import inspect
import numpy as np
from autodiffcc.core import AD, differentiate


def _norm(vector):
    """Returns the absolute value norm of the input vector

    INPUTS
    =======
    vector: numpy array or list-like

    RETURNS
    ========
    the absolute value norm of vector

    EXAMPLES
    =========
    >>> x = np.array([1,2,3])
    >>> _norm(x)
    6.
    """
    return np.sum(np.abs(vector))


def _newton_raphson(function, values, threshold, max_iter):
    """#TODO
    """
    jacobian = differentiate(function)

    for i in range(max_iter):
        flat_variables = values.flatten()
        if len(flat_variables) == 1:
            flat_variables = flat_variables - function(*values) / jacobian(*values)
        else:
            flat_variables = flat_variables - np.matmul(np.linalg.pinv(jacobian(*values)), function(*values))
        values = flat_variables.reshape(values.shape)
        if _norm(function(*values)) < threshold:
            return values
    raise Exception()

    Warning("Maximum number of iterations exceeded before converging.")


def _method_2(function, start, threshold):
    pass


def _method_3(function, start, threshold):
    pass


def root(function, method, start_values, threshold=1e-8, max_iter=2000):
    """Returns the root of the function found using the specified method

    INPUTS
    =======
    function: a function using ADmath methods
    start: the starting point of the root-finding method (scalar or vector)
    method: the method to do root-finding   #TODO: add list of possible methods
    threshold: the minimum threshold for finding a root
    max_iter: maximum number of iterations that algorithm will look for before quitting
        - in case method does not converge

    RETURNS
    ========
    the root of the function found using the specified method

    EXAMPLES
    =========
    >>> x = AD(val = 3, der = 1)
    >>> 2 ** x
    (8.0, 5.54517744)
    """
    # process variable inputs
    signature = inspect.signature(function).parameters.keys()

    if isinstance(start_values, dict):
        # turn keyword values into positional values matching function signature
        values = []
        for key in signature:
            try:
                values.append(start_values[key])
            except KeyError:
                raise KeyError(f"key {key} in function signature missing from start_values")
        if len(start_values.keys()) != len(signature):
            raise KeyError("Too many keys passed in start_values dictionary")
    elif isinstance(start_values, (list, np.ndarray)):
        # if list-like
        values = start_values
    elif np.isscalar(start_values):
        values = [start_values]
    else:
        raise TypeError("Must include start_values as dict or list/array.")
    values = np.array(values)
    # check to make sure have correct number of variables
    if len(values) != len(signature):
        raise KeyError("Incorrect number of variables passed in start_values.")

    # find roots
    if method.lower() in ['newton', 'newton-raphson', 'n-r']:
        return _newton_raphson(function, values, threshold, max_iter)
