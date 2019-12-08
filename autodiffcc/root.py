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


def _newton_raphson(function, start_posvars: list, start_kwvars: dict, threshold, max_iter):
    """#TODO
    """
    signature = inspect.signature(function).parameters.keys()
    n_vars_function = len(signature)

    if start_posvars and start_kwvars:
        raise KeyError("Cannot pass both start_posvars and start_kwvars. Must only choose one.")
    elif not (start_posvars or start_kwvars):
        raise KeyError("Must include either start_posvars or start_kwvars.")
    elif start_kwvars:
        # turn keyword variables into positional variables matching function signature
        variables = []
        for key in signature:
            try:
                variables.append(start_kwvars[key])
            except KeyError:
                raise KeyError(f"key {key} in function signature missing from start_kwvars")
    else:
        # using positional variables
        variables = start_posvars
    variables = np.array(variables)

    jacobian = differentiate(function)

    for i in range(max_iter):
        flat_variables = variables.flatten()
        if len(flat_variables) == 1:
            flat_variables = flat_variables - function(*variables) / jacobian(*variables)
        else:
            flat_variables = flat_variables - np.matmul(np.linalg.pinv(jacobian(*variables)), function(*variables))
        variables = flat_variables.reshape(variables.shape)
        if _norm(function(*variables)) < threshold:
            return variables
    raise Exception()

    Warning("Maximum number of iterations exceeded before converging.")


def _method_2(function, start, threshold):
    pass


def _method_3(function, start, threshold):
    pass


def root(function, method, start_posvars: list = None, start_kwvars: dict = None, threshold=1e-8, max_iter=2000):
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
    if method.lower() in ['newton', 'newton-raphson', 'n-r']:
        return _newton_raphson(function, start_posvars, start_kwvars, threshold, max_iter)
