import numpy as np
from autodiffcc.root import root


def solver(base_function, start, threshold, max_iter, method='newton-raphson', **kwargs):
    """Returns the solution to a system of equations using the specific root-finding method

    INPUTS
    =======
    function: a function using ADmath methods
    start: the starting point of the root-finding method (scalar or vector)
    method: the method to do root-finding   # TODO: add list of possible methods
    threshold: the minimum threshold for finding a root
    max_iter: maximum number of iterations that algorithm will look for before quitting
        - in case method does not converge

    RETURNS
    ========
    The root of the function found using the specified method

    EXAMPLES
    =========
    # TODO: Write Examples
    """

    result = root(function=base_function, start=start, method=method, threshold=threshold, max_iter=max_iter, **kwargs)

    return result

'''
def solver(lhs, rhs, start, method='newton-raphson', **kwargs):
    def zero(lhs, rhs, **kwargs):
        return np.array(lhs(**kwargs)) - np.array(rhs(**kwargs))

    base_function = zero(lhs=lhs, rhs=rhs, **kwargs)
    print(base_function(**kwargs))
    result = root(function=base_function, start=start, method=method, threshold=1e-6)
    return result
'''