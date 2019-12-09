import inspect
import matplotlib.pyplot as plt
import numpy as np
import itertools
from autodiffcc.core import differentiate


def _check_start_values(start_values, signature):
    """ A service that checks the shape of start_values and reformats them for the specific method
    INPUTS
    =======
    start_values: List or dictionary of start values provided to the root finder
    signature: Signature of the function

    RETURNS
    ========
    Returns an array of values to pass to the specific root finding method
    """
    if not start_values:
        raise ValueError("Must include start_values as dict or list/array for this method.")

    if isinstance(start_values, dict):
        # turn keyword values into positional values matching function signature
        values = []
        for key in signature:
            try:
                values.append(start_values[key])
            except KeyError:
                raise KeyError(f"key {key} in function signature missing from start_values.")
        if len(start_values.keys()) != len(signature):
            raise KeyError("Too many keys passed in start_values dictionary.")
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

    return values


def _check_interval(interval, signature):
    """ A service that checks the shape of interval and reformats it for the specific method

    INPUTS
    =======
    interval: List/array or a list of two dictionaries [interval_start, interval_end] provided to the root finder
    signature: Signature of the function

    RETURNS
    ========
    Returns two arrays interval_start, interval_end to pass to the specific root finding method
    """

    if interval is None:
        raise ValueError("Must provide interval for this method.")

    elif not isinstance(interval[0], type(interval[1])):
        raise TypeError("Must include interval as list of two dicts or numeric list/array.")

    elif isinstance(interval[0], dict) and isinstance(interval[1], dict):
        # turn keyword values into positional values matching function signature
        interval_start_values = []
        start_dict = interval[0]

        interval_end_values = []
        end_dict = interval[1]

        if len(start_dict.keys()) != len(end_dict.keys()):
            raise KeyError("The interval_start and interval_end dictionaries must have the same number of keys.")

        for key in signature:
            try:
                interval_start_values.append(start_dict[key])
                interval_end_values.append(end_dict[key])
            except KeyError:
                raise KeyError(f"key {key} in function signature missing from interval dictionary.")
        if len(start_dict.keys()) != len(signature):
            raise KeyError("Too many keys passed in interval dictionary.")

        interval_start = np.asarray(interval_start_values)
        interval_end = np.asarray(interval_end_values)

    elif isinstance(interval, (list, np.ndarray)):
        # if list-like
        values = np.asarray(interval)
        interval_start = values[:, 0]
        interval_end = values[:, 1]

    else:
        raise TypeError("Must include interval as list of two dicts or list/array.")

    # check to make sure have correct number of variables
    if len(interval_start) != len(signature):
        raise KeyError("Incorrect number of variables passed in interval.")

    # interval_start = interval_start.reshape(-1,1)
    # interval_end = interval_end.reshape(-1,1)

    return interval_start, interval_end


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


def _bisect(function, interval_start, interval_end, max_iter, signature):
    """Performs the bisection method on a function of one or more variables to find a root

    INPUTS
    =======
    function: A function defined using the autodiffcc.ADmath methods
    interval_start: The start of the initial interval of values on which to attempt to find a root, as an array
    interval_end: The end of the initial interval of values on which to attempt to find a root, as an array
    threshold: Minimum threshold to declare convergence on a root
    signature: The signature of the function
    
    RETURNS
    ========
    An approximate root of function found starting from the interval or raised Exception if none are found


    NOTES
    =====
    This function considers a value to be 0 if round(value, 15) == 0)
    This function does not find all the roots. You can change the intervals to
    look for different roots.

    EXAMPLES
    =========

    >>> def f(x, y):
    >>>    return(2*x*y - 2)

    >>> _bisect(f, interval_start = [-10, 10], interval_end = [-4, 10])
    [0.12999772724065328, 7.692442177460448]
    """

    print("Please note that this function uses approximations")

    # check how many parameters there are in the function
    nParam = len(signature)

    # create a point array storing coordinates

    ## example of points
    #    start-interval finish-interval
    # x       1               2
    # y       3               10
    # z      -2               0

    points = np.c_[interval_start, interval_end]

    #### PLOT if can do 2D graph
    if nParam == 1:
        # x axis values
        a = interval_start
        b = interval_end
        interval = np.linspace(a, b, num=1000)
        # corresponding y axis values
        values = function(interval)
        # plotting the points
        plt.plot(interval, values, color='green', linestyle='dashed', linewidth=0.5,
                 marker='o', markerfacecolor='blue', markersize=1)
        # naming the x axis
        plt.xlabel(' Interval ')
        # naming the y axis
        plt.ylabel(' Values ')
        plt.axhline(y=0)
        # giving a title to my graph
        plt.title('Graphs function in the specified interval')
        # function to show the plot
        plt.show()

        # get the starting points of each variable
    # get the ending points for each variable
    # get their combinations. Should be 4 different combinations

    matrix = np.empty((nParam, 2))
    for p in range(0, nParam):
        matrix[p] = points[p]
    allpoints = list(itertools.product(*matrix))

    # check sign change
    results = []
    for elements in allpoints:
        results.append(function(*elements))
        asign = np.sign(results)
        # detect sign change
        signchange = sum(((np.roll(asign, 1) - asign) != 0).astype(int)) > 0

    # if signs are not different
    if not signchange:
        raise Exception("No change in sign, please try different intervals")

    if signchange:
        print("Root between in the specified intervals")
        i = 1
        # if signs are different
        while signchange:
            print("----------------------")
            print("iteration", i)

            i = i + 1

            # middle points
            c = []
            for k in range(0, nParam):
                c.append((points[k][0] + points[k][1]) / 2)

            middlePointResult = function(*c)

            # approx to 14ths decimal point
            # if found root:
            if (round(middlePointResult, 15) == 0):
                print("root found for ", c)
                return (c)  # return middle as the approximate root value
            # if did not find root yet:
            else:
                j = 0

                for n in results:
                    if (n * middlePointResult < 0):
                        print("root between", c, "and", allpoints[j])
                        corner1 = list(allpoints[j])
                        corner2 = c
                    j = j + 1
                print("choosing to look in the area between", corner1, "and", corner2)

                # update points for intervals
                if corner1 > corner2:
                    points = np.c_[corner2, corner1]
                else:
                    points = np.c_[corner1, corner2]

                points = np.asarray(points)

                matrix = np.empty((nParam, 2))
                for p in range(0, nParam):
                    matrix[p] = points[p]
                allpoints = list(itertools.product(*matrix))

                # check sign change
                results = []
                for elements in allpoints:
                    results.append(function(*elements))
                    asign = np.sign(results)
                    # detect sign change
                    signchange = sum(((np.roll(asign, 1) - asign) != 0).astype(int)) > 0

            if i >= max_iter:
                break


def _newton_raphson(function, values, threshold, max_iter):

    """Returns a root found starting from values using the Newton-Raphson method

    INPUTS
    =======
    function: A function defined using the autodiffcc.ADmath methods
    values: Starting point for root-finding method as a scalar or vector
    threshold: Minimum threshold to declare convergence on a root
    max_iter: Maximum number of iterations taken for the algorithm to converge

    RETURNS
    ========
    A root of function found starting from values or raised Exception if none are found

    EXAMPLES
    =========
    >>> _newton_raphson(lambda x: x, 3, 1e-8, 2000)
    0.
    """
    jacobian = differentiate(function)
    output_shape = len(np.array(function(*values)).flatten())

    for i in range(max_iter):
        flat_variables = values.flatten()
        if len(flat_variables) == 1 and output_shape == 1:
            flat_variables = flat_variables - function(*values) / jacobian(*values)
        else:
            flat_variables = flat_variables - np.matmul(np.linalg.pinv(jacobian(*values)), function(*values))
        values = flat_variables.reshape(values.shape)
        if _norm(function(*values)) < threshold:
            return values
    raise Exception("Newton-Raphson did not converge, try increasing max_iter.")


def _newton_fourier(function, interval_start: np.ndarray, interval_end: np.ndarray, threshold, max_iter):
    """Returns a root of the function found using the Newton-Fourier algorithm

    INPUTS
    =======
    function: A function defined using the autodiffcc.ADmath methods
    interval_start: The start of the initial interval of values on which to attempt to find a root, as an array
    interval_end: The end of the initial interval of values on which to attempt to find a root, as an array
    threshold: Minimum threshold to declare convergence on a root
    max_iter: Maximum number of iterations taken for the algorithm to converge
    
    RETURNS
    ========
    A root of function found (approximately, within the threshold) starting from the interval or raised Exception if none are found

    EXAMPLES
    =========
    >>> def f(x, y):
    >>>     return 2 * x + y - 2, y+2
    >>> interval_start = [3, -3]
    >>> interval_end = [3, -3]
    >>> my_root = _newton_fourier(f, interval_start=interval, interval_end=interval_end, threshold=1e-8, max_iter=2000)
    >>> print(my_root)
    [ 2. -2.]    """
    # using positional variables
    x_vars = interval_start
    z_vars = interval_end

    # Starting values for x_0, z_0
    flat_x = x_vars.flatten()
    flat_z = z_vars.flatten()
    limit_numerator = (x_vars.flatten() - z_vars.flatten())**2

    jacobian = differentiate(function)

    for i in range(max_iter):
        if len(flat_x) == 1:
            common_jacobian = jacobian(*x_vars)
            flat_x = flat_x - function(*x_vars) / common_jacobian
            flat_z = flat_z - function(*z_vars) / common_jacobian
        else:
            common_jacobian = np.linalg.pinv(jacobian(*x_vars))
            flat_x = flat_x - np.matmul(common_jacobian, function(*x_vars))
            flat_z = flat_z - np.matmul(common_jacobian, function(*z_vars))

        limit_denominator = limit_numerator
        limit_numerator = flat_x - flat_z
        limit = limit_numerator/limit_denominator

        x_vars = flat_x.reshape(x_vars.shape)
        z_vars = flat_z.reshape(z_vars.shape)

        if limit.all() < threshold:
            return np.mean(np.vstack([x_vars, z_vars]), axis=0)

    raise Exception("Newton-Fourier did not converge, try increasing max_iter.")


def find_root(function, start_values=None, interval=None, method='newton-raphson', threshold=1e-8, max_iter=2000):
    """Returns the root of a function defined using the autodiffcc.ADmath methods

    INPUTS
    =======
    function: A function defined using the autodiffcc.ADmath methods
    start_values: Starting point for root-finding method as a scalar or vector; used only in newton-raphson
    interval: Initial interval of values on which to attempt to find a root, either as an array or a list of
              two dicts; used only in newton-fourier and bisection
    method: Root-finding algorithm to use ['newton-raphson', 'newton-fourier', 'bisection']
    threshold: Minimum threshold to declare convergence for the newton-raphson and newton-fourier methods
    max_iter: Maximum number of iterations taken for the algorithm to converge

    RETURNS
    ========
    An root of the function found or raised Exception if no root is found. If a function of multiple variables,
    this is is returned as an array where the values correspond with the variable positions supplied to find_root.

    EXAMPLES
    =========
    >>> def f(x, y):
    >>>     return 2 * x + y - 2, y + 2
    >>> interval = [[3, -3], [3, -3]]
    >>> my_root = find_root(f, interval=interval, method='newton-fourier', threshold=1e-8, max_iter=2000)
    >>> print(my_root)
    [ 2. -2.]

    >>> find_root(lambda x: x+2, 2, method='newton')
    -2.
    """
    # process variable inputs
    signature = inspect.signature(function).parameters.keys()

    # find roots
    if method.lower() in ['newton', 'newton-raphson', 'n-r']:
        values = _check_start_values(start_values=start_values, signature=signature)
        return _newton_raphson(function, values, threshold, max_iter)

    if method.lower() in ['bisect', 'bisection', 'b']:
        interval_start, interval_end = _check_interval(interval=interval, signature=signature)
        return _bisect(function, interval_start, interval_end, max_iter, signature)

    if method.lower() in ['newton-fourier', 'n-f']:
        interval_start, interval_end = _check_interval(interval=interval, signature=signature)
        return _newton_fourier(function, interval_start, interval_end, threshold, max_iter)
