import pytest
from autodiffcc.root import *
from autodiffcc.ADmath import *

def test_root_bad_inputs():
    def f2var(x, y):
        return 2*x+y, x-1

    # Incorrect number of vars in start_values
    with pytest.raises(KeyError, match="Incorrect number of variables passed in start_values."):
        find_root(function=f2var, method='newton', start_values=2)

    # Function signature variable missing from dictionary keys
    with pytest.raises(KeyError, match="key y in function signature missing from start_values"):
        find_root(function=f2var, method='newton', start_values={'x': 2})

    # Too many dictionary keys passed
    with pytest.raises(KeyError, match="Too many keys passed in start_values dictionary"):
        find_root(function=f2var, method='newton', start_values={'x': 2, 'y': 2, 'z':4})

    # Incorrect type for start_values
    with pytest.raises(TypeError, match="Must include start_values as dict or list/array."):
        find_root(function=f2var, method='newton', start_values=f2var)

    # Incorrect number of variables
    with pytest.raises(KeyError, match="Incorrect number of variables passed in start_values."):
        find_root(function=f2var, method='newton', start_values=[1,2,3])

def test_newton_raphson_scalar():
    def f1var(x):
        return (x + 2) * (x - 3)
    this_root = find_root(function=f1var, method='newton', start_values=1)
    assert np.isclose(this_root, 3.0)

def test_newton_raphson_vector():
    def f2var(x, y):
        return 2*x+y, x-1
    this_root = find_root(function=f2var, method='newton', start_values=[1,2])
    assert np.allclose(this_root, np.array([1,-2]))

def test_newton_raphson_no_solution_scalar():
    def f1var(x):
        return x**2 + 1
    with pytest.raises(Exception, match="Newton-Raphson did not converge, try increasing max_iter."):
        find_root(function=f1var, method='newton', start_values=1)



