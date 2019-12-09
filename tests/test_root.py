import pytest
from autodiffcc.root import *
from autodiffcc.ADmath import *

# if isinstance(start_values, dict):
#     # turn keyword values into positional values matching function signature
#     values = []
#     for key in signature:
#         try:
#             values.append(start_values[key])
#         except KeyError:
#             raise KeyError(f"key {key} in function signature missing from start_values")
# elif isinstance(start_values, (list, np.ndarray)):
#     # if list-like
#     values = start_values
# else:
#     raise TypeError("Must include start_values as dict or list/array.")
# values = np.array(values)
# # check to make sure have correct number of variables
# if len(values) != len(signature):
#     raise KeyError("Incorrect number of variables passed in start_values.")

def test_root_bad_inputs():
    def f2var(x, y):
        return 2*x+y, x-1

    # Incorrect number of vars in start_values
    with pytest.raises(KeyError, match="Incorrect number of variables passed in start_values."):
        print(root(function=f2var, method='newton', start_values=2))

    # Function signature variable missing from dictionary keys
    with pytest.raises(KeyError, match="key y in function signature missing from start_values"):
        print(root(function=f2var, method='newton', start_values={'x': 2}))

    # Too many dictionary keys passed
    with pytest.raises(KeyError, match="Too many keys passed in start_values dictionary"):
        print(root(function=f2var, method='newton', start_values={'x': 2, 'y': 2, 'z':4}))

    # Incorrect type for start_values
    with pytest.raises(TypeError, match="Must include start_values as dict or list/array."):
        print(root(function=f2var, method='newton', start_values=f2var))

    # Incorrect number of variables
    with pytest.raises(KeyError, match="Incorrect number of variables passed in start_values."):
        print(root(function=f2var, method='newton', start_values=[1,2,3]))
    

