import inspect
import numpy as np
from autodiffcc.core import differentiate


def _newton_fourier(function, start_interval_posvars: list, end_interval_posvars: list, threshold, max_iter):

    signature = inspect.signature(function).parameters.keys()
    n_vars_function = len(signature)

    if not (start_interval_posvars and end_interval_posvars):
        raise KeyError("Must include both start_interval_posvars and end_interval_posvars.")

    # using positional variables
    x_vars = np.array(start_interval_posvars)
    z_vars = np.array(end_interval_posvars)

    if not (x_vars.shape == z_vars.shape):
        raise KeyError("start_interval_posvars and end_interval_posvars must be the same shape.")

    if not (n_vars_function in x_vars.shape):
        raise KeyError("start_interval_posvars and end_interval_posvars must have the same number of variables as the function.")

    jacobian = differentiate(function)

    # Starting values for x_0, z_0
    flat_x = x_vars.flatten()
    flat_z = z_vars.flatten()
    limit_numerator = (x_vars.flatten() - z_vars.flatten())**2

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

        print(limit)

        x_vars = flat_x.reshape(x_vars.shape)
        z_vars = flat_z.reshape(z_vars.shape)

        if limit.all() < threshold:
            return x_vars, z_vars

    raise Exception()

    Warning("Maximum number of iterations exceeded before converging.")


def test_func_2d(x, y):
    return x + 2 * y, x * y + 2


start = [-3, 2]
end = [0, 0]

print(_newton_fourier(test_func_2d, start_interval_posvars=start, end_interval_posvars=end, threshold=1e-8, max_iter=2000))
