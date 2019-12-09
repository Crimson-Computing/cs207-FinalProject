import pytest
from autodiffcc.ADmath import *
from autodiffcc.core import AD, differentiate


def test_matrix_value():
    with pytest.raises(ValueError):
        t1 = AD(val=np.array([[1, 2], [2, 4]]), n_vars=1, idx=0)


def test_nvars_and_der():
    with pytest.raises(ValueError, match='Either specify der or n_vars and idx, but not both'):
        t1 = AD(val=3, der=1, n_vars=2)


def test_missing_der_missing_nvars_idx():
    # test missing n_vars
    with pytest.raises(KeyError):
        t1 = AD(val=3, idx=1)

    # test missing idx
    with pytest.raises(KeyError):
        t2 = AD(val=3, n_vars=2)

    # test declaring with only n_vars
    t3 = AD(val=3, n_vars=1)


def test_pos():
    t1 = +AD(val=3, der=1)
    assert t1.val == 3
    assert t1.der == 1
    t2 = +AD(val=-3, der=1)
    assert t2.val == -3
    assert t2.der == 1
    t3 = +AD(val=-3, der=[1, 2])
    assert t3.val == -3
    assert t3.der.tolist() == [1, 2]


def test_neg():
    t1 = -AD(val=3, der=1)
    assert t1.val == -3
    assert t1.der == -1
    t2 = -AD(val=-3, der=1)
    assert t2.val == 3
    assert t2.der == -1
    t3 = -AD(val=-3, der=[1, 2])
    assert t3.val == 3
    assert t3.der.tolist() == [-1, -2]


def test_add():
    t1 = AD(val=3, der=1) + 3
    assert t1.val == 6
    assert t1.der == 1
    t2 = AD(val=3, der=1) + AD(val=-4, der=1)
    assert t2.val == -1
    assert t2.der == 2
    t3 = AD(val=-4, der=[1, 2]) + 3
    assert t3.val == -1
    assert t3.der.tolist() == [1, 2]
    t4 = AD(val=3, der=[1, 2]) + AD(val=-8, der=[3, 2])
    assert t4 == -5
    assert t4.der.tolist() == [4, 4]


def test_radd():
    t1 = 3 + AD(val=3, der=1)
    assert t1.val == 6
    assert t1.der == 1
    t3 = -4 + AD(val=3, der=[1, 2])
    assert t3.val == -1
    assert t3.der.tolist() == [1, 2]


def test_sub():
    t1 = AD(val=3, der=1) - 3
    assert t1.val == 0
    assert t1.der == 1
    t2 = AD(val=3, der=1) - AD(val=4, der=1)
    assert t2.val == -1
    assert t2.der == 0
    t3 = AD(val=3, der=[1, 2]) - 4
    assert t3.val == -1
    assert t3.der.tolist() == [1, 2]
    t4 = AD(val=3, der=[1, 2]) - AD(val=-8, der=[3, 1])
    assert t4 == 11
    assert t4.der.tolist() == [-2, 1]


def test_rsub():
    t1 = 3 - AD(val=3, der=1)
    assert t1.val == 0
    assert t1.der == -1
    t3 = -4 - AD(val=3, der=[1, 2])
    assert t3.val == -7
    assert t3.der.tolist() == [-1, -2]


def test_mul():
    t1 = AD(val=3, der=1) * 3
    assert t1.val == 9
    assert t1.der == 3
    t2 = AD(val=3, der=1) * AD(val=4, der=1)
    assert t2.val == 12
    assert t2.der == 7
    t3 = AD(val=-3, der=[1, 2]) * (- 4)
    assert t3.val == 12
    assert t3.der.tolist() == [-4, -8]
    t4 = AD(val=3, der=[1, 2]) * AD(val=-8, der=[3, 1])
    assert t4 == -24
    assert t4.der.tolist() == [1, -13]


def test_rmul():
    t1 = 3 * AD(val=3, der=1)
    assert t1.val == 9
    assert t1.der == 3
    t3 = (-4) * AD(val=-3, der=[1, 2])
    assert t3.val == 12
    assert t3.der.tolist() == [-4, -8]


def test_truediv():
    t1 = AD(val=3, der=1) / 3
    assert t1.val == 1
    assert t1.der == pytest.approx(0.3333333333333333)
    t2 = AD(val=3, der=1) / AD(val=4, der=1)
    assert t2.val == pytest.approx(0.75)
    assert t2.der == pytest.approx(0.0625)
    t3 = AD(val=-4, der=[1, 2]) / (- 4)
    assert t3.val == 1
    assert t3.der.tolist() == [pytest.approx(-0.25), pytest.approx(-0.5)]
    t4 = AD(val=3, der=[1, 2]) / AD(val=-8, der=[3, 1])
    assert t4 == pytest.approx(-0.375)
    assert t4.der.tolist() == [-0.265625, -0.296875]


def test_rtruediv():
    t1 = 3 / AD(val=3, der=1)
    assert t1.val == 1
    assert t1.der == pytest.approx(-0.3333333333333333)
    t3 = -8 / AD(val=-4, der=[1, 2])
    assert t3.val == 2
    assert t3.der.tolist() == [pytest.approx(0.5), pytest.approx(1)]


def test_pow():
    t1 = AD(val=3, der=1) ** 2
    assert t1.val == 9
    assert t1.der == 6
    t2 = AD(val=3, der=1) ** AD(val=5, der=1)
    assert t2.val == 243
    assert t2.der == pytest.approx(671.96278615)
    t3 = AD(val=0, der=1) ** AD(val=5, der=1)
    assert t3.val == 0
    assert t3.der == pytest.approx(0.)
    t4 = AD(val=0, der=1) ** 2
    assert t4.val.tolist() == 0
    assert t4.der.tolist() == 0
    t5 = AD(val=-1, der=1) ** 2
    assert t5.val.tolist() == 1
    assert t5.der.tolist() == -2


def test_rpow():
    t1 = 2 ** AD(val=3, der=1)
    assert t1.val == 8
    assert t1.der == pytest.approx(5.54517744)
    t2 = 0 ** AD(val=3, der=1)
    assert t2.val == 0
    assert t2.der == 0
    t2 = 1 ** AD(val=0, der=1)
    assert t2.val == 1
    assert t2.der == 0

def test_eq():
    t1 = AD(val=3, der=1)
    assert t1 == 3
    t2 = AD(val=3, der=1)
    assert t1 == t2
    t3 = AD(val=3, der=[1, 2])
    assert t3 == 3
    t4 = AD(val=3, der=[1, 2])
    assert t4 == t3


def test_gt():
    t1 = AD(val=3, der=1)
    assert t1 > 2
    t2 = AD(val=2, der=1)
    assert t1 > t2
    t3 = AD(val=3, der=[1, 2])
    assert t3 > 2
    t4 = AD(val=4, der=[1, 2])
    assert t4 > t3


def test_ge():
    t1 = AD(val=3, der=1)
    assert t1 >= 2
    t2 = AD(val=3, der=1)
    assert t1 >= t2
    t3 = AD(val=3, der=[1, 2])
    assert t3 >= 2
    t4 = AD(val=3, der=[1, 2])
    assert t4 >= t3


def test_lt():
    t1 = AD(val=3, der=1)
    assert t1 < 4
    t2 = AD(val=4, der=1)
    assert t1 < t2
    t3 = AD(val=3, der=[1, 2])
    assert t3 < 4
    t4 = AD(val=2, der=[1, 2])
    assert t4 < t3


def test_le():
    t1 = AD(val=3, der=1)
    assert t1 <= 3
    t2 = AD(val=4, der=1)
    assert t1 <= t2
    t3 = AD(val=3, der=[1, 2])
    assert t3 <= 3
    t4 = AD(val=3, der=[1, 2])
    assert t4 <= t3


def test_combination():
    t1 = (AD(val=3, der=1) / 3 + 1) * 6 - 4
    assert t1.val == 8
    assert t1.der == 2


def test_differentiate_scalar_function():
    def f(x):
        return sin(3 * (x ** 2)) + tan(sqrt(x * 7))

    dfdx = differentiate(f)
    # test kwargs
    assert np.allclose(dfdx(x=5), 28.3316)
    assert np.allclose(dfdx(x=np.array([2,1,3])), np.array([11.4996,-4.2300,40.3201]))
    # test args
    assert np.allclose(dfdx(5), 28.3316)

def test_differentiate_scalar_function_multiple_inputs():
    def f(x, y):
        return sin(3 * (x ** 2)) + x * tan(sqrt(y * 7))

    dfdx = differentiate(f)
    assert np.allclose(dfdx(x=4, y=2), np.array([-14.6792, 5.49340]))
    assert np.allclose(dfdx(x=np.array([1, 2, 3]),
                            y=np.array([2, 1, 4])), np.array([-5.25572, 9.58533, -6.78778, 1.37335, 3.41987, 6.62502]))

def test_differentiate_vector_function():
    def f(x):
        f1 = cos(2 ** sin(x)) + 3 * x * sin(sqrt(x))
        f2 = x ** 3 - sin(x)
        return f1, f2

    dfdx = differentiate(f)
    assert np.allclose(dfdx(x=6), np.array([[-1.31673351], [107.03982971]]))
    assert np.allclose(dfdx(x=np.array([1, 3])), np.array([[2.6801, 3.2193], [2.4597, 27.990]]))


def test_differentiate_scalar_function_multiple_inputs():
    def f(x, y):
        f1 = sin(3 * (x ** 2)) + x * tan(sqrt(y * 7))
        f2 = y ** (3 * x) - sin(x)
        return f1, f2

    dfdx = differentiate(f)
    result1 = np.array([[-1.46792323e+01, 5.49340116e+00], [8.51804620e+03, 2.45760000e+04]])
    assert np.allclose(dfdx(x=4, y=2), result1)


def test_differentiate_args_issues():
    def f(x, y):
        f1 = sin(3 * (x ** 2)) + x * tan(sqrt(y * 7))
        f2 = y ** (3 * x) - sin(x)
        return f1, f2
    dfdx = differentiate(f)

    # test no args
    with pytest.raises(KeyError):
        dfdx()
    
    # test both args
    with pytest.raises(KeyError):
        dfdx(4, x=2)
    
    # test incorrect kwargs key
    with pytest.raises(KeyError):
        dfdx(x=2, z=2)

    # test incorrect number of keys
    with pytest.raises(KeyError):
        dfdx(x=2)

    # test too many keys
    with pytest.raises(KeyError):
        dfdx(x=2, y=3, z=3)

def test_differentiate_scalar_function_0_der():
    def f(x):
        return 2
    assert np.isclose(differentiate(f)(3), 0)

def test_differentiate_vector_function_0_der():
    def f(x, y):
        return 2, 3
    assert np.allclose(differentiate(f)(3,1), 0)

