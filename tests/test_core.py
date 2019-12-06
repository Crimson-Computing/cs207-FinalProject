import pytest
import sys
from autodiffcc.core import *

def test_matrix_value():
    with pytest.raises(ValueError):
        t1 = AD(val = np.array([[1,2],[2,4]]), n_vars = 1, idx = 0)

def test_nvars_and_der():
    # test working n_vars and der
    t1 = ad.AD(val = 3, der = 1, n_vars = 1)
    t2 = ad.AD(val = np.array([3,1]), der = np.array([1,0]), n_vars = 2)
    
    # test scalar derivative
    with pytest.raises(ValueError):
        t3 = ad.AD(val = 3, der = 1, n_vars = 2)
    
    # test vector derivative
    with pytest.raises(ValueError):
        t4 = ad.AD(val = np.array([3,1]), der = 1, n_vars = 2)

def test_missing_der_missing_nvars_idx():
    # test missing n_vars
    with pytest.raises(KeyError):
        t1 = ad.AD(val = 3, idx = 1)

    # test missing idx
    with pytest.raises(KeyError):
        t2 = ad.AD(val = 3, n_vars = 2)

    # test declaring with only n_vars
    t3 = ad.AD(val = 3, n_vars = 1)

def test_pos():
    t1 = +AD(val = 3, der = 1)
    assert t1.val == 3
    assert t1.der == 1

def test_neg():
    t1 = -AD(val = 3, der = 1)
    assert t1.val == -3
    assert t1.der == -1

def test_add():
    t1 = AD(val = 3, der = 1) + 3
    assert t1.val == 6
    assert t1.der == 1
    t2 = AD(val = 3, der = 1) + AD(val = 4, der = 1)
    assert t2.val == 7
    assert t2.der == 2

def test_radd():
    t1 = 3 + AD(val = 3, der = 1)
    assert t1.val == 6
    assert t1.der == 1

def test_sub():
    t1 = AD(val = 3, der = 1) - 3
    assert t1.val == 0
    assert t1.der == 1
    t2 = AD(val = 3, der = 1) - AD(val = 4, der = 1)
    assert t2.val == -1
    assert t2.der == 0

def test_rsub():
    t1 = 3 - AD(val = 3, der = 1)
    assert t1.val == 0
    assert t1.der == -1

def test_mul():
    t1 = AD(val = 3, der = 1) * 3
    assert t1.val == 9
    assert t1.der == 3
    t2 = AD(val = 3, der = 1) * AD(val = 4, der = 1)
    assert t2.val == 12
    assert t2.der == 7

def test_rmul():
    t1 = 3 * AD(val = 3, der = 1)
    assert t1.val == 9
    assert t1.der == 3

def test_truediv():
    t1 = AD(val = 3, der = 1) / 3
    assert t1.val == 1
    assert t1.der == pytest.approx(0.3333333333333333)
    t2 = AD(val = 3, der = 1) / AD(val = 4, der = 1)
    assert t2.val == pytest.approx(0.75)
    assert t2.der == pytest.approx(0.0625)

def test_rsub():
    t1 = 3 / AD(val = 3, der = 1)
    assert t1.val == 1
    assert t1.der == pytest.approx(-0.3333333333333333)

def test_pow():
    t1 = AD(val = 3, der = 1) ** 2
    assert t1.val == 9
    assert t1.der == 6
    t2 = AD(val = 3, der = 1) ** AD(val = 5, der = 1)
    assert t2.val == 243
    assert t2.der == pytest.approx(671.96278615)
    t3 = AD(val = 0, der = 1) ** AD(val = 5, der = 1)
    assert t2.val == 0
    assert t2.der == pytest.approx(0.)

def test_rpow():
    t1 = 2 ** AD(val = 3, der = 1)
    assert t1.val == 8
    assert t1.der == pytest.approx(5.54517744)

def test_eq():
    t1 = AD(val = 3, der = 1)
    assert t1 == 3
    t2 = AD(val = 3, der = 1)
    assert t1 == t2

def test_gt():
    t1 = AD(val = 3, der = 1)
    assert t1 > 2
    t2 = AD(val = 2, der = 1)
    assert t1 > t2

def test_ge():
    t1 = AD(val = 3, der = 1)
    assert t1 >= 2
    t2 = AD(val = 3, der = 1)
    assert t1 >= t2

def test_lt():
    t1 = AD(val = 3, der = 1)
    assert t1 < 4
    t2 = AD(val = 4, der = 1)
    assert t1 < t2

def test_le():
    t1 = AD(val = 3, der = 1)
    assert t1 <= 3
    t2 = AD(val = 3, der = 1)
    assert t1 <= t2

def test_combination():
    t1 = (AD(val = 3, der = 1) / 3  + 1) * 6 - 4
    assert t1.val == 8
    assert t1.der == 2
