import pytest
import sys
from autodiffcc import addition
from autodiffcc import ADmath
from autodiffcc import AD

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

def test_combination():
    t1 = (AD(val = 3, der = 1) / 3  + 1) * 6 - 4
    assert t1.val == 8
    assert t1.der == 2
