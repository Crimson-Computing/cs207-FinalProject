import pytest
from autodiffcc.ADmath import *
from autodiffcc.core import AD
from autodiffcc.parser import expressioncc

def test_log():
    fn = expressioncc('3 * log(x,2) + sin(7)', ['x']).get_fn()
    t1 = fn(AD(4, n_vars=1))
    assert t1.val == pytest.approx(6.6569866)
    assert t1.der == pytest.approx(1.08202128)

def test_equation():
    fn = expressioncc('3 * log(x,2) = - sin(7)', ['x']).get_fn()
    t1 = fn(AD(4, n_vars=1))
    assert t1.val == pytest.approx(6.6569866)
    assert t1.der == pytest.approx(1.08202128)  

def test_multi_var():
    fn = expressioncc('log(x,2) + sin(y)', ['x', 'y']).get_fn()
    x = AD(4, der = [1, 0])
    y = AD(3, der = [0, 1])
    t1 = fn(x,y)
    assert t1.val == pytest.approx(2.1411200080598674)
    assert t1.der.tolist() == [pytest.approx(0.36067376), pytest.approx(-0.9899925)] 