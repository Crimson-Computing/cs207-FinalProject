import pytest
from autodiffcc import ADmath
from autodiffcc.core import AD


def test_cos():
    t1 = ADmath.cos(AD(val=3, der=1))
    assert t1.val == pytest.approx(-0.9899924966004454)
    assert t1.der == pytest.approx(-0.1411200080598672)
    with pytest.raises(TypeError, match=r".* This function can only take AD"
                                         r" objects as inputs .*"):
        t2 = ADmath.cos(1)


def test_sin():
    t1 = ADmath.sin(AD(val=3, der=1))
    assert t1.val == pytest.approx(0.1411200080598672, 0.1)
    assert t1.der == pytest.approx(-0.9899924966004454)
    with pytest.raises(TypeError, match=r".* This function can only take AD"
                                         r" objects as inputs .*"):
        t2 = ADmath.sin(1)


def test_tan():
    t1 = ADmath.tan(AD(val=3, der=1))
    assert t1.val == pytest.approx(-0.1425465430742778)
    assert t1.der == pytest.approx(1.020319516942427)
    with pytest.raises(TypeError, match=r".* This function can only take AD"
                                         r" objects as inputs .*"):
        t2 = ADmath.tan(1)


def test_exp():
    t1 = ADmath.exp(AD(val=3, der=1))
    assert t1.val == pytest.approx(20.085536923187668)
    assert t1.der == pytest.approx(20.085536923187668)
    with pytest.raises(TypeError, match=r".* This function can only take AD"
                                         r" objects as inputs .*"):
        t2 = ADmath.exp(1)


def test_sqrt():
    t1 = ADmath.sqrt(AD(val=3, der=1))
    assert t1.val == pytest.approx(1.7320508075688772)
    assert t1.der == pytest.approx(0.28867513459481287)
    with pytest.raises(TypeError, match=r".* This function can only take AD"
                                         r" objects as inputs .*"):
        t2 = ADmath.sqrt(1)
