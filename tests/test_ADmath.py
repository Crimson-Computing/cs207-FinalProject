import pytest
from autodiffcc.ADmath import *
from autodiffcc.core import AD


def test_cos():
    t1 = cos(AD(val=3, der=1))
    assert t1.val == pytest.approx(-0.9899924966004454)
    assert t1.der == pytest.approx(-0.1411200080598672)
    t2 = cos(AD(val=3, der=[1, 2]))
    assert t2.val == pytest.approx(-0.9899924966004454)
    assert t2.der.tolist() == [pytest.approx(-0.1411200080598672), pytest.approx(-0.2822400161197344)]
    t3 = cos(3)
    assert t3 == pytest.approx(-0.9899924966004454)


def test_sin():
    t1 = sin(AD(val=3, der=1))
    assert t1.val == pytest.approx(0.1411200080598672, 0.1)
    assert t1.der == pytest.approx(-0.9899924966004454)
    t2 = sin(AD(val=3, der=[1, 2]))
    assert t2.val == pytest.approx(0.1411200080598672, 0.1)
    assert t2.der.tolist() == [pytest.approx(-0.9899924966004454), pytest.approx(-1.9799849932008908)]
    t3 = sin(3)
    assert t3 == pytest.approx(0.1411200080598672, 0.1)


def test_tan():
    t1 = tan(AD(val=3, der=1))
    assert t1.val == pytest.approx(-0.1425465430742778)
    assert t1.der == pytest.approx(1.020319516942427)
    t2 = tan(AD(val=3, der=[1, 2]))
    assert t2.val == pytest.approx(-0.1425465430742778)
    assert t2.der.tolist() == [pytest.approx(1.020319516942427), pytest.approx(2.040639033884854)]
    t3 = tan(3)
    assert t3 == pytest.approx(-0.1425465430742778)


def test_exp():
    t1 = exp(AD(val=3, der=1))
    assert t1.val == pytest.approx(20.085536923187668)
    assert t1.der == pytest.approx(20.085536923187668)
    t2 = exp(AD(val=3, der=[1, 2]))
    assert t2.val == pytest.approx(20.085536923187668)
    assert t2.der.tolist() == [pytest.approx(20.08553692), pytest.approx(40.17107385)]
    t3 = exp(3)
    assert t3 == pytest.approx(20.085536923187668)


def test_sqrt():
    t1 = sqrt(AD(val=3, der=[1]))
    assert t1.val == pytest.approx(1.7320508075688772)
    assert t1.der == pytest.approx(0.28867513459481287)
    t2 = sqrt(AD(val=3, der=[1, 2]))
    assert t2.val == pytest.approx(1.7320508075688772)
    assert t2.der.tolist() == [pytest.approx(0.28867513459481287), pytest.approx(0.57735027)]
    t3 = sqrt(3)
    assert t3 == pytest.approx(1.7320508075688772)
    with pytest.raises(TypeError, match=r"Sqrt has a positive domain, input negative"):
        t4 = sqrt(-1)
    with pytest.raises(TypeError, match=r"Sqrt has a positive domain, input negative"):
        t5 = sqrt(AD(val=-1, der=[1, 2]))


def test_arcsin():
    t1 = arcsin(AD(val=0.3, der=[1]))
    assert t1.val == pytest.approx(0.30469265)
    assert t1.der == pytest.approx(1.04828484)
    t2 = arcsin(AD(val=0.3, der=[1, 2]))
    assert t2.val == pytest.approx(0.30469265)
    assert t2.der.tolist() == [pytest.approx(1.04828484), pytest.approx(1.1043152607484654)]
    t3 = arcsin(0.3)
    assert t3 == pytest.approx(0.30469265)
    with pytest.raises(ValueError):
        t4 = arcsin(-10)
    with pytest.raises(ValueError):
        t5 = arcsin(AD(val=-10, der=[1, 2]))


def test_arccos():
    t1 = arccos(AD(val=0.3, der=[1]))
    assert t1.val == pytest.approx(1.26610367)
    assert t1.der == pytest.approx(-1.04828484)
    t2 = arccos(AD(val=0.3, der=[1, 2]))
    assert t2.val == pytest.approx(1.26610367)
    assert t2.der.tolist() == [pytest.approx(-1.04828484), pytest.approx(-1.1043152607484654)]
    t3 = arccos(0.3)
    assert t3 == pytest.approx(1.26610367)
    with pytest.raises(ValueError):
        t4 = arccos(-10)
    with pytest.raises(ValueError):
        t5 = arccos(AD(val=-10, der=[1, 2]))


def test_arctan():
    t1 = arctan(AD(val=0.3, der=[1]))
    assert t1.val == pytest.approx(0.29145679)
    assert t1.der == pytest.approx(0.91743119)
    t2 = arctan(AD(val=0.3, der=[1, 2]))
    assert t2.val == pytest.approx(0.29145679)
    assert t2.der.tolist() == [pytest.approx(0.9174311926605504), pytest.approx(1.8348623853211008)]
    t3 = arctan(0.3)
    assert t3 == pytest.approx(0.2914567944778671)


def test_sinh():
    t1 = sinh(AD(val=0.3, der=[1]))
    assert t1.val == pytest.approx(0.30452029)
    assert t1.der == pytest.approx(1.04533851)
    t2 = sinh(AD(val=0.3, der=[1, 2]))
    assert t2.val == pytest.approx(0.30452029)
    assert t2.der.tolist() == [pytest.approx(1.0453385141), pytest.approx(2.090677028257721)]
    t3 = sinh(0.3)
    assert t3 == pytest.approx(0.30452029)


def test_cosh():
    t1 = cosh(AD(val=0.3, der=[1]))
    assert t1.val == pytest.approx(1.04533851)
    assert t1.der == pytest.approx(0.30452029)
    t2 = cosh(AD(val=0.3, der=[1, 2]))
    assert t2.val == pytest.approx(1.04533851)
    assert t2.der.tolist() == [pytest.approx(0.30452029), pytest.approx(0.6090405868942852)]
    t3 = cosh(0.3)
    assert t3 == pytest.approx(1.0453385141288605)


def test_tanh():
    t1 = tanh(AD(val=0.3, der=[1]))
    assert t1.val == pytest.approx(0.29131261)
    assert t1.der == pytest.approx(0.91513696)
    t2 = tanh(AD(val=0.3, der=[1, 2]))
    assert t2.val == pytest.approx(0.29131261)
    assert t2.der.tolist() == [pytest.approx(0.91513696), pytest.approx(1.8302739236532586)]
    t3 = tanh(0.3)
    assert t3 == pytest.approx(0.29131261)


def test_logistic():
    t1 = logistic(AD(val=0.3, der=[1]))
    assert t1.val == pytest.approx(0.57444252)
    assert t1.der == pytest.approx(0.24445831)
    t2 = logistic(AD(val=0.3, der=[1, 2]))
    assert t2.val == pytest.approx(0.57444252)
    assert t2.der.tolist() == [pytest.approx(0.24445831), pytest.approx(0.4889166233814918)]
    t3 = logistic(0.3)
    assert t3 == pytest.approx(0.57444252)


def test_log():
    t1 = log(AD(val=0.3, der=[1]))
    assert t1.val == pytest.approx(-1.2039728)
    assert t1.der == pytest.approx(3.33333333)
    t2 = log(AD(val=0.3, der=[1, 2]))
    assert t2.val == pytest.approx(-1.2039728)
    assert t2.der.tolist() == [pytest.approx(3.33333333), pytest.approx(6.666666666666667)]
    t3 = log(0.3)
    assert t3 == pytest.approx(-1.2039728)
