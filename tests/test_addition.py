import pytest
import sys
from autodiffcc import addition
from autodiffcc import ADmath
from autodiffcc import AD

def test_adding():
    assert addition.adding(2, 2) == 4


