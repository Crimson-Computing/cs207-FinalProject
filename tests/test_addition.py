import pytest
import sys
sys.path.append('../autodiffcc')
import addition
import ADmath
from autodiffcc import AD

def test_adding():
    assert addition.adding(2, 2) == 4


