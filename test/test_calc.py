import pytest
from src.calculator import addition, subtraction, multiplication, division


def test_addition():
    assert addition(7,19) == 26
    assert addition(-5, 1) == -4
    assert addition(-3,3) == 0


def test_subtraction():
    assert subtraction(8,3) == 5
    assert subtraction(0, 9) == -9
    assert subtraction(-25, -10) == -15

def test_multiplication():
    assert multiplication(7,8) == 56
    assert multiplication(-2, 2) == -4
    assert multiplication(-4, 0) == 0


def test_division():
    assert division(20,5) == 4
    assert division(-9, 3) == -3
    assert division(-48, -8) == 6


def test_division_by_zero():
    assert division(5, 0) == "Error: Division by zero"
