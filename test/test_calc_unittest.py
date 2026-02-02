import unittest
import sys
import os

# FIXED: Point to the project root (remove '/src' from the end)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now Python can find 'src' as a module
from src.calculator import addition, subtraction, multiplication, division

class TestCalculator(unittest.TestCase):

    def test_addition(self):
        self.assertEqual(addition(3, 4), 7)
        self.assertEqual(addition(-1, 1), 0)
        self.assertEqual(addition(-1, -1), -2)

    def test_subtraction(self):
        self.assertEqual(subtraction(10, 5), 5)
        self.assertEqual(subtraction(-1, 1), -2)

    def test_multiplication(self):
        self.assertEqual(multiplication(3, 7), 21)
        self.assertEqual(multiplication(-1, 3), -3)

    def test_division(self):
        self.assertEqual(division(10, 2), 5)
        self.assertEqual(division(9, 3), 3)

    def test_division_by_zero(self):
        with self.assertRaises(ValueError):
            division(10, 0)

if __name__ == '__main__':
    unittest.main()