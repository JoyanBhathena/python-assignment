import unittest
import numpy as np
from practice import IdealFunctionSelector

class TestIdealFunctionSelectorClass(unittest.TestCase):

    def test_calculate_lse(self):
        # Create an instance of IdealFunctionSelector with some example data
        train = [1, 2, 3, 4, 5]
        ideal_functions = [3, 4, 5, 6, 7]
        selector = IdealFunctionSelector(train, ideal_functions)

        # Test the calculate_lse method with different inputs
        y1 = np.array([2, 3, 4, 5, 6])
        y2 = np.array([3, 4, 5, 6, 7])
        result = selector.calculate_lse(y1, y2)
        expected_lse = np.sum((y1 - y2) ** 2)
        self.assertEqual(result, expected_lse, "LSE calculation incorrect")

        # Test with different input data
        y1 = np.array([1, 1, 1, 1, 1])
        y2 = np.array([0, 0, 0, 0, 0])
        result = selector.calculate_lse(y1, y2)
        expected_lse = np.sum((y1 - y2) ** 2)
        self.assertEqual(result, expected_lse, "LSE calculation incorrect")

        # Test with empty input arrays
        y1 = np.array([])
        y2 = np.array([])
        result = selector.calculate_lse(y1, y2)
        expected_lse = 0  # LSE of empty arrays should be 0
        self.assertEqual(result, expected_lse, "LSE calculation incorrect")

        # Test with arrays of different lengths
        y1 = np.array([1, 2, 3, 4, 5])
        y2 = np.array([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            selector.calculate_lse(y1, y2)  # LSE calculation should raise a ValueError

if __name__ == '__main__':
    unittest.main()