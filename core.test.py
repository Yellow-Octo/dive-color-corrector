import unittest

# Assuming the function getNormalizingInterval is defined in core.py
from core import getNormalizingInterval


class TestGetNormalizingInterval(unittest.TestCase):

    def test_normal_case(self):
        array = [1, 3, 7, 2, 8, 4]
        expected = (2, 8)
        result = getNormalizingInterval(array)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
