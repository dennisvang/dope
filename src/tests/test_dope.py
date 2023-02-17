from unittest import TestCase
import numpy
from src.dope import normalize, points_to_line, DoPe


class NormalizeTests(TestCase):
    def test_normalize(self):
        rng = numpy.random.default_rng()
        data = 10 * rng.standard_normal((10, 2)) + numpy.array([-100, 100])
        normalized = normalize(data)
        self.assertTrue(numpy.all(normalized.min(axis=0, initial=None) == 0))
        self.assertTrue(numpy.all(normalized.max(axis=0, initial=None) == 1))


class PointToLineTests(TestCase):
    def test_point_to_line(self):
        actual_distance = 2
        line_segment_length = 3
        line = numpy.array([[0, 0], [0, line_segment_length]])
        point = numpy.array([actual_distance, 0])
        cases = [point, numpy.vstack((point, point))]
        for case in cases:
            with self.subTest(case=case):
                self.assertTrue(numpy.all(
                    points_to_line(point=case, line=line) == actual_distance))


class DoPeTests(TestCase):
    def test_simplify_epsilon(self):
        data = numpy.array(
            [[0, 0], [1, -1], [2, 2], [3, 0], [4, 0], [5, -1], [6, 1], [7, 0]])
        cases = [(0.0, 8), (0.1, 8), (0.2, 6), (0.3, 5), (0.4, 4), (0.5, 3),
                 (0.6, 3), (0.7, 2), (1.0, 2)]
        for epsilon, expected_length in cases:
            with self.subTest(epsilon=epsilon):
                dp = DoPe(data=data, epsilon=epsilon, max_depth=None)
                dp.simplify()
                dp.plot()
                self.assertEqual(expected_length, dp.indices.size)

    def test_simplify_max_depth(self):
        data = numpy.array(
            [[0, 0], [1, -1], [2, 2], [3, 0], [4, 0], [5, -1], [6, 1], [7, 0]])
        cases = [0, 1, 2, 3, 4]  # the None case is covered in the epsilon test
        for max_depth in cases:
            # max. number of nodes in the tree at given depth (plus two edges)
            expected_max_length = sum(2**i for i in range(max_depth)) + 2
            with self.subTest(max_depth=max_depth):
                dp = DoPe(data=data, epsilon=0, max_depth=max_depth)
                dp.simplify()
                dp.plot()
                self.assertLessEqual(dp.indices.size, expected_max_length)
