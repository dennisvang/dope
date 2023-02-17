from unittest import TestCase
import numpy
from dope import normalize, distance_point_to_line, DoPe


class NormalizeTests(TestCase):
    def test_normalize(self):
        rng = numpy.random.default_rng()
        data = 10 * rng.standard_normal((10, 2)) + numpy.array([-100, 100])
        normalized = normalize(data)
        self.assertTrue(numpy.all(normalized.min(axis=0, initial=None) == 0))
        self.assertTrue(numpy.all(normalized.max(axis=0, initial=None) == 1))


class DistancePointToLineTests(TestCase):
    def test_distance_point_to_line(self):
        actual_distance = 2
        line_segment_length = 3
        line = numpy.array([[0, 0], [0, line_segment_length]])
        point = numpy.array([actual_distance, 0])
        cases = [point, numpy.vstack((point, point))]
        for case in cases:
            with self.subTest(case=case):
                distances = distance_point_to_line(point=case, line=line)
                self.assertTrue(numpy.all(distances == actual_distance))


class DoPeTests(TestCase):
    def test_simplify_epsilon(self):
        data = numpy.array(
            [[0, 0], [1, -1], [2, 2], [3, 0], [4, 0], [5, -1], [6, 1], [7, 0]]
        )
        cases = [
            (0.0, 8),
            (0.1, 8),
            (0.2, 6),
            (0.3, 5),
            (0.4, 4),
            (0.5, 3),
            (0.6, 3),
            (0.7, 2),
            (1.0, 2),
        ]
        for epsilon, expected_length in cases:
            with self.subTest(epsilon=epsilon):
                dp = DoPe(data=data, epsilon=epsilon, max_depth=None)
                dp.simplify()
                dp.plot()
                self.assertEqual(expected_length, dp.indices.size)

    def test_simplify_max_depth(self):
        data = numpy.array(
            [[0, 0], [1, -1], [2, 2], [3, 0], [4, 0], [5, -1], [6, 1], [7, 0]]
        )
        cases = [0, 1, 2, 3, 4]  # the None case is covered in the epsilon test
        for max_depth in cases:
            with self.subTest(max_depth=max_depth):
                dp = DoPe(data=data, epsilon=0, max_depth=max_depth)
                dp.simplify()
                dp.plot()
                self.assertLessEqual(dp.indices.size, dp.max_length)
