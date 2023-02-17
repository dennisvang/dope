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
