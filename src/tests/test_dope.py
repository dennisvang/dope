from unittest import TestCase
import numpy
from src.dope import normalize, point_to_line, DoPe


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
        point = numpy.array([actual_distance, 0])
        line = numpy.array([[0, 0], [0, line_segment_length]])
        cases = [(False, actual_distance),
                 (True, actual_distance * line_segment_length)]
        for ignore, expected_length in cases:
            with self.subTest(ignore=ignore):
                self.assertEqual(
                    expected_length,
                    point_to_line(
                        point=point, line=line, ignore_denominator=ignore))
