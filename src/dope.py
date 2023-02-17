import numpy


def normalize(data: numpy.ndarray) -> numpy.ndarray:
    """ transform data columns so their values are between zero and one """
    return (data - data.min(axis=0, initial=None)) / data.ptp(axis=0)


def point_to_line(point, line, ignore_denominator=False):
    """
    distance from point to line defined by two nodes

    ignore denominator when comparing distances of multiple points to the
    same line element, to prevent calculation of the square root

    point = [x0, y0]
    line = [[x1, y1], [x2, y2]]

    distance = abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1)) / sqrt((x2-x1)^2+(y2-y1)^2)

    reference:

    https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
    """
    diffs = numpy.diff(numpy.vstack((point, line)), n=1, axis=0)
    distance = numpy.abs(diffs[1][0] * diffs[0][1] - diffs[0][0] * diffs[1][1])
    if not ignore_denominator:
        distance /= numpy.sqrt(diffs[1][0]**2 + diffs[1][1]**2)
    return distance
