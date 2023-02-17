import numpy


def normalize(data: numpy.ndarray) -> numpy.ndarray:
    """ transform data columns so their values are between zero and one """
    return (data - data.min(axis=0, initial=None)) / data.ptp(axis=0)


def points_to_line(point, line):
    """
    distance from one or more point(s) to a line defined by two nodes

    point = [x0, y0]  (can also be an m x 2 array of points)
    line = [[x1, y1], [x2, y2]]

    distance = abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1)) / sqrt((x2-x1)^2+(y2-y1)^2)

    reference:

    https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
    """
    if point.ndim == 1:
        point.resize((1, 2))
    s = numpy.diff(line, n=1, axis=0)
    numerator = numpy.abs(
        s[0, 0] * (line[0, 1]-point[:, 1]) - (line[0, 0]-point[:, 0]) * s[0, 1])
    denominator = numpy.sqrt(numpy.dot(s[0], s[0]))
    return numerator / denominator
