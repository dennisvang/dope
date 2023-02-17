import sys

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None
import numpy

# single source of truth for package version
# https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
# https://semver.org/
__version__ = '0.0.1'

RECURSION_LIMIT = sys.getrecursionlimit()


def normalize(data: numpy.ndarray) -> numpy.ndarray:
    """transform data columns so their values are between zero and one"""
    return (data - data.min(axis=0, initial=None)) / data.ptp(axis=0)


def distance_point_to_line(point, line):
    """
    distance from one (or more) point(s) to a line defined by two nodes

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
        s[0, 0] * (line[0, 1] - point[:, 1])
        - (line[0, 0] - point[:, 0]) * s[0, 1]
    )
    denominator = numpy.sqrt(numpy.dot(s[0], s[0]))
    return numerator / denominator


class DoPe(object):
    def __init__(
        self, data: numpy.ndarray, epsilon: float, max_depth: int = None
    ):
        self.data = normalize(data)
        self.epsilon = epsilon
        if max_depth is None or max_depth >= RECURSION_LIMIT:
            max_depth = RECURSION_LIMIT
        self.max_depth = max_depth
        self.indices = None

    @property
    def max_length(self):
        """max. number of nodes in the tree at given depth (plus two edges)"""
        return sum(2**i for i in range(self.max_depth)) + 2

    def simplify(self, interval=None, depth=0):
        """recursive (depth-first) Douglas-Peucker line simplification"""
        # init
        if interval is None:
            interval = [0, self.data.shape[0] - 1]
            self.indices = numpy.array(interval)
        # calculate point-line distances
        distances = distance_point_to_line(
            point=self.data[interval[0] + 1 : interval[1], :],
            line=self.data[interval][:],
        )
        # evaluate conditions
        bottom_reached = not distances.size
        max_depth_reached = depth == self.max_depth
        local_max_index = numpy.argmax(distances) if distances.size else None
        epsilon_reached = distances[local_max_index] < self.epsilon
        # return or split
        if bottom_reached or max_depth_reached or epsilon_reached:
            # base case
            return
        else:
            # recursion case
            global_max_index = local_max_index + interval[0] + 1
            insert_index = numpy.nonzero(self.indices == interval[1])[0]
            # store the split node
            self.indices = numpy.insert(
                self.indices, insert_index, global_max_index
            )
            # split and evaluate recursively
            depth += 1
            self.simplify(
                interval=[interval[0], global_max_index], depth=depth
            )
            self.simplify(
                interval=[global_max_index, interval[1]], depth=depth
            )

    def plot(self):
        if plt:
            plt.plot(self.data[:, 0], self.data[:, 1], color='0.7')
            plt.plot(
                self.data[self.indices, 0],
                self.data[self.indices, 1],
                color='r',
                linestyle=':',
                marker='o',
            )
            plt.title(
                f'normalized data, epsilon={self.epsilon}, '
                f'max_depth={self.max_depth}, '
                f'reduction: {self.data.shape[0]} to {self.indices.size}'
            )
            plt.grid()
            plt.axis('equal')
            plt.show()
        else:
            Warning('matplotlib not found, try installing extras: dope[plot]')
