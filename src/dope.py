from matplotlib import pyplot as plt
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


class DoPe(object):
    def __init__(self, data: numpy.ndarray, epsilon: float):
        self.epsilon = epsilon
        self.data = normalize(data)
        self.indices = None

    def simplify(self, interval=None):
        """ recursive implementation of Douglas-Peucker line simplification """
        # init
        if interval is None:
            interval = [0, self.data.shape[0] - 1]
            self.indices = numpy.array(interval)
        # calculate point-line distances
        distances = points_to_line(
            # include the first point, for convenient indexing
            point=self.data[interval[0]:interval[1], :],
            line=self.data[interval][:])
        # check if largest distance meets requirement
        max_index_local = numpy.argmax(distances)
        if distances[max_index_local] < self.epsilon:
            # base case
            return
        else:
            # recursion case
            max_index_global = max_index_local + interval[0]
            insert_index = numpy.nonzero(self.indices == interval[1])[0]
            self.indices = numpy.insert(
                self.indices, insert_index, max_index_global)
            self.simplify(interval=[interval[0], max_index_global])
            self.simplify(interval=[max_index_global, interval[1]])

    def plot(self):
        plt.plot(self.data[:, 0], self.data[:, 1], color='0.8')
        plt.plot(self.data[self.indices, 0], self.data[self.indices, 1],
                 color='r', linestyle='', marker='o')
        plt.title(f'normalized data, epsilon={self.epsilon}')
        plt.grid()
        plt.axis('equal')
        plt.show()
