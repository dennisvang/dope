# DoPe

**Do**uglas-**Pe**ucker line simplification (data reduction).

Currently includes only a recursive implementation (depth-first). An iterative implementation may follow (breadth-first).

## Example

```python
from dope import DoPeR
import numpy

data_original = [
    [0, 0], [1, -1], [2, 2], [3, 0], [4, 0], [5, -1], [6, 1], [7, 0]
]

dp = DoPeR(data=data_original)

# either use epsilon threshold (i.e. max. error w.r.t. normalized data)
data_simplified_eps = dp.simplify(epsilon=0.2)

# or use maximum recursion depth
data_simplified_depth = dp.simplify(max_depth=4)

# compare original data and simplified data in a plot
dp.plot()
```

Also see examples in [tests][2].

## References:

[Douglas DH, Peucker TK. *Algorithms for the reduction of the number of points required to represent a digitized line or its caricature.*
Cartographica: the international journal for geographic information and geovisualization. 1973 Dec 1;10(2):112-22.][1]

[1]: https://doi.org/10.3138/FM57-6770-U75U-7727
[2]: https://github.com/dennisvang/dope/tree/main/tests