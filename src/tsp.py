import itertools
import numpy as np


def validate(matrix):
    d = np.asarray(matrix, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or not 3 <= len(d) <= 5:
        raise ValueError('Provide a square matrix of 3 to 5 cities.')
    if not np.all(np.isfinite(d)) or np.any(d < 0) or not np.allclose(np.diag(d), 0):
        raise ValueError('Distances must be finite, nonnegative, with zero diagonal.')
    return d


def distance(route, d):
    return float(sum(d[route[k], route[(k+1) % len(route)]] for k in range(len(route))))


def exact(d):
    best = min(((0,) + p for p in itertools.permutations(range(1, len(d)))), key=lambda r: distance(r, d))
    return list(best), distance(best, d)


def nearest(d):
    route = [0]
    remaining = set(range(1, len(d)))
    while remaining:
        nxt = min(remaining, key=lambda j: (d[route[-1], j], j))
        route.append(nxt)
        remaining.remove(nxt)
    return route, distance(route, d)
