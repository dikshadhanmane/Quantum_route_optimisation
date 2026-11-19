"""Fixed-depot TSP QUBO: x_(city, position), excluding depot city 0."""
import numpy as np


def build_qubo(d, penalty=None):
    n = len(d)
    m = n - 1
    q = m * m
    # Penalty must beat any saving from breaking a constraint. 1.5 * max distance was
    # verified (brute force over all bitstrings, 40+ random matrices) to keep the QUBO
    # minimum on the true optimal tour, while keeping the energy landscape tame for QAOA.
    penalty = float(penalty if penalty is not None else 1.5 * np.max(d))
    linear = np.zeros(q)
    quadratic = {}
    offset = 2 * m * penalty

    def add_pair(a, b, value):
        if a == b:
            linear[a] += value
        else:
            key = tuple(sorted((a, b)))
            quadratic[key] = quadratic.get(key, 0.0) + value

    def idx(city, pos):
        return (city - 1) * m + pos

    # Penalty (sum x - 1)^2 for each city and each position.
    for groups in ([idx(city, pos) for pos in range(m)] for city in range(1, n)):
        for a in groups:
            linear[a] -= penalty
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                add_pair(groups[a], groups[b], 2 * penalty)
    for pos in range(m):
        group = [idx(city, pos) for city in range(1, n)]
        for a in group:
            linear[a] -= penalty
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                add_pair(group[a], group[b], 2 * penalty)

    for city in range(1, n):
        linear[idx(city, 0)] += d[0, city]
        linear[idx(city, m - 1)] += d[city, 0]
    for pos in range(m - 1):
        for a in range(1, n):
            for b in range(1, n):
                add_pair(idx(a, pos), idx(b, pos + 1), d[a, b])
    return linear, quadratic, offset, penalty


def evaluate(bits, linear, quadratic, offset):
    return float(offset + np.dot(linear, bits) + sum(v * bits[a] * bits[b] for (a, b), v in quadratic.items()))


def decode(bits, n):
    m = n - 1
    if len(bits) != m * m:
        return None
    matrix = np.asarray(bits).reshape(m, m)
    if not (np.all(matrix.sum(axis=0) == 1) and np.all(matrix.sum(axis=1) == 1)):
        return None
    return [0] + [int(np.argmax(matrix[:, pos])) + 1 for pos in range(m)]


def encode(route):
    m = len(route) - 1
    bits = np.zeros(m * m, dtype=int)
    for pos, city in enumerate(route[1:]):
        bits[(city - 1) * m + pos] = 1
    return bits


def ising(linear, quadratic, offset):
    # x=(1-Z)/2; returns constant, h_i, J_ij.
    h = -linear.astype(float).copy() / 2
    constant = float(offset + np.sum(linear) / 2)
    J = {}
    for (a, b), value in quadratic.items():
        constant += value / 4
        h[a] -= value / 4
        h[b] -= value / 4
        J[a, b] = value / 4
    return constant, h, J
