import itertools
import numpy as np
import pytest
from src.tsp import exact, distance, nearest, validate
from src.qubo import build_qubo, evaluate, encode, decode, ising
from src.quantum import solve, qiskit_available

D = np.array([[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]], dtype=float)

def test_exact():
    route, cost = exact(D)
    assert cost == 80
    assert route[0] == 0 and set(route) == set(range(4))

def test_nearest():
    route, cost = nearest(D)
    assert cost == distance(route, D)

def test_validate_rejects_bad_input():
    with pytest.raises(ValueError):
        validate(np.ones((4, 4)))  # nonzero diagonal

def test_qubo_valid_routes():
    linear, quad, offset, _ = build_qubo(D)
    for p in itertools.permutations([1,2,3]):
        route = [0, *p]
        bits = encode(route)
        assert decode(bits, 4) == route
        assert np.isclose(evaluate(bits, linear, quad, offset), distance(route, D))

def test_qubo_minimum_is_optimal_tour():
    linear, quad, offset, _ = build_qubo(D)
    best = min(range(512), key=lambda k: evaluate(np.array([(k >> i) & 1 for i in range(9)]), linear, quad, offset))
    route = decode(np.array([(best >> i) & 1 for i in range(9)]), 4)
    assert route is not None and distance(route, D) == 80

def test_ising_all_states():
    linear, quad, offset, _ = build_qubo(D)
    c, h, J = ising(linear, quad, offset)
    for k in range(512):
        bits = np.array([(k >> i) & 1 for i in range(9)])
        z = 1 - 2 * bits
        energy = c + np.dot(h, z) + sum(v*z[a]*z[b] for (a,b),v in J.items())
        assert np.isclose(energy, evaluate(bits, linear, quad, offset))

def test_invalid():
    assert decode(np.zeros(9, dtype=int), 4) is None

def test_qaoa_numpy_backend_finds_valid_route():
    r = solve(D, depth=2, iterations=60, shots=2048, seed=42, restarts=3, backend="numpy")
    assert r["route"] is not None and r["valid_shots"] > 0
    assert r["distance"] == distance(r["route"], D) >= 80

@pytest.mark.skipif(not qiskit_available(), reason="qiskit / qiskit-aer not installed")
def test_qaoa_qiskit_backend_finds_valid_route():
    r = solve(D, depth=2, iterations=60, shots=2048, seed=42, restarts=3, backend="qiskit")
    assert r["backend"] == "qiskit" and r["route"] is not None

@pytest.mark.skipif(not qiskit_available(), reason="qiskit / qiskit-aer not installed")
def test_qiskit_and_numpy_statevectors_agree():
    """Same parameters -> same expected energy on both backends (validates the circuit)."""
    a = solve(D, depth=1, iterations=1, shots=64, seed=3, restarts=1, backend="qiskit")
    b = solve(D, depth=1, iterations=1, shots=64, seed=3, restarts=1, backend="numpy")
    assert np.isclose(a["history"][0], b["history"][0], rtol=1e-6)
