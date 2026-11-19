"""Explicit QAOA (cost layer RZ/RZZ + X-mixer RX) for the TSP QUBO.

Two interchangeable simulation backends run the SAME ansatz:
  * "qiskit": circuits executed on qiskit-aer (statevector + shot sampling)
  * "numpy" : a small built-in statevector simulator (no extra dependencies)
backend="auto" uses Qiskit Aer when it imports and falls back to NumPy otherwise.
"""
import time
import numpy as np
from scipy.optimize import minimize
from .qubo import build_qubo, decode, ising
from .tsp import distance

COST_WIDTH = 20.0  # cost-Hamiltonian spectrum is rescaled to this width so gamma ~ O(1)


def qiskit_available():
    try:
        import qiskit  # noqa: F401
        import qiskit_aer  # noqa: F401
        return True
    except Exception:
        return False


def _all_energies(q, linear, quadratic, offset):
    """QUBO energy of every computational basis state (bit i of index = qubit i)."""
    idx = np.arange(1 << q, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(q)) & 1).astype(float)
    energies = offset + bits @ linear
    for (a, b), value in quadratic.items():
        energies += value * bits[:, a] * bits[:, b]
    return energies


def _numpy_mixer(psi, q, beta):
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.reshape([2] * q)
    for axis in range(q):
        a, b = np.take(psi, 0, axis=axis), np.take(psi, 1, axis=axis)
        psi = np.stack([c * a + s * b, s * a + c * b], axis=axis)
    return psi.reshape(-1)


def _numpy_state(params, depth, q, scaled_energy):
    psi = np.full(1 << q, 1 / np.sqrt(1 << q), dtype=complex)
    for layer in range(depth):
        psi = psi * np.exp(-1j * params[layer] * scaled_energy)  # exp(-i*gamma*H_C)
        psi = _numpy_mixer(psi, q, params[depth + layer])        # exp(-i*beta*sum X)
    return psi


def solve(d, depth=1, iterations=35, shots=2048, seed=42, restarts=3, backend="auto"):
    if backend == "auto":
        backend = "qiskit" if qiskit_available() else "numpy"
    if backend == "qiskit":
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        simulator = AerSimulator(method="statevector", seed_simulator=int(seed))

    linear, quadratic, offset, penalty = build_qubo(d)
    constant, h, couplings = ising(linear, quadratic, offset)
    q = len(linear)
    energies = _all_energies(q, linear, quadratic, offset)
    scale = COST_WIDTH / max(float(energies.max() - energies.min()), 1e-12)
    scaled_energy = (energies - energies.min()) * scale
    rng = np.random.default_rng(int(seed))
    start = time.perf_counter()

    def circuit(params, measure=False):
        qc = QuantumCircuit(q, q if measure else 0)
        qc.h(range(q))
        for layer in range(depth):
            gamma, beta = float(params[layer]) * scale, float(params[depth + layer])
            for i, value in enumerate(h):
                if abs(value) > 1e-12:
                    qc.rz(2 * gamma * value, i)
            for (a, b), value in couplings.items():
                if abs(value) > 1e-12:
                    qc.rzz(2 * gamma * value, a, b)
            for i in range(q):
                qc.rx(2 * beta, i)
        if measure:
            qc.measure(range(q), range(q))
        return qc

    def probabilities(params):
        if backend == "qiskit":
            qc = circuit(params)
            qc.save_statevector()
            state = np.asarray(simulator.run(qc).result().get_statevector())
        else:
            state = _numpy_state(params, depth, q, scaled_energy)
        probs = np.abs(state) ** 2
        return probs / probs.sum()

    best_run = None
    total_evals = 0
    for _ in range(max(1, int(restarts))):
        history = []

        def expectation(params):
            value = float(probabilities(params) @ energies)
            history.append(value)
            return value

        x0 = np.r_[rng.uniform(0, 1.0, depth), rng.uniform(0, np.pi, depth)]
        opt = minimize(expectation, x0, method="COBYLA", options={"maxiter": int(iterations), "rhobeg": 0.3})
        total_evals += len(history)
        if best_run is None or opt.fun < best_run[0].fun:
            best_run = (opt, history)
    opt, history = best_run

    if backend == "qiskit":
        measured = simulator.run(circuit(opt.x, measure=True), shots=int(shots)).result().get_counts()
        measured = {k.replace(" ", ""): v for k, v in measured.items()}
        circuit_text = str(circuit(opt.x).draw(output="text"))
    else:
        draws = rng.multinomial(int(shots), probabilities(opt.x))
        measured = {format(int(k), f"0{q}b"): int(draws[k]) for k in np.nonzero(draws)[0]}
        circuit_text = (f"NumPy statevector backend (Qiskit Aer not installed). Equivalent circuit: "
                        f"H on {q} qubits, then {depth} x [cost layer RZ/RZZ(gamma) + RX(2*beta) mixer], "
                        f"gamma/beta = {np.round(opt.x, 4).tolist()}")

    valid = []
    for bitstring, count in measured.items():
        bits = np.array([int(c) for c in bitstring[::-1]])  # Qiskit prints qubit q-1 first
        route = decode(bits, len(d))
        if route is not None:
            valid.append((distance(route, d), -count, route, count))
    valid.sort()
    best = valid[0] if valid else None
    return {
        "route": best[2] if best else None,
        "distance": best[0] if best else None,
        "valid_shots": sum(v[3] for v in valid),
        "shots": int(shots),
        "counts": measured,
        "history": history,
        "iterations": total_evals,
        "seconds": time.perf_counter() - start,
        "penalty": penalty,
        "qubits": q,
        "expected_qubo_cost": float(opt.fun),
        "circuit": circuit_text,
        "backend": backend,
    }
