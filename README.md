# Quantum Route Optimization using QAOA

An end-to-end educational project implementing the fixed-depot Traveling Salesman Problem (TSP) with a QUBO, Ising Hamiltonian, explicit QAOA circuit on Qiskit Aer, classical brute force, nearest-neighbor heuristic, and Streamlit dashboard.

## Windows setup (PowerShell / VS Code terminal)

Install **Python 3.11** (64-bit), then open the extracted folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt and `.venv\Scripts\activate.bat`, or run `.venv\Scripts\python.exe -m streamlit run app.py` directly.

On macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && streamlit run app.py`.

Open the local URL shown by Streamlit (usually http://localhost:8501).

## Quick check without the UI

```powershell
python cli.py
```

Prints exact, nearest-neighbor and QAOA routes for the 4-city example. Windows users can also double-click `run.bat` (creates `.venv`, installs, launches the app); macOS/Linux: `./run.sh`.

## Simulation backends

QAOA runs on **Qiskit Aer** when installed (default). If Qiskit/Aer cannot be imported, a built-in NumPy statevector simulator runs the identical ansatz, so the app still works. Choose the backend in the sidebar (`auto` / `qiskit` / `numpy`).

## Tests

```powershell
python -m pytest -q
```

## What happens

1. Enter a distance matrix (A is depot).
2. Exact brute-force and nearest-neighbor algorithms compute classical reference solutions.
3. TSP is encoded with `(n-1)^2` binary variables `x_(city, position)` and exactly-one penalties.
4. QUBO maps to an Ising Hamiltonian using `x=(1-Z)/2`.
5. QAOA alternates cost RZ/RZZ gates (angles rescaled so gamma is O(1)) with RX mixer gates; SciPy COBYLA with several random restarts minimizes the exact simulated expectation energy.
6. The optimized circuit is sampled; only valid measured routes are decoded.
7. Compare route lengths, runtime, optimality gap, energy convergence, and measured states.

## Example

For the provided four-city matrix, exact optimum is **80 distance units**, for example A → B → D → C → A. QAOA is stochastic and **may not** measure an optimal or even valid route in limited shots. Adjust depth, optimizer evaluations and shots to experiment.

## Limits and honest interpretation

- Supports 3–5 cities; 5 cities = 16 qubits, with 65,536 basis states, and can be slow/memory-heavy. Four cities is the recommended demonstration.
- The X mixer does not preserve TSP constraints. Invalid samples are discarded, and some runs may yield no valid measured routes.
- This is a **local simulator**, not a real quantum hardware job; no quantum speedup is claimed.
- Distances are supplied by the user; there is no live mapping or traffic API.
- Route diagrams use illustrative circle coordinates and do not represent actual geography.
- The QAOA optimizer evaluates the exact statevector expectation, then samples the final circuit; optimization runtime includes simulation and Python overhead.
- Qiskit APIs change; requirements specify broad compatible major-version ranges, not a frozen lockfile.

## Project layout

```
app.py                 Streamlit dashboard
src/tsp.py             Exact and greedy classical TSP
src/qubo.py            QUBO model, decoding, Ising conversion
src/quantum.py         Explicit QAOA + Aer + COBYLA
 tests/test_core.py     Mathematical correctness tests
requirements.txt       Dependencies
```
