"""Command-line demo (no Streamlit needed):  python cli.py"""
import numpy as np
from src.tsp import exact, nearest
from src.quantum import solve

D = np.array([[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]], dtype=float)
name = lambda r: " -> ".join("ABCDE"[i] for i in r + [0]) if r else "none"
er, ed = exact(D); gr, gd = nearest(D)
q = solve(D, depth=2, iterations=60, shots=2048, seed=42, restarts=3)
print(f"Exact            : {name(er)}  ({ed:g})")
print(f"Nearest neighbor : {name(gr)}  ({gd:g})")
print(f"QAOA [{q['backend']}]  : {name(q['route'])}  ({q['distance']})  valid shots {q['valid_shots']}/{q['shots']}  {q['seconds']:.1f}s")
