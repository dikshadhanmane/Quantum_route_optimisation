import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from src.tsp import validate, exact, nearest

st.set_page_config(page_title='Quantum Route Lab', page_icon='⚛️', layout='wide')
st.title('⚛️ Quantum Route Optimization using QAOA')
st.caption('Fixed-depot Travelling Salesman Problem • QUBO → Ising → QAOA → comparison')

sample = np.array([[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]], dtype=float)
with st.sidebar:
    st.header('Experiment settings')
    n = st.selectbox('Number of cities (including depot)', [3,4,5], index=1)
    depth = st.slider('QAOA layers (p)', 1, 3, 2)
    iterations = st.slider('Optimizer evaluations per restart', 10, 150, 60, 5)
    restarts = st.slider('Random restarts', 1, 6, 3)
    shots = st.select_slider('Measurement shots', [256,512,1024,2048,4096], value=2048)
    seed = st.number_input('Random seed', 0, 100000, 42)
    backend = st.selectbox('Simulation backend', ['auto', 'qiskit', 'numpy'], help='auto = Qiskit Aer if installed, else built-in NumPy statevector simulator')
    st.info('5 cities require 16 qubits and can be slow. Start with 4 cities (9 qubits).')

labels = [chr(65+i) for i in range(n)]
if n == 4:
    default = sample
else:
    coords = np.array([[0,0],[3,1],[1,4],[5,5],[6,1]], dtype=float)[:n]
    default = np.rint(np.linalg.norm(coords[:,None,:]-coords[None,:,:], axis=2)*5)
    np.fill_diagonal(default, 0)
st.subheader('1. Distance matrix')
st.write('City A is the warehouse/depot. Edit distances directly (directed/asymmetric distances are supported).')
df = st.data_editor(pd.DataFrame(default, index=labels, columns=labels), key=f'matrix_{n}')
try:
    d = validate(df.to_numpy(dtype=float))
except (ValueError, TypeError) as exc:
    st.error(str(exc))
    st.stop()

if st.button('🚀 Run classical and QAOA optimization', type='primary'):
    with st.spinner('Running algorithms and simulating the quantum circuit...'):
        t = time.perf_counter()
        exact_route, exact_dist = exact(d)
        exact_time = time.perf_counter() - t
        t = time.perf_counter()
        greedy_route, greedy_dist = nearest(d)
        greedy_time = time.perf_counter() - t
        try:
            from src.quantum import solve
            q = solve(d, depth=depth, iterations=iterations, shots=shots, seed=int(seed), restarts=restarts, backend=backend)
        except Exception as exc:  # noqa: BLE001 - show any simulator failure in the UI
            st.error(f'Quantum simulation failed: {exc}. Try backend = numpy, fewer cities, or check memory.')
            st.stop()
    st.session_state['result'] = (d.copy(), labels, exact_route, exact_dist, exact_time, greedy_route, greedy_dist, greedy_time, q)

if 'result' in st.session_state:
    old_d, names, er, ed, et, gr, gd, gt, q = st.session_state['result']
    if old_d.shape != d.shape or not np.array_equal(old_d, d):
        st.warning('Inputs changed. Run optimization again to refresh results.')
        st.stop()
    st.subheader('2. Results')
    def fmt(route):
        return ' → '.join(names[i] for i in (route + [0])) if route else 'No valid route measured'
    cols = st.columns(3)
    cols[0].metric('Exact optimum', f'{ed:g} distance units')
    cols[1].metric('Nearest neighbor', f'{gd:g} distance units')
    cols[2].metric('QAOA best valid sample', f'{q["distance"]:g}' if q['distance'] is not None else 'Not found')
    st.write('**Exact:**', fmt(er))
    st.write('**Nearest neighbor:**', fmt(gr))
    st.write('**QAOA:**', fmt(q['route']))
    qgap = ((q['distance']/ed-1)*100) if q['distance'] is not None and ed > 0 else None
    st.dataframe(pd.DataFrame([
        {'Algorithm':'Brute force (exact)', 'Distance':ed, 'Time (s)':et, 'Gap (%)':0},
        {'Algorithm':'Nearest neighbor', 'Distance':gd, 'Time (s)':gt, 'Gap (%)':(gd/ed-1)*100 if ed else None},
        {'Algorithm':'QAOA (best valid measured)', 'Distance':q['distance'], 'Time (s)':q['seconds'], 'Gap (%)':qgap},
    ]), hide_index=True)
    st.write(f'**Qubits:** {q["qubits"]} · **Penalty:** {q["penalty"]:g} · **Valid samples:** {q["valid_shots"]}/{q["shots"]} · **Evaluations:** {q["iterations"]} · **Backend:** {q["backend"]}')
    st.caption('The best valid measured QAOA route is reported; this is not proof of quantum advantage. Invalid bitstrings are rejected.')
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(q['history'])  # best restart; axes[0].set(title='QAOA expected QUBO energy', xlabel='Evaluation', ylabel='Expected energy')
    axes[1].bar(['Exact','Nearest','QAOA'], [ed, gd, q['distance'] if q['distance'] is not None else 0]); axes[1].set(title='Best route distances', ylabel='Distance')
    st.pyplot(fig)
    plt.close(fig)
    st.subheader('3. Route visualization (schematic)')
    coords = np.array([[np.cos(2*np.pi*i/len(names)), np.sin(2*np.pi*i/len(names))] for i in range(len(names))])
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    for ax, route, title in zip(axes, [er, q['route']], ['Classical exact', 'QAOA best valid']):
        ax.scatter(coords[:,0], coords[:,1], s=300)
        for i, name in enumerate(names):
            ax.annotate(name, coords[i], ha='center', va='center', color='white', weight='bold')
        if route:
            cycle = route + [0]
            ax.plot(coords[cycle,0], coords[cycle,1], '-o')
        ax.set_title(title); ax.axis('equal'); ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)
    st.caption('Diagram positions are illustrative, not geographic coordinates. Distances come from the matrix.')
    with st.expander('Quantum circuit (text)'):
        st.code(q['circuit'])
    with st.expander('Top measured bitstrings'):
        st.dataframe(pd.DataFrame(sorted(q['counts'].items(), key=lambda x:-x[1])[:20], columns=['Bitstring (Qiskit order)','Count']), hide_index=True)
    st.download_button('Download results CSV', pd.DataFrame([{'method':'exact','distance':ed},{'method':'nearest','distance':gd},{'method':'qaoa','distance':q['distance']}]).to_csv(index=False), 'route_results.csv', 'text/csv')
else:
    st.info('Choose settings, edit distances, then click Run.')
