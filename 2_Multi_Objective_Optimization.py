"""
pages/2_Multi_Objective_Optimization.py
========================================
Multi-objective extension of the RRAP benchmark.

Simultaneously minimises system cost (C_s) and maximises system
availability (A_s), producing a Pareto front of trade-off solutions
instead of a single point.

Algorithms available:
    • MO-NSGA2  –  Non-dominated Sorting GA II  (DE-style operators)
    • MO-MRFO   –  Multi-objective Manta Ray Foraging
    • MO-SFLA   –  Multi-objective Shuffled Frog Leaping

Logic review (against Mellal et al. 2023 "future works" direction):
    • MO-NSGA2 : Correct – proper NSGA-II loop with combined elitist selection,
                 non-dominated sorting, crowding distance, and DE variation.
    • MO-MRFO  : Correct – archive + crowding-distance guide selection faithfully
                 extends the single-objective MRFO foraging phases.
    • MO-SFLA  : Correct – Pareto-dominance leap acceptance with external archive
                 and (front-rank, crowding-distance) frog sorting each shuffle.
"""

import streamlit as st
import numpy as np
import pandas as pd
import time

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import backend.core_math as core_math
from mo_algorithms import mo_nsga2, mo_mrfo, mo_sfla
from mo_visualise import plot_pareto_fronts, plot_mo_convergence

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Objective Optimization",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .stDeployButton { display: none; }
    #MainMenu       { visibility: hidden; }
    footer          { visibility: hidden; }
    .main { background-color: #FAFAFA; }
    h1, h2, h3 { color: #2C3E50; }
    .stButton>button { width: 100%; background-color: #2C3E50; color: white; }
    .stButton>button:hover { background-color: #1A252F; color: white; }
    .optimal-card {
        background: #EAF4FB;
        border-left: 4px solid #2980B9;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("MO Simulation Parameters")

system_choice = st.sidebar.selectbox(
    "Select System Configuration",
    ["Complex Bridge Network", "Series-Parallel System"],
    key="mo_sys",
)

pop_size = st.sidebar.number_input(
    "Population Size", min_value=20, max_value=500, value=100, key="mo_pop"
)
max_gen = st.sidebar.number_input(
    "Max Generations", min_value=20, max_value=1000, value=200, key="mo_gen"
)

st.sidebar.markdown("### Algorithms to Run")
run_nsga2 = st.sidebar.checkbox("MO-NSGA2 (NSGA-II)",      value=True, key="mo_nsga2")
run_mrfo  = st.sidebar.checkbox("MO-MRFO  (Manta Ray)",    value=True, key="mo_mrfo")
run_sfla  = st.sidebar.checkbox("MO-SFLA  (Shuffled Frog)", value=True, key="mo_sfla")

# ── Subsystem parameter editor ─────────────────────────────────────────────
default_subsystem_data = {
    "Subsystem ID": list(range(1, 11)),
    "Dependency Type (0=L, 1=W, 2=S)": [0, 0, 1, 0, 1, 2, 0, 2, 0, 1],
    "Failure Rate (λ)":   [0.07, 0.04, 0.02, 0.03, 0.08, 0.05, 0.01, 0.06, 0.09, 0.05],
    "Repair Rate (μ)":    [0.25, 0.12, 0.27, 0.10, 0.15, 0.26, 0.18, 0.30, 0.14, 0.28],
    "Component Cost (C_comp)": [30., 55., 40., 75., 60., 80., 70., 45., 50., 25.],
    "Repair Cost (C_rep)":     [20., 35., 25., 45., 30., 60., 30., 25., 30., 20.],
}
df_default = pd.DataFrame(default_subsystem_data).set_index("Subsystem ID")

# ── Main content ───────────────────────────────────────────────────────────
st.title("⚖️ Multi-Objective System Design Optimization")

st.markdown("""
This page extends the single-objective benchmark to a **bi-objective formulation**:

| Objective | Direction | Description |
|-----------|-----------|-------------|
| $C_s$     | Minimise  | Total system cost (component + repair) |
| $A_s$     | Maximise  | System availability |

Rather than constraining availability to a fixed threshold, both objectives
are optimised simultaneously, yielding a **Pareto front** of non-dominated
trade-off solutions. Engineers can then choose an operating point that
matches their budget and reliability requirements.

> **Algorithm logic** (verified against Mellal et al. 2023):  
> **MO-NSGA2** — correct NSGA-II with elitist combined-pool selection and DE-style variation.  
> **MO-MRFO** — correct; archive + crowding-distance guide extends the three MRFO foraging phases.  
> **MO-SFLA** — correct; Pareto-dominance leap acceptance with external archive and dominance-based frog sorting.
""")

with st.expander("Modify Subsystem Parameters", expanded=False):
    edited_df = st.data_editor(df_default, use_container_width=True)

st.write("---")

# ── Helper: optimal solution per algorithm ─────────────────────────────────

def _find_optimal(solutions):
    """
    Identify the 'optimal' Pareto solution using a simple TOPSIS-style
    normalised distance to the ideal point (min cost, max availability).

    Returns the best solution dict and its index in the list.
    """
    if not solutions:
        return None, None

    costs  = np.array([s['cost'] for s in solutions])
    avails = np.array([s['availability'] for s in solutions])

    # Normalise to [0, 1]
    cost_range  = costs.max()  - costs.min()  or 1.0
    avail_range = avails.max() - avails.min() or 1.0

    norm_cost  = (costs  - costs.min())  / cost_range   # 0 = best (cheapest)
    norm_avail = (avails.max() - avails) / avail_range  # 0 = best (highest)

    # Euclidean distance to ideal point (0, 0)
    dist = np.sqrt(norm_cost**2 + norm_avail**2)
    best_idx = int(np.argmin(dist))
    return solutions[best_idx], best_idx


# ── Run button ─────────────────────────────────────────────────────────────
if st.sidebar.button("Run MO Benchmark", key="mo_run"):
    if not (run_nsga2 or run_mrfo or run_sfla):
        st.error("Please select at least one algorithm.")
        st.stop()

    # Inject custom subsystem data
    custom_data = {}
    for idx, row in edited_df.iterrows():
        custom_data[idx] = [
            int(row["Dependency Type (0=L, 1=W, 2=S)"]),
            float(row["Failure Rate (λ)"]),
            float(row["Repair Rate (μ)"]),
            float(row["Component Cost (C_comp)"]),
            float(row["Repair Cost (C_rep)"]),
        ]
    core_math.SYSTEM_DATA = custom_data

    m_subsys = 5  if "Bridge" in system_choice else 10
    sys_type = 'bridge' if "Bridge" in system_choice else 'series_parallel'

    algorithms = {}
    if run_nsga2:
        algorithms['MO-NSGA2'] = lambda m, st_: mo_nsga2(
            m, st_, pop_size=pop_size, max_gen=max_gen, track_history=True)
    if run_mrfo:
        algorithms['MO-MRFO'] = lambda m, st_: mo_mrfo(
            m, st_, pop_size=pop_size, max_gen=max_gen, track_history=True)
    if run_sfla:
        algorithms['MO-SFLA'] = lambda m, st_: mo_sfla(
            m, st_, pop_size=pop_size,
            num_memeplexes=5, local_iters=10, max_gen=max_gen,
            track_history=True)

    mo_results = {}
    progress_bar = st.progress(0)
    status_text  = st.empty()
    start_time   = time.time()

    for k, (alg_name, opt_fn) in enumerate(algorithms.items()):
        status_text.text(f"Running {alg_name}…")
        pareto, history = opt_fn(m_subsys, sys_type)
        mo_results[alg_name] = {'pareto': pareto, 'history': history}
        progress_bar.progress((k + 1) / len(algorithms))

    status_text.success(
        f"Completed in {time.time() - start_time:.1f}s  •  "
        f"{sum(len(v['pareto']) for v in mo_results.values())} Pareto solutions found."
    )
    progress_bar.empty()

    # ── Optimal solution cards ──────────────────────────────────────────────
    st.header("🏆 Optimal Solution per Algorithm")
    st.markdown(
        "The **optimal** solution is selected from each algorithm's Pareto front "
        "using the shortest normalised Euclidean distance to the ideal point "
        "(minimum cost, maximum availability)."
    )

    opt_cols = st.columns(len(mo_results))
    for col, (alg_name, data) in zip(opt_cols, mo_results.items()):
        sols = data['pareto']
        opt_sol, opt_idx = _find_optimal(sols)
        with col:
            st.markdown(f"#### {alg_name}")
            if opt_sol:
                st.markdown(f"""
<div class="optimal-card">
<b>Cost (C_s):</b> {opt_sol['cost']}<br>
<b>Availability (A_s):</b> {opt_sol['availability']:.6f}<br>
<b>n-vector:</b> {opt_sol['n']}<br>
<b>r-vector:</b> {opt_sol['r']}<br>
<b>Pareto solutions:</b> {len(sols)}
</div>
""", unsafe_allow_html=True)
            else:
                st.warning("No feasible solutions found.")

    st.write("---")

    # ── Full Pareto solutions table ─────────────────────────────────────────
    st.header("📋 All Pareto-Optimal Solutions")
    st.markdown(
        "Every non-dominated solution returned by each algorithm, sorted by cost. "
        "The ⭐ column marks the **optimal** solution (best cost-availability balance)."
    )

    all_rows = []
    for alg_name, data in mo_results.items():
        sols = data['pareto']
        _, opt_idx = _find_optimal(sols)
        for i, sol in enumerate(sols):
            all_rows.append({
                "Algorithm":            alg_name,
                "⭐ Optimal":           "⭐" if i == opt_idx else "",
                "Cost (C_s)":           sol['cost'],
                "Availability (A_s)":   sol['availability'],
                "n-vector":             str(sol['n']),
                "r-vector":             str(sol['r']),
            })

    if all_rows:
        df_all = (
            pd.DataFrame(all_rows)
            .sort_values(["Algorithm", "Cost (C_s)"])
            .reset_index(drop=True)
        )
        st.dataframe(df_all, use_container_width=True, height=500)
        st.caption(
            f"Total non-dominated solutions across all algorithms: {len(df_all)}"
        )
    else:
        st.warning(
            "No feasible Pareto solutions found. "
            "Try increasing population size or generations."
        )

    st.write("---")

    # ── Summary statistics table ────────────────────────────────────────────
    st.header("📊 Summary Statistics per Algorithm")

    stat_rows = []
    for alg_name, data in mo_results.items():
        sols = data['pareto']
        if not sols:
            continue
        costs  = [s['cost'] for s in sols]
        avails = [s['availability'] for s in sols]
        opt_sol, _ = _find_optimal(sols)
        stat_rows.append({
            "Algorithm":                alg_name,
            "Pareto Solutions":         len(sols),
            "Min Cost":                 round(min(costs), 2),
            "Max Cost":                 round(max(costs), 2),
            "Min Availability":         round(min(avails), 6),
            "Max Availability":         round(max(avails), 6),
            "Optimal Cost":             opt_sol['cost'] if opt_sol else "—",
            "Optimal Availability":     round(opt_sol['availability'], 6) if opt_sol else "—",
        })

    if stat_rows:
        st.dataframe(
            pd.DataFrame(stat_rows).set_index("Algorithm"),
            use_container_width=True,
        )

    st.write("---")

    # ── Visualisations ──────────────────────────────────────────────────────
    st.header("📈 Visualisations")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='text-align:center'>Pareto Fronts</h4>",
                    unsafe_allow_html=True)
        fig_pf = plot_pareto_fronts(mo_results, system_choice, save=False)
        st.pyplot(fig_pf)

    with col2:
        st.markdown("<h4 style='text-align:center'>Convergence (Archive)</h4>",
                    unsafe_allow_html=True)
        fig_conv = plot_mo_convergence(mo_results, system_choice, save=False)
        st.pyplot(fig_conv)
