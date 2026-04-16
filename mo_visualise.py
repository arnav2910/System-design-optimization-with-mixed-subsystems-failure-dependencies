"""
mo_visualise.py  –  Visualisation helpers for multi-objective results
=====================================================================
Generates:
    • Pareto-front scatter plots  (cost vs availability)
    • Convergence traces for MO runs  (best cost / best availability per gen)
    • Side-by-side comparison of Pareto fronts across algorithms

All functions accept the same `mo_results` dict produced by the
multi-objective page in app.py and by run_mo_benchmark() in main.py.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(current_dir, 'results')
os.makedirs(SAVE_DIR, exist_ok=True)

# Consistent colours across all plots
ALG_COLORS = {
    'MO-NSGA2': '#1f77b4',
    'MO-MRFO':  '#ff7f0e',
    'MO-SFLA':  '#2ca02c',
}
ALG_MARKERS = {
    'MO-NSGA2': 'o',
    'MO-MRFO':  's',
    'MO-SFLA':  '^',
}


# ---------------------------------------------------------------------------
# 1.  Pareto-front scatter plots
# ---------------------------------------------------------------------------

def plot_pareto_fronts(mo_results, system_name, save=True):
    """
    Plot Pareto fronts for all algorithms on a single axes.

    Parameters
    ----------
    mo_results  : dict  {alg_name: {'pareto': list[dict], 'history': list}}
    system_name : str
    save        : bool – write PNG to results/ directory

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for alg_name, data in mo_results.items():
        solutions = data['pareto']
        if not solutions:
            continue
        costs  = [s['cost'] for s in solutions]
        avails = [s['availability'] for s in solutions]
        ax.scatter(
            costs, avails,
            label=alg_name,
            color=ALG_COLORS.get(alg_name, '#333333'),
            marker=ALG_MARKERS.get(alg_name, 'o'),
            s=60, alpha=0.85, edgecolors='black', linewidths=0.5
        )
        # Connect the Pareto front with a step line
        sorted_pairs = sorted(zip(costs, avails), key=lambda p: p[0])
        sc = [p[0] for p in sorted_pairs]
        sa = [p[1] for p in sorted_pairs]
        ax.step(sc, sa,
                where='post',
                color=ALG_COLORS.get(alg_name, '#333333'),
                linewidth=1.0, alpha=0.45, linestyle='--')

    ax.set_xlabel("System Cost ($C_s$)", fontsize=11)
    ax.set_ylabel("System Availability ($A_s$)", fontsize=11)
    ax.set_title(f"Pareto Front  –  {system_name}", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

    if save:
        fname = f"{system_name.replace(' ', '_').lower()}_pareto_fronts.png"
        fig.savefig(os.path.join(SAVE_DIR, fname), bbox_inches='tight', dpi=150)

    return fig


# ---------------------------------------------------------------------------
# 2.  MO convergence curves  (best cost & best availability per generation)
# ---------------------------------------------------------------------------

def plot_mo_convergence(mo_results, system_name, save=True):
    """
    Two-panel plot: left = best cost per generation, right = best avail.

    Parameters
    ----------
    mo_results  : dict  {alg_name: {'pareto': ..., 'history': list[tuple]}}
    system_name : str
    save        : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_cost, ax_avail) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"MO Convergence  –  {system_name}", fontsize=12, fontweight='bold')

    for alg_name, data in mo_results.items():
        history = data.get('history', [])
        if not history:
            continue
        gens   = [h[0] for h in history]
        costs  = [h[1] for h in history]
        avails = [h[2] for h in history]
        color  = ALG_COLORS.get(alg_name, '#333333')

        ax_cost.plot(gens, costs,  label=alg_name, color=color, linewidth=1.5)
        ax_avail.plot(gens, avails, label=alg_name, color=color, linewidth=1.5)

    for ax, ylabel in [(ax_cost, "Best Cost ($C_s$)"),
                       (ax_avail, "Best Availability ($A_s$)")]:
        ax.set_xlabel("Generation", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()

    if save:
        fname = f"{system_name.replace(' ', '_').lower()}_mo_convergence.png"
        fig.savefig(os.path.join(SAVE_DIR, fname), bbox_inches='tight', dpi=150)

    return fig


# ---------------------------------------------------------------------------
# 3.  Solution heatmap – n and r vectors along the Pareto front
# ---------------------------------------------------------------------------

def plot_solution_heatmap(mo_results, system_name, alg_name=None, save=True):
    """
    For a chosen algorithm (default: first in dict), visualise how the n and r
    vectors change along the Pareto front (sorted by cost).

    Returns
    -------
    matplotlib.figure.Figure  or  None if no solutions found
    """
    if alg_name is None:
        alg_name = next(iter(mo_results))

    solutions = mo_results[alg_name].get('pareto', [])
    if not solutions:
        return None

    solutions = sorted(solutions, key=lambda s: s['cost'])
    n_sols = len(solutions)
    m = len(solutions[0]['n'])

    n_matrix = np.array([s['n'] for s in solutions], dtype=float)
    r_matrix = np.array([s['r'] for s in solutions], dtype=float)

    fig, (ax_n, ax_r) = plt.subplots(1, 2, figsize=(max(8, m), max(4, n_sols * 0.5 + 2)))
    fig.suptitle(
        f"Pareto Solution Structure  –  {system_name}  ({alg_name})",
        fontsize=11, fontweight='bold'
    )

    ylabels = [f"C={s['cost']:.0f} / A={s['availability']:.4f}" for s in solutions]
    xlabels = [f"Sub {i+1}" for i in range(m)]

    for ax, mat, title in [(ax_n, n_matrix, "Component Count (n)"),
                           (ax_r, r_matrix, "Min Working (r)")]:
        im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=1, vmax=8)
        ax.set_xticks(range(m))
        ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_sols))
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_title(title, fontsize=10)
        # Annotate cells
        for row in range(n_sols):
            for col in range(m):
                ax.text(col, row, f"{int(mat[row, col])}",
                        ha='center', va='center', fontsize=7,
                        color='black' if mat[row, col] < 6 else 'white')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()

    if save:
        fname = (f"{system_name.replace(' ', '_').lower()}"
                 f"_{alg_name.lower()}_solution_heatmap.png")
        fig.savefig(os.path.join(SAVE_DIR, fname), bbox_inches='tight', dpi=150)

    return fig


# ---------------------------------------------------------------------------
# Convenience wrapper – generate all MO plots in one call
# ---------------------------------------------------------------------------

def generate_all_mo_visualisations(mo_results, system_name):
    """
    Generate and save all multi-objective visualisation plots.

    Called from main.py's run_mo_benchmark().
    """
    plot_pareto_fronts(mo_results, system_name, save=True)
    plot_mo_convergence(mo_results, system_name, save=True)
    # Heatmap for each algorithm
    for alg_name in mo_results:
        plot_solution_heatmap(mo_results, system_name,
                              alg_name=alg_name, save=True)
