import matplotlib.pyplot as plt
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

SAVE_DIR = os.path.join(os.path.dirname(current_dir), 'results')

def plot_box_diagrams(data, system_name, targets):
    n = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=(6, 4 * n))
    fig.tight_layout(pad=5.0)
    axes = np.atleast_1d(axes)
    labels = [f'({chr(97+i)})' for i in range(n)]

    for ax, As_min, label in zip(axes, targets, labels):
        algos = list(data[As_min].keys())
        box_data = [data[As_min][alg]['costs'] for alg in algos]
        
        ax.boxplot(box_data, labels=algos, patch_artist=True,
                   boxprops=dict(facecolor='#E8F0FE', color='#2C3E50'),
                   medianprops=dict(color='#E74C3C', linewidth=2))
        ax.set_title(f"{system_name} ($A_{{s,min}} = {As_min}$)")
        ax.set_ylabel("System cost")
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        ax.text(0.5, -0.2, label, transform=ax.transAxes, fontsize=12, ha='center', fontweight='bold')

    filename = f"{system_name.replace(' ', '_').lower()}_boxplots.png"
    filepath = os.path.join(SAVE_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    return fig

def plot_convergence_curves(data, system_name, targets):
    n = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=(6, 4 * n))
    fig.tight_layout(pad=5.0)
    axes = np.atleast_1d(axes)
    labels = [f'({chr(97+i)})' for i in range(n)]
    colors = {'DE': '#1f77b4', 'MRFO': '#ff7f0e', 'SFLA': '#2ca02c'}

    for ax, As_min, label in zip(axes, targets, labels):
        algos = list(data[As_min].keys())
        for alg in algos:
            history = data[As_min][alg]['history']
            if history:
                ax.plot(range(len(history)), history, label=alg, color=colors.get(alg, '#333'), linewidth=1.5)
        
        ax.set_title(f"{system_name} ($A_{{s,min}} = {As_min}$)")
        ax.set_xlabel("Generation number")
        ax.set_ylabel("System cost")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        ax.text(0.5, -0.25, label, transform=ax.transAxes, fontsize=12, ha='center', fontweight='bold')

    filename = f"{system_name.replace(' ', '_').lower()}_convergence.png"
    filepath = os.path.join(SAVE_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    return fig

def plot_bar_charts(data, system_name, targets):
    n = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=(6, 4 * n))
    fig.tight_layout(pad=5.0)
    axes = np.atleast_1d(axes)
    labels = [f'({chr(97+i)})' for i in range(n)]

    for ax, As_min, label in zip(axes, targets, labels):
        algos = list(data[As_min].keys())
        best_costs = [data[As_min][alg]['best_cost'] for alg in algos]
        
        ax.bar(algos, best_costs, color='#0052A5', edgecolor='black', alpha=0.8)
        ax.set_title(f"{system_name} ($A_{{s,min}} = {As_min}$)")
        ax.set_ylabel("System cost")
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        ax.text(0.5, -0.2, label, transform=ax.transAxes, fontsize=12, ha='center', fontweight='bold')

    filename = f"{system_name.replace(' ', '_').lower()}_barcharts.png"
    filepath = os.path.join(SAVE_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    return fig

def generate_all_visualisations(data, system_name):
    targets = sorted(list(data.keys()))
    plot_box_diagrams(data, system_name, targets)
    plot_convergence_curves(data, system_name, targets)
    plot_bar_charts(data, system_name, targets)