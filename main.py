import numpy as np
import copy

from backend.core_math import calc_subsystem_availability, evaluate_solution
from backend.algorithms import optimize_de, optimize_mrfo, optimize_sfla

from utils.visualise import generate_all_visualisations

def verify_availability_formulas():
    print("\n--- Availability Formula Verification ---")
    a = calc_subsystem_availability(n=2, r=1, lam=0.07, mu=0.25, dep_type=0)
    print(f"Sub1 (L, n=2, r=1): A = {a:.6f}  (expected ≈ 0.9423)")
    a3 = calc_subsystem_availability(n=2, r=1, lam=0.02, mu=0.27, dep_type=1)
    print(f"Sub3 (W, n=2, r=1): A = {a3:.6f}")
    a6 = calc_subsystem_availability(n=2, r=1, lam=0.05, mu=0.26, dep_type=2)
    print(f"Sub6 (S, n=2, r=1): A = {a6:.6f}")
    print(f"  Ordering check (same n=2,r=1,lam=0.05,mu=0.26):")
    aL = calc_subsystem_availability(2, 1, 0.05, 0.26, dep_type=0)
    aW = calc_subsystem_availability(2, 1, 0.05, 0.26, dep_type=1)
    aS = calc_subsystem_availability(2, 1, 0.05, 0.26, dep_type=2)
    print(f"    Linear={aL:.4f}, Weak={aW:.4f}, Strong={aS:.4f}\n")

def run_benchmark(num_runs=10):
    np.random.seed(42)
    systems = [
        {'name': 'Complex bridge network system', 'm': 5,  'type': 'bridge'},
        {'name': 'Parallel-series system', 'm': 10, 'type': 'series_parallel'},
    ]
    avail_targets = [0.90, 0.95, 0.99]
    algorithms = {
        'DE':   lambda m, sys, av: optimize_de(m, sys, av, pop_size=100, max_gen=200, track_history=True),
        'MRFO': lambda m, sys, av: optimize_mrfo(m, sys, av, pop_size=100, max_gen=200, track_history=True),
        'SFLA': lambda m, sys, av: optimize_sfla(m, sys, av, pop_size=100, num_memeplexes=5, local_iters=10, max_gen=200, track_history=True),
    }

    print(f"\n{'='*110}")
    print(" RRAP BENCHMARK  —  DE vs MRFO vs SFLA  (Mellal et al. 2023)")
    print(f"{'='*110}")

    for cfg in systems:
        print(f"\n>>> SYSTEM: {cfg['name'].upper()}  (m = {cfg['m']})")
        
        plot_data = {0.90: {}, 0.95: {}, 0.99: {}}

        for As_min in avail_targets:
            print(f"\n  [ A_s,min = {As_min} ]")
            print(f"  {'─'*106}")
            print(f"  {'Algo':<6} │ {'n vector':<30} │ {'r vector':<30} │ "
                  f"{'A_s':<8} │ {'Best Cs':<9} │ {'Mean Cs':<9} │ {'Std'}")
            print(f"  {'─'*106}")

            for alg_name, opt_fn in algorithms.items():
                run_costs  = []
                best_cost  = np.inf
                best_n     = None
                best_r     = None
                best_avail = 0.0
                best_history = []

                for run in range(num_runs):
                    n_v, r_v, history = opt_fn(cfg['m'], cfg['type'], As_min)
                    
                    x_flat = np.zeros(cfg['m'] * 2)
                    x_flat[0::2] = n_v
                    x_flat[1::2] = r_v
                    cost, avail = evaluate_solution(x_flat, cfg['m'], cfg['type'])

                    run_costs.append(cost)
                    if avail >= As_min and cost < best_cost:
                        best_cost  = cost
                        best_n     = n_v.tolist()
                        best_r     = r_v.tolist()
                        best_avail = avail
                        best_history = copy.deepcopy(history)

                mean_c = np.mean(run_costs)
                std_c  = np.std(run_costs)
                cost_str = f"{best_cost:.0f}" if best_cost < np.inf else "INFEASIBLE"

                plot_data[As_min][alg_name] = {
                    'costs': run_costs,
                    'best_cost': best_cost,
                    'history': best_history
                }

                print(f"  {alg_name:<6} │ {str(best_n):<30} │ {str(best_r):<30} │ "
                      f"{best_avail:<8.4f} │ {cost_str:<9} │ {mean_c:<9.1f} │ {std_c:.2f}")
            print(f"  {'─'*106}")
            
        print(f"\n  Generating and saving visualisations for {cfg['name']}...")
        generate_all_visualisations(plot_data, cfg['name'])

if __name__ == "__main__":
    verify_availability_formulas()
    run_benchmark(num_runs=10)