import numpy as np
from backend.core_math import evaluate_solution, decode_solution

def dominates(obj1, obj2):
    """
    Returns True if obj1 dominates obj2.
    Objective is to MINIMIZE cost (index 0) and MAXIMIZE availability (index 1).
    """
    cost1, avail1 = obj1
    cost2, avail2 = obj2
    
    better_or_equal = (cost1 <= cost2) and (avail1 >= avail2)
    strictly_better = (cost1 < cost2) or (avail1 > avail2)
    
    return better_or_equal and strictly_better

def optimize_mode(m, sys_type, pop_size=100, max_gen=200):
    """
    Multi-Objective Differential Evolution (MODE)
    Returns a sorted list of Pareto-optimal solutions.
    """
    dim = m * 2
    # Realistic hardware limits: maximum of 4 redundant components and 4 repair teams
    bounds = np.array([[1, 4]] * dim, dtype=float)
    
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, dim))
    objs = np.array([evaluate_solution(ind, m, sys_type) for ind in pop])
    
    CR = 0.9
    F = 0.8
    
    for gen in range(max_gen):
        for i in range(pop_size):
            x_i = pop[i]
            
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant = pop[a] + F * (pop[b] - pop[c])
            mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])
            
            cross_mask = np.random.rand(dim) < CR
            cross_mask[np.random.randint(0, dim)] = True
            trial = np.where(cross_mask, mutant, x_i)
            
            trial_obj = evaluate_solution(trial, m, sys_type)
            
            if dominates(trial_obj, objs[i]):
                pop[i] = trial
                objs[i] = trial_obj
            elif not dominates(objs[i], trial_obj):
                if np.random.rand() < 0.5:
                    pop[i] = trial
                    objs[i] = trial_obj

    pareto_front = []
    MAX_BUDGET = 2500  # Cost ceiling to prevent asymptotic spending

    for i in range(pop_size):
        is_dominated = False
        for j in range(pop_size):
            if i != j and dominates(objs[j], objs[i]):
                is_dominated = True
                break
        
        if not is_dominated:
            n_v, r_v = decode_solution(pop[i], m)
            cost, avail = objs[i]
            
            # Filters for realism: >=80% avail, under budget, strictly < 0.99999
            if 0.80 < avail < 0.99999 and cost <= MAX_BUDGET:
                # Format to exactly 5 decimal places to prevent float artifacting to 1.0
                formatted_avail = float(f"{avail:.5f}")
                
                sol = {
                    'n': n_v.tolist(), 
                    'r': r_v.tolist(), 
                    'cost': round(cost, 2), 
                    'avail': formatted_avail
                }
                
                if sol not in pareto_front:
                    pareto_front.append(sol)
                    
    pareto_front.sort(key=lambda x: x['cost'])
    
    # Clean up artifacts (same cost but worse availability)
    cleaned_front = []
    for sol in pareto_front:
        if not cleaned_front or sol['avail'] > cleaned_front[-1]['avail']:
            cleaned_front.append(sol)
            
    return cleaned_front