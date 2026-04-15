import numpy as np
from backend.core_math import decode_solution, penalised_objective

def optimize_de(m, sys_type, As_min, pop_size=100, max_gen=200, track_history=False):
    dim    = m * 2
    bounds = np.array([[1, 8]] * dim, dtype=float)
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, dim))
    fitness = np.array([penalised_objective(ind, m, sys_type, As_min) for ind in pop])

    best_idx = np.argmin(fitness)
    best_x   = pop[best_idx].copy()
    best_fit = fitness[best_idx]
    history = [best_fit] if track_history else None
    CR = 0.9

    for gen in range(max_gen):
        for i in range(pop_size):
            F = np.random.uniform(0.5, 1.0)
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
            mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

            cross_mask = np.random.rand(dim) < CR
            cross_mask[np.random.randint(0, dim)] = True
            trial = np.where(cross_mask, mutant, pop[i])

            trial_fit = penalised_objective(trial, m, sys_type, As_min)

            if trial_fit <= fitness[i]:
                pop[i]     = trial
                fitness[i] = trial_fit
                if trial_fit < best_fit:
                    best_x   = trial.copy()
                    best_fit = trial_fit

        if track_history:
            history.append(best_fit)

    n_vars, r_vars = decode_solution(best_x, m)
    return n_vars, r_vars, history

def optimize_mrfo(m, sys_type, As_min, pop_size=100, max_gen=200, track_history=False):
    dim    = m * 2
    bounds = np.array([[1, 8]] * dim, dtype=float)
    X       = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, dim))
    fitness = np.array([penalised_objective(ind, m, sys_type, As_min) for ind in X])

    best_idx = np.argmin(fitness)
    X_best   = X[best_idx].copy()
    fit_best = fitness[best_idx]
    history = [fit_best] if track_history else None
    S = 2.0

    for t in range(max_gen):
        rand_val = np.random.rand()
        for i in range(pop_size):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            if rand_val < 0.5:
                if i == 0:
                    X_new = X[i] + r1 * (X_best - X[i])
                else:
                    X_new = X[i] + r1 * (X[i-1] - X[i]) + r2 * (X_best - X[i])
            else:
                if t / max_gen < 0.5:
                    X_rand = np.random.uniform(bounds[:, 0], bounds[:, 1], dim)
                    X_ref  = X_rand
                else:
                    X_ref  = X_best

                beta  = 2 * np.exp(r1 * (max_gen - t) / max_gen) * np.sin(2 * np.pi * r1)
                X_new = X_ref + r1 * (X_ref - X[i]) + beta * (X_ref - X[i])

            X_new = np.clip(X_new, bounds[:, 0], bounds[:, 1])
            fit_new = penalised_objective(X_new, m, sys_type, As_min)

            if fit_new < fitness[i]:
                X[i]       = X_new
                fitness[i] = fit_new
                if fit_new < fit_best:
                    X_best   = X_new.copy()
                    fit_best = fit_new

        for i in range(pop_size):
            r3    = np.random.rand(dim)
            r4    = np.random.rand(dim)
            X_new = X[i] + S * (r3 * X_best - r4 * X[i])
            X_new = np.clip(X_new, bounds[:, 0], bounds[:, 1])
            fit_new = penalised_objective(X_new, m, sys_type, As_min)

            if fit_new < fitness[i]:
                X[i]       = X_new
                fitness[i] = fit_new
                if fit_new < fit_best:
                    X_best   = X_new.copy()
                    fit_best = fit_new

        if track_history:
            history.append(fit_best)

    n_vars, r_vars = decode_solution(X_best, m)
    return n_vars, r_vars, history

def optimize_sfla(m_subsys, sys_type, As_min, pop_size=100, num_memeplexes=5, local_iters=10, max_gen=200, track_history=False):
    dim    = m_subsys * 2
    bounds = np.array([[1, 8]] * dim, dtype=float)
    q      = pop_size // num_memeplexes

    frogs   = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, dim))
    fitness = np.array([penalised_objective(f, m_subsys, sys_type, As_min) for f in frogs])

    sort_idx = np.argsort(fitness)
    frogs    = frogs[sort_idx]
    fitness  = fitness[sort_idx]

    X_global_best  = frogs[0].copy()
    fit_global_best = fitness[0]
    history = [fit_global_best] if track_history else None

    for shuffle in range(max_gen):
        for p in range(num_memeplexes):
            mem_idx = list(range(p, pop_size, num_memeplexes))
            if len(mem_idx) < 2:
                continue

            for _ in range(local_iters):
                sub_fitness = fitness[mem_idx]
                local_best_pos  = mem_idx[np.argmin(sub_fitness)]
                local_worst_pos = mem_idx[np.argmax(sub_fitness)]

                X_lb = frogs[local_best_pos]
                X_lw = frogs[local_worst_pos]

                step  = np.random.rand(dim) * (X_lb - X_lw)
                X_new = np.clip(X_lw + step, bounds[:, 0], bounds[:, 1])
                fit_new = penalised_objective(X_new, m_subsys, sys_type, As_min)

                if fit_new < fitness[local_worst_pos]:
                    frogs[local_worst_pos]   = X_new
                    fitness[local_worst_pos] = fit_new
                else:
                    step  = np.random.rand(dim) * (X_global_best - X_lw)
                    X_new = np.clip(X_lw + step, bounds[:, 0], bounds[:, 1])
                    fit_new = penalised_objective(X_new, m_subsys, sys_type, As_min)

                    if fit_new < fitness[local_worst_pos]:
                        frogs[local_worst_pos]   = X_new
                        fitness[local_worst_pos] = fit_new
                    else:
                        X_rand = np.random.uniform(bounds[:, 0], bounds[:, 1], dim)
                        frogs[local_worst_pos]   = X_rand
                        fitness[local_worst_pos] = penalised_objective(X_rand, m_subsys, sys_type, As_min)

        sort_idx = np.argsort(fitness)
        frogs    = frogs[sort_idx]
        fitness  = fitness[sort_idx]

        if fitness[0] < fit_global_best:
            fit_global_best = fitness[0]
            X_global_best   = frogs[0].copy()

        if track_history:
            history.append(fit_global_best)

    n_vars, r_vars = decode_solution(X_global_best, m_subsys)
    return n_vars, r_vars, history