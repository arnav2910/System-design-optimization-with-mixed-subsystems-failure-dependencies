"""
mo_algorithms.py  –  Multi-Objective Redundancy Allocation
===========================================================
Extends the single-objective RRAP formulation to a bi-objective problem:

    Objective 1 : Minimise total system cost  C_s
    Objective 2 : Maximise system availability A_s  (i.e. minimise 1 – A_s)

Three multi-objective solvers are provided, each mirroring the structure of
its single-objective counterpart in algorithms.py so that the rest of the
codebase (app.py, main.py) can use them without any changes:

    mo_nsga2   –  Non-dominated Sorting Genetic Algorithm II  (NSGA-II)
    mo_mrfo_mo –  Multi-objective Manta Ray Foraging Optimisation
    mo_sfla_mo –  Multi-objective Shuffled Frog Leaping Algorithm

All three return:
    pareto_solutions  : list[dict]  –  each entry has keys
                            'n', 'r', 'cost', 'availability', 'x'
    history           : list[tuple] –  (gen, best_cost, best_avail) per gen
                        (empty list when track_history=False)
"""

import numpy as np
from backend.core_math import decode_solution, evaluate_solution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluate_mo(x, m, sys_type):
    """Return (cost, 1 - availability) for a raw decision vector x."""
    cost, avail = evaluate_solution(x, m, sys_type)
    return cost, 1.0 - avail          # both objectives to be minimised


def _dominates(obj_a, obj_b):
    """True iff a dominates b (both objectives minimised)."""
    return (obj_a[0] <= obj_b[0] and obj_a[1] <= obj_b[1] and
            (obj_a[0] < obj_b[0] or obj_a[1] < obj_b[1]))


def _non_dominated_sort(pop_objs):
    """
    Fast non-dominated sorting.

    Parameters
    ----------
    pop_objs : list of (f1, f2) tuples

    Returns
    -------
    fronts : list of lists – each inner list contains indices into pop_objs
    """
    n = len(pop_objs)
    domination_count = [0] * n
    dominated_by = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(pop_objs[i], pop_objs[j]):
                dominated_by[i].append(j)
            elif _dominates(pop_objs[j], pop_objs[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    return [f for f in fronts if f]       # drop trailing empty front


def _crowding_distance(front_indices, pop_objs, n_obj=2):
    """Compute crowding distance for individuals in a front."""
    n = len(front_indices)
    dist = {i: 0.0 for i in front_indices}
    if n <= 2:
        for i in front_indices:
            dist[i] = np.inf
        return dist

    for obj in range(n_obj):
        sorted_front = sorted(front_indices, key=lambda i: pop_objs[i][obj])
        f_min = pop_objs[sorted_front[0]][obj]
        f_max = pop_objs[sorted_front[-1]][obj]
        dist[sorted_front[0]] = np.inf
        dist[sorted_front[-1]] = np.inf
        span = f_max - f_min if f_max != f_min else 1e-12
        for k in range(1, n - 1):
            dist[sorted_front[k]] += (
                pop_objs[sorted_front[k + 1]][obj] -
                pop_objs[sorted_front[k - 1]][obj]
            ) / span
    return dist


def _extract_pareto(population, objectives, m):
    """Return list of solution dicts on the first (Pareto) front."""
    fronts = _non_dominated_sort(objectives)
    results = []
    for idx in fronts[0]:
        x = population[idx]
        n_vars, r_vars = decode_solution(x, m)
        cost, avail = objectives[idx][0], 1.0 - objectives[idx][1]
        results.append({
            'n': n_vars.tolist(),
            'r': r_vars.tolist(),
            'cost': round(cost, 2),
            'availability': round(avail, 6),
            'x': x.tolist(),
        })
    # Sort Pareto front by cost ascending for easier reading
    results.sort(key=lambda d: d['cost'])
    return results


def _bounds(m):
    return np.array([[1, 8]] * (m * 2), dtype=float)


# ---------------------------------------------------------------------------
# 1.  NSGA-II
# ---------------------------------------------------------------------------

def mo_nsga2(m, sys_type, pop_size=100, max_gen=200, track_history=False):
    """
    Non-dominated Sorting Genetic Algorithm II (NSGA-II).

    Uses DE-style mutation + binomial crossover as the variation operator,
    which works well on continuous/integer decision spaces.

    Parameters
    ----------
    m            : int   – number of subsystems
    sys_type     : str   – 'bridge' or 'series_parallel'
    pop_size     : int   – population size (even number)
    max_gen      : int   – number of generations
    track_history: bool  – record (gen, best_cost, best_avail) each gen

    Returns
    -------
    pareto_solutions : list[dict]
    history          : list[tuple]  (empty if track_history=False)
    """
    dim = m * 2
    bnds = _bounds(m)
    CR = 0.9

    # Initialise
    pop = np.random.uniform(bnds[:, 0], bnds[:, 1], (pop_size, dim))
    objs = [_evaluate_mo(ind, m, sys_type) for ind in pop]

    history = []

    for gen in range(max_gen):
        # --- Variation (DE mutation + crossover) ---
        fronts = _non_dominated_sort(objs)
        cd = {}
        for front in fronts:
            cd.update(_crowding_distance(front, objs))

        # Rank each individual: (front_rank, -crowding_distance)
        front_rank = {}
        for rank, front in enumerate(fronts):
            for idx in front:
                front_rank[idx] = rank

        offspring_pop = []
        offspring_objs = []

        for i in range(pop_size):
            F = np.random.uniform(0.4, 0.9)
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = np.random.choice(idxs, 3, replace=False)

            # Tournament selection for base vector: prefer lower front rank,
            # then higher crowding distance
            def tournament(p, q):
                if front_rank[p] < front_rank[q]:
                    return p
                elif front_rank[p] > front_rank[q]:
                    return q
                else:
                    return p if cd.get(p, 0) >= cd.get(q, 0) else q

            best_t = tournament(a, b)
            mutant = pop[i] + F * (pop[best_t] - pop[i]) + F * (pop[a] - pop[b])
            mutant = np.clip(mutant, bnds[:, 0], bnds[:, 1])

            mask = np.random.rand(dim) < CR
            mask[np.random.randint(0, dim)] = True
            trial = np.where(mask, mutant, pop[i])

            offspring_pop.append(trial)
            offspring_objs.append(_evaluate_mo(trial, m, sys_type))

        # --- Environmental selection (elitism via combined population) ---
        combined_pop = list(pop) + offspring_pop
        combined_objs = objs + offspring_objs

        all_fronts = _non_dominated_sort(combined_objs)
        all_cd = {}
        for front in all_fronts:
            all_cd.update(_crowding_distance(front, combined_objs))

        selected = []
        for front in all_fronts:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
            else:
                remaining = pop_size - len(selected)
                sorted_front = sorted(
                    front,
                    key=lambda i: -all_cd.get(i, 0)
                )
                selected.extend(sorted_front[:remaining])
                break

        pop = np.array([combined_pop[i] for i in selected])
        objs = [combined_objs[i] for i in selected]

        if track_history:
            # Best cost on Pareto front this generation
            pareto_front_objs = [combined_objs[i] for i in all_fronts[0]]
            best_cost_gen = min(o[0] for o in pareto_front_objs)
            best_avail_gen = 1.0 - min(o[1] for o in pareto_front_objs)
            history.append((gen, best_cost_gen, best_avail_gen))

    pareto_solutions = _extract_pareto(pop, objs, m)
    return pareto_solutions, history


# ---------------------------------------------------------------------------
# 2.  Multi-objective MRFO
# ---------------------------------------------------------------------------

def mo_mrfo(m, sys_type, pop_size=100, max_gen=200, track_history=False):
    """
    Multi-objective Manta Ray Foraging Optimisation (MO-MRFO).

    Maintains an external archive of non-dominated solutions.
    The guide individual is chosen from the archive via crowding-distance
    tournament to balance exploration and coverage of the Pareto front.

    Returns
    -------
    pareto_solutions : list[dict]
    history          : list[tuple]
    """
    dim = m * 2
    bnds = _bounds(m)
    S = 2.0

    X = np.random.uniform(bnds[:, 0], bnds[:, 1], (pop_size, dim))
    fitness = [_evaluate_mo(x, m, sys_type) for x in X]

    # Archive: list of (x, obj)
    archive_X = list(X)
    archive_O = list(fitness)

    def _prune_archive(ax, ao, max_size=100):
        """Keep only non-dominated solutions; truncate by crowding if needed."""
        fronts = _non_dominated_sort(ao)
        nd_idx = fronts[0]
        ax = [ax[i] for i in nd_idx]
        ao = [ao[i] for i in nd_idx]
        if len(ax) > max_size:
            cd = _crowding_distance(list(range(len(ax))), ao)
            order = sorted(range(len(ax)), key=lambda i: -cd[i])
            ax = [ax[i] for i in order[:max_size]]
            ao = [ao[i] for i in order[:max_size]]
        return ax, ao

    def _select_guide(ax, ao):
        """Tournament selection from archive weighted by crowding distance."""
        if len(ax) == 1:
            return ax[0]
        cd = _crowding_distance(list(range(len(ax))), ao)
        a, b = np.random.choice(len(ax), 2, replace=False)
        winner = a if cd[a] >= cd[b] else b
        return ax[winner]

    history = []

    for t in range(max_gen):
        X_guide = _select_guide(archive_X, archive_O)
        rand_val = np.random.rand()

        for i in range(pop_size):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            if rand_val < 0.5:
                X_new = (X[i] + r1 * (X_guide - X[i])
                         if i == 0
                         else X[i] + r1 * (X[i - 1] - X[i]) + r2 * (X_guide - X[i]))
            else:
                if t / max_gen < 0.5:
                    X_ref = np.random.uniform(bnds[:, 0], bnds[:, 1], dim)
                else:
                    X_ref = X_guide
                beta = 2 * np.exp(r1 * (max_gen - t) / max_gen) * np.sin(2 * np.pi * r1)
                X_new = X_ref + r1 * (X_ref - X[i]) + beta * (X_ref - X[i])

            X_new = np.clip(X_new, bnds[:, 0], bnds[:, 1])
            obj_new = _evaluate_mo(X_new, m, sys_type)

            # Greedy replacement: accept if new solution dominates current
            if _dominates(obj_new, fitness[i]):
                X[i] = X_new
                fitness[i] = obj_new

            archive_X.append(X_new)
            archive_O.append(obj_new)

        # Somersault phase
        for i in range(pop_size):
            r3 = np.random.rand(dim)
            r4 = np.random.rand(dim)
            X_new = np.clip(X[i] + S * (r3 * X_guide - r4 * X[i]),
                            bnds[:, 0], bnds[:, 1])
            obj_new = _evaluate_mo(X_new, m, sys_type)
            if _dominates(obj_new, fitness[i]):
                X[i] = X_new
                fitness[i] = obj_new
            archive_X.append(X_new)
            archive_O.append(obj_new)

        archive_X, archive_O = _prune_archive(archive_X, archive_O)

        if track_history:
            best_cost_gen = min(o[0] for o in archive_O)
            best_avail_gen = 1.0 - min(o[1] for o in archive_O)
            history.append((t, best_cost_gen, best_avail_gen))

    pareto_solutions = _extract_pareto(
        np.array(archive_X), archive_O, m
    )
    return pareto_solutions, history


# ---------------------------------------------------------------------------
# 3.  Multi-objective SFLA
# ---------------------------------------------------------------------------

def mo_sfla(m, sys_type, pop_size=100, num_memeplexes=5,
            local_iters=10, max_gen=200, track_history=False):
    """
    Multi-objective Shuffled Frog Leaping Algorithm (MO-SFLA).

    Uses Pareto dominance for local frog comparison and maintains an
    external archive of non-dominated solutions across shuffles.

    Returns
    -------
    pareto_solutions : list[dict]
    history          : list[tuple]
    """
    dim = m * 2
    bnds = _bounds(m)
    q = pop_size // num_memeplexes

    frogs = np.random.uniform(bnds[:, 0], bnds[:, 1], (pop_size, dim))
    fitness = [_evaluate_mo(f, m, sys_type) for f in frogs]

    # Archive
    archive_X = list(frogs)
    archive_O = list(fitness)

    def _prune_archive(ax, ao, max_size=100):
        if not ax:
            return ax, ao
        fronts = _non_dominated_sort(ao)
        nd_idx = fronts[0]
        ax = [ax[i] for i in nd_idx]
        ao = [ao[i] for i in nd_idx]
        if len(ax) > max_size:
            cd = _crowding_distance(list(range(len(ax))), ao)
            order = sorted(range(len(ax)), key=lambda i: -cd[i])
            ax = [ax[i] for i in order[:max_size]]
            ao = [ao[i] for i in order[:max_size]]
        return ax, ao

    def _global_best(ax, ao):
        """Return the archive member with lowest cost (for leaping step)."""
        best_idx = min(range(len(ax)), key=lambda i: ao[i][0])
        return ax[best_idx]

    history = []

    for shuffle in range(max_gen):
        # Sort frogs: primary key = front rank, secondary = crowding distance
        fronts = _non_dominated_sort(fitness)
        rank = {}
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r
        cd_all = {}
        for front in fronts:
            cd_all.update(_crowding_distance(front, fitness))

        sort_key = lambda i: (rank[i], -cd_all.get(i, 0))
        order = sorted(range(pop_size), key=sort_key)
        frogs = frogs[order]
        fitness = [fitness[i] for i in order]

        X_gb = _global_best(archive_X, archive_O)

        for p in range(num_memeplexes):
            mem_idx = list(range(p, pop_size, num_memeplexes))
            if len(mem_idx) < 2:
                continue

            for _ in range(local_iters):
                # Best in memeplex = first (already sorted)
                lb_pos = mem_idx[0]
                lw_pos = mem_idx[-1]        # worst = last in sorted memeplex

                X_lb = frogs[lb_pos]
                X_lw = frogs[lw_pos]

                # Leap towards local best
                step = np.random.rand(dim) * (X_lb - X_lw)
                X_new = np.clip(X_lw + step, bnds[:, 0], bnds[:, 1])
                obj_new = _evaluate_mo(X_new, m, sys_type)

                if _dominates(obj_new, fitness[lw_pos]):
                    frogs[lw_pos] = X_new
                    fitness[lw_pos] = obj_new
                    archive_X.append(X_new)
                    archive_O.append(obj_new)
                else:
                    # Leap towards global best
                    step = np.random.rand(dim) * (X_gb - X_lw)
                    X_new = np.clip(X_lw + step, bnds[:, 0], bnds[:, 1])
                    obj_new = _evaluate_mo(X_new, m, sys_type)

                    if _dominates(obj_new, fitness[lw_pos]):
                        frogs[lw_pos] = X_new
                        fitness[lw_pos] = obj_new
                        archive_X.append(X_new)
                        archive_O.append(obj_new)
                    else:
                        # Random regeneration
                        X_rand = np.random.uniform(bnds[:, 0], bnds[:, 1], dim)
                        frogs[lw_pos] = X_rand
                        fitness[lw_pos] = _evaluate_mo(X_rand, m, sys_type)

        archive_X, archive_O = _prune_archive(archive_X, archive_O)

        if track_history:
            best_cost_gen = min(o[0] for o in archive_O)
            best_avail_gen = 1.0 - min(o[1] for o in archive_O)
            history.append((shuffle, best_cost_gen, best_avail_gen))

    pareto_solutions = _extract_pareto(
        np.array(archive_X), archive_O, m
    )
    return pareto_solutions, history
