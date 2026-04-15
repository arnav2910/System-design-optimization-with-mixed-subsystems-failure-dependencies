import math
import numpy as np
from backend.data import SYSTEM_DATA

def calc_subsystem_availability(n, r, lam, mu, dep_type):
    n, r = int(n), int(r)
    n = max(1, n)
    r = max(1, min(r, n))
    ratio = mu / lam
    sum1 = 0.0
    for j in range(1, n - r + 1):
        base = (r ** j) * (ratio ** j)
        if dep_type == 0:
            term = base
        elif dep_type == 1:
            term = base * math.pow(math.factorial(j), -0.5)
        else:
            term = base * math.pow(math.factorial(j), 0.5)
        sum1 += term

    sum2 = 0.0
    r_pow_nr = r ** (n - r)
    r_fact   = math.factorial(r)

    for j in range(n - r + 1, n + 1):
        denom_fact = math.factorial(n - j)
        coeff = r_pow_nr * r_fact / denom_fact
        base  = coeff * (ratio ** j)

        if dep_type == 0:
            term = base
        elif dep_type == 1:
            term = base * math.pow(math.factorial(j), -0.5)
        else:
            term = base * math.pow(math.factorial(j), 0.5)
        sum2 += term

    denominator = 1.0 + sum1 + sum2
    return 1.0 - (1.0 / denominator)

def calc_system_availability(A_subs, sys_type):
    if sys_type == 'bridge':
        A1, A2, A3, A4, A5 = A_subs[:5]
        As = (  A1*A4
              + A2*A5
              + A2*A3*A4
              + A1*A3*A5
              + 2*A1*A2*A3*A4*A5
              - A2*A3*A4*A5
              - A1*A3*A4*A5
              - A1*A2*A4*A5
              - A1*A2*A3*A5
              - A1*A2*A3*A4 )
        return As
    else:
        result = 1.0
        for a in A_subs:
            result *= a
        return result

def decode_solution(x, m):
    n_vars = np.round(x[0::2]).astype(int)
    r_vars = np.round(x[1::2]).astype(int)
    for i in range(m):
        n_vars[i] = max(1, n_vars[i])
        r_vars[i] = max(1, min(r_vars[i], n_vars[i]))
    return n_vars, r_vars

def evaluate_solution(x, m, sys_type):
    n_vars, r_vars = decode_solution(x, m)
    cost   = 0.0
    A_subs = []
    for i in range(m):
        dep_type, lam, mu, c_comp, c_rep = SYSTEM_DATA[i + 1]
        ni, ri = n_vars[i], r_vars[i]
        cost += c_comp * ni + c_rep * ri
        A_subs.append(calc_subsystem_availability(ni, ri, lam, mu, dep_type))
    As = calc_system_availability(A_subs, sys_type)
    return cost, As

def penalised_objective(x, m, sys_type, As_min):
    n_vars, r_vars = decode_solution(x, m)
    cost   = 0.0
    A_subs = []
    penalty = 0.0

    for i in range(m):
        dep_type, lam, mu, c_comp, c_rep = SYSTEM_DATA[i + 1]
        ni, ri = n_vars[i], r_vars[i]
        raw_n = int(round(x[2*i]))
        raw_r = int(round(x[2*i+1]))
        if raw_r > raw_n:
            penalty += 1e5 * (raw_r - raw_n)
        cost   += c_comp * ni + c_rep * ri
        A_subs.append(calc_subsystem_availability(ni, ri, lam, mu, dep_type))

    As = calc_system_availability(A_subs, sys_type)

    if As < As_min:
        penalty += 1e8 * (As_min - As)

    return cost + penalty