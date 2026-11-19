from __future__ import annotations

from core.models import Solution, TDVRPProblem


def apply_exploration_operator(solution: Solution, problem: TDVRPProblem, rng) -> Solution:
    candidate = solution.copy()
    op = int(rng.integers(5))
    nonempty = [i for i, r in enumerate(candidate.routes) if len(r) > 2]
    if not nonempty:
        return candidate
    if op == 0:
        eligible = [i for i in nonempty if len(np_nodes(candidate.routes[i])) >= 2]
        if not eligible: return candidate
        i = int(rng.choice(eligible)); a, b = rng.choice(np_nodes(candidate.routes[i]), 2, replace=False); candidate.routes[i][a], candidate.routes[i][b] = candidate.routes[i][b], candidate.routes[i][a]
    elif op == 1:
        eligible = [i for i in nonempty if len(np_nodes(candidate.routes[i])) >= 2]
        if not eligible: return candidate
        i = int(rng.choice(eligible)); a, b = rng.choice(np_nodes(candidate.routes[i]), 2, replace=False); node = candidate.routes[i].pop(a); candidate.routes[i].insert(b, node)
    elif op == 2:
        eligible = [i for i in nonempty if len(np_nodes(candidate.routes[i])) >= 2]
        if not eligible: return candidate
        i = int(rng.choice(eligible)); a, b = sorted(rng.choice(np_nodes(candidate.routes[i]), 2, replace=False)); candidate.routes[i][a:b + 1] = reversed(candidate.routes[i][a:b + 1])
    elif op == 3 and len(nonempty) >= 1:
        i = int(rng.choice(nonempty)); targets = [j for j in range(len(candidate.routes)) if j != i];
        if targets:
            j = int(rng.choice(targets)); a = int(rng.choice(np_nodes(candidate.routes[i]))); node = candidate.routes[i].pop(a); candidate.routes[j].insert(int(rng.integers(1, len(candidate.routes[j])) + 1), node)
    else:
        if len(nonempty) >= 2:
            i, j = rng.choice(nonempty, 2, replace=False); a = int(rng.choice(np_nodes(candidate.routes[i]))); b = int(rng.choice(np_nodes(candidate.routes[j]))); candidate.routes[i][a], candidate.routes[j][b] = candidate.routes[j][b], candidate.routes[i][a]
    return candidate


def np_nodes(route):
    return list(range(1, len(route) - 1))
