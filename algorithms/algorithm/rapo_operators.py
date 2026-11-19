"""Permutation moves for the ARO (broad) and POA (local) families.

No scoring here. Route-aware moves use the parent's stored decoded routes;
the resulting permutation is always re-split by the shared evaluator.
The original ARO/POA equations remain available as core proposals, adapted
through rank keys. These discrete adaptations are not claims of a new standard.
"""
import numpy as np


def rank_keys(permutation):
    keys = np.empty(len(permutation))
    for rank, customer in enumerate(permutation):
        keys[customer-1] = (rank + 0.5) / len(permutation)
    return keys


def from_keys(keys):
    return tuple(int(i)+1 for i in np.argsort(keys, kind='stable'))


def _swap(p, rng):
    a, b = rng.choice(len(p), 2, replace=False)
    p[a], p[b] = p[b], p[a]


def _relocate(p, a, b):
    customer = p.pop(a)
    p.insert(b, customer)


ARO_OPERATORS = ('aro_core', 'guided_order_injection', 'multi_swap', 'long_relocate',
                 'large_inversion', 'segment_exchange', 'destroy_repair')
POA_OPERATORS = ('poa_core', 'elite_path_relink', 'swap', 'relocate', 'two_opt',
                 'inter_route_relocate', 'inter_route_exchange')
FAMILIES = {'ARO': ARO_OPERATORS, 'POA': POA_OPERATORS}
ELITE_OPERATORS = ('elite_path_relink', 'relocate', 'two_opt', 'swap',
                   'inter_route_relocate', 'inter_route_exchange')
STRONG_OPERATORS = frozenset(ARO_OPERATORS[1:])


def make_candidate(operator, parent, population, index, generation, progress_limit, rng, config):
    p = list(parent.permutation)
    n = len(p)
    if operator in ('aro_core', 'poa_core'):
        # Lazy imports avoid a cycle between the standalone solvers and engine.
        from algorithms.aro import ARO
        from algorithms.poa import POA
        positions = np.array([rank_keys(x.permutation) for x in population])
        values = np.array([x.cost for x in population])
        solver = ARO if operator == 'aro_core' else POA
        proposal = next(solver.propose(positions, values, index, generation,
                                      progress_limit, rng))
        target = from_keys(np.clip(proposal, 0, 1))
        if operator == 'aro_core':
            p = list(target)
            if tuple(p) == parent.permutation:
                # Rank ties can erase a continuous step; diversify the rabbit.
                rng.shuffle(p)
        else:
            # POA direction guides one local swap, not a wholesale reordering.
            mismatch = [i for i in range(n) if p[i] != target[i]]
            if mismatch:
                a = int(rng.choice(mismatch))
                b = p.index(target[a])
                p[a], p[b] = p[b], p[a]
            else:
                _swap(p, rng)
    elif operator == 'guided_order_injection':
        # Copy one long, ordered block from a good peer and repair by stable
        # insertion. This transfers useful adjacency information while still
        # producing a broad ARO move and consuming exactly one official NFE.
        peers = [x for i, x in enumerate(population) if i != index]
        ranked = sorted(peers, key=lambda x: x.cost)
        guide = ranked[int(rng.integers(max(1, (len(ranked)+1)//2)))]
        low = min(n, max(2, n//5))
        high = min(n, max(low, n//2))
        width = int(rng.integers(low, high+1))
        start = int(rng.integers(n-width+1))
        block = list(guide.permutation[start:start+width])
        block_set = set(block)
        remainder = [customer for customer in p if customer not in block_set]
        insert_at = int(rng.integers(len(remainder)+1))
        p = remainder[:insert_at] + block + remainder[insert_at:]
    elif operator == 'multi_swap':
        # Disjoint swaps guarantee several customers change position.
        indices = rng.permutation(n)[:2 * max(2, n//3)]
        for a, b in zip(indices[::2], indices[1::2]):
            p[a], p[b] = p[b], p[a]
    elif operator == 'long_relocate':
        a, b = rng.choice(n, 2, replace=False)
        if abs(a-b) < n//2:
            b = (a + n//2) % n
        _relocate(p, int(a), int(b))
    elif operator == 'large_inversion':
        length = int(rng.integers(max(4, n//2), n+1))
        a = int(rng.integers(n-length+1))
        p[a:a+length] = reversed(p[a:a+length])
    elif operator == 'segment_exchange':
        width = max(2, n//3)
        a = int(rng.integers(n-2*width+1))
        b = int(rng.integers(a+width, n-width+1))
        p[a:a+width], p[b:b+width] = p[b:b+width], p[a:a+width]
    elif operator == 'destroy_repair':
        fraction = rng.uniform(config.destroy_fraction_min, config.destroy_fraction_max)
        count = max(1, min(n-1, int(np.ceil(n*fraction))))
        removed = set(int(i) for i in rng.choice(n, count, replace=False))
        values = [p[i] for i in sorted(removed)]
        p = [c for i, c in enumerate(p) if i not in removed]
        rng.shuffle(values)
        # Stochastic reinsertion, NOT best insertion with hidden evaluations.
        for c in values:
            p.insert(int(rng.integers(len(p)+1)), c)
    elif operator == 'elite_path_relink':
        # Move several positions toward the best current permutation. Unlike
        # crossover this is an intensification step and remains strictly greedy.
        guides = [x for i, x in enumerate(population)
                  if i != index and x.permutation != parent.permutation]
        guide = min(guides, key=lambda x: x.cost).permutation if guides else None
        if guide is None:
            _swap(p, rng)
            guide = tuple(p)
        mismatch = [i for i in range(n) if p[i] != guide[i]]
        steps = min(len(mismatch), max(1, n//12))
        for _ in range(steps):
            mismatch = [i for i in range(n) if p[i] != guide[i]]
            if not mismatch:
                break
            a = int(rng.choice(mismatch))
            b = p.index(guide[a])
            p[a], p[b] = p[b], p[a]
    elif operator == 'swap':
        _swap(p, rng)
    elif operator == 'relocate':
        a = int(rng.integers(n))
        distance = int(rng.choice((-2, -1, 1, 2)))
        b = max(0, min(n-1, a+distance))
        if b == a:
            b = a-1 if a else 1
        _relocate(p, a, b)
    elif operator == 'two_opt':
        # Reverse a segment inside one decoded route. The old implementation
        # reversed only 2--4 arbitrary permutation positions and often crossed
        # Split boundaries, so it was a weak TDVRP neighbourhood for n=50.
        eligible = [route for route in parent.routes if len(route) >= 2]
        if not eligible:
            _swap(p, rng)
        else:
            route = eligible[int(rng.integers(len(eligible)))]
            left, right = sorted(int(x) for x in rng.choice(len(route), 2, replace=False))
            positions = [p.index(customer) for customer in route[left:right+1]]
            a, b = min(positions), max(positions)
            p[a:b+1] = reversed(p[a:b+1])
    elif operator in ('inter_route_relocate', 'inter_route_exchange'):
        if len(parent.routes) < 2:
            _swap(p, rng)
        else:
            ra, rb = rng.choice(len(parent.routes), 2, replace=False)
            ca = int(rng.choice(parent.routes[ra]))
            cb = int(rng.choice(parent.routes[rb]))
            a, b = p.index(ca), p.index(cb)
            if operator == 'inter_route_exchange':
                p[a], p[b] = p[b], p[a]
            else:
                p.pop(a)
                p.insert(p.index(cb), ca)
    else:
        raise ValueError(f'Unknown operator: {operator}')
    candidate = tuple(p)
    if sorted(candidate) != list(range(1, n+1)):
        raise AssertionError(f'{operator} violated permutation invariants')
    return candidate
