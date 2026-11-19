"""Independent ablation variants for RAPO-2709.

The production solver in rapo_2709.py is not modified. This module reuses its
exact evaluator, Split, search moves and NFE ledger, while controlling which
search branches are enabled. It deliberately does not represent ARO-Original
or POA-Original.
"""

from __future__ import annotations

import math
import random
from time import perf_counter
from typing import Iterable

from . import rapo_2709 as base


ABLATION_VARIANTS = ("ARO-only", "POA-only", "RAPO-no-AOS")


def optimize_ablation(
    instance,
    graph,
    traffic,
    *,
    variant: str,
    config: base.RAPO2709Config | None = None,
    seed: int | None = None,
    initial_population: Iterable[Iterable[int]] | None = None,
) -> base.RAPO2709Result:
    """Run one variant with RAPO-2709's evaluator and strict NFE protocol."""
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Biến thể ablation không được hỗ trợ: {variant}")
    config = config or base.RAPO2709Config()
    config.validate()
    rng = random.Random(config.seed if seed is None else seed)
    evaluator = base._ExactSplitEvaluator(
        instance,
        graph,
        traffic,
        earliest_departure=config.earliest_departure,
        exact_split=config.exact_split,
    )
    ledger = base._NFELedger(evaluator, config.max_function_evaluations)
    started = perf_counter()

    base_order = list(range(1, instance.num_customers + 1))
    if initial_population is None:
        initial_orders = [
            tuple(rng.sample(base_order, len(base_order)))
            for _ in range(config.population_size)
        ]
        initialization_policy = "independent_random"
    else:
        initial_orders = [tuple(int(customer) for customer in order) for order in initial_population]
        if len(initial_orders) != config.population_size:
            raise ValueError("Quần thể P0 phải có đúng population_size hoán vị")
        for order in initial_orders:
            evaluator._validate_permutation(order)
        initialization_policy = "provided_common_population"

    population: list[base.RAPO2709Candidate] = []
    archive: base.RAPO2709Candidate | None = None
    convergence: list[float] = []
    for permutation in initial_orders:
        candidate = ledger.evaluate(permutation, "initialization")
        if candidate is None:
            raise RuntimeError("Không thể khởi tạo quần thể RAPO ablation khả thi")
        population.append(candidate)
        if archive is None or candidate.fitness < archive.fitness:
            archive = candidate
        convergence.append(archive.fitness)

    assert archive is not None
    scores = {name: 1.0 for name in base.ELITE_OPERATORS}
    uses = {name: 0 for name in base.ELITE_OPERATORS}
    successes = {name: 0 for name in base.ELITE_OPERATORS}
    elite_quota = (
        math.floor(config.elite_budget_fraction * config.max_function_evaluations)
        if variant == "RAPO-no-AOS" else 0
    )

    while ledger.used < ledger.budget:
        if variant == "ARO-only":
            available = ["ARO"]
        elif variant == "POA-only":
            available = ["POA"]
        else:
            available = ["ARO", "POA"]
            if ledger.by_branch["Elite"] < elite_quota:
                available.append("Elite")
        branch = rng.choice(available)

        if branch == "ARO":
            parent_index = rng.randrange(len(population))
            partner_index = rng.choice([index for index in range(len(population)) if index != parent_index])
            parent = population[parent_index]
            child_order = base.order_crossover(
                parent.permutation, population[partner_index].permutation, rng
            )
            child = ledger.evaluate(child_order, "ARO")
            if child is not None and child.fitness < parent.fitness:
                population[parent_index] = child
                if child.fitness < archive.fitness:
                    archive = child

        elif branch == "POA":
            parent_index = rng.randrange(len(population))
            parent = population[parent_index]
            child_order = base.poa_restructure(parent, evaluator, config, rng)
            child = ledger.evaluate(child_order, "POA")
            if child is not None and child.fitness < parent.fitness:
                population[parent_index] = child
                if child.fitness < archive.fitness:
                    archive = child

        else:
            parent_index = base._best_index(population)
            parent = population[parent_index]
            # The production AOS only controls Elite operators; branch
            # selection itself is already uniform in RAPO-2709.
            operator = rng.choices(
                base.ELITE_OPERATORS,
                weights=[1.0] * len(base.ELITE_OPERATORS),
                k=1,
            )[0]
            uses[operator] += 1
            child_order = base._elite_move(parent.permutation, operator, rng)
            child = ledger.evaluate(child_order, "Elite")
            if child is not None:
                if child.fitness < parent.fitness:
                    population[parent_index] = child
                    successes[operator] += 1
                if child.fitness < archive.fitness:
                    archive = child

        convergence.append(archive.fitness)

    solution, evaluation = evaluator.materialize(archive)
    return base.RAPO2709Result(
        solution,
        evaluation,
        archive.permutation,
        convergence,
        perf_counter() - started,
        ledger.used,
        dict(ledger.by_branch),
        scores,
        uses,
        successes,
        {
            "variant": variant,
            "objective": "sum of exact optimal route durations",
            "split_decoder": (
                "exact minimum-duration order-first split-second"
                if config.exact_split else "time-zero proxy order-first split-second"
            ),
            "departure_time_mode": "exact_ndcpwlf",
            "fleet_policy": "unlimited_nonempty_routes",
            "initialization_policy": initialization_policy,
            "elite_quota": elite_quota,
            "elite_operator_selection": "uniform_fixed" if variant == "RAPO-no-AOS" else "disabled",
            "route_count": len(archive.routes),
            "strict_nfe": ledger.used == ledger.budget,
        },
    )
