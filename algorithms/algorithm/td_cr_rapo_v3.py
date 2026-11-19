"""TD-CR-RAPO-V3 adapted to the HCMC TDVRP data layer.

This file intentionally keeps the V3 architecture from the supplied source:
strict NFE accounting, dual elite memory, split-aware block displacement,
multi-candidate POA pre-screening, recovery, elite AOS and share guards.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields
from time import perf_counter
from typing import Any, Iterable

from core.data.evaluator import HCMCEvaluationResult, HCMCTDVRPEvaluator
from core.data.models import RoadEdge, RoadNode, TDVRPInstance, TrafficOverlay
from core.data.solution import validate_solution
from core.models import Solution

from .greedy import build_greedy_solution
from .operators.discrete import cross_route_swap, insertion, relocate, reverse_two_opt, swap
from .operators.hcmc_pipeline import repair_validate_evaluate
from .td_cr_rapo import compute_customer_criticality, split_decoder


@dataclass
class TDCRRAPOV3Config:
    population_size: int = 10
    max_function_evaluations: int = 300
    max_iterations: int | None = None
    use_exact_parent_dedup: bool = False
    target_aro_share: float = 0.50
    min_aro_share: float = 0.47
    target_poa_share: float = 0.35
    max_poa_share: float = 0.38
    target_elite_share: float = 0.15
    max_elite_share: float = 0.15
    max_recovery_share: float = 0.08
    warmup_nfe_fraction: float = 0.20
    warmup_aro_target: float = 0.70
    s1_fraction: float = 0.10
    s2_fraction: float = 0.20
    p_memory_guidance: float = 0.25
    memory_lambda: float = 0.50
    memory_decay: float = 0.95
    poa_candidates_k: int = 6
    block_epsilon: float = 0.10
    aos_alpha: float = 0.20
    aos_epsilon: float = 0.10

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None = None) -> "TDCRRAPOV3Config":
        values = dict(mapping or {})
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def validate(self) -> None:
        if self.population_size < 2 or self.max_function_evaluations < self.population_size:
            raise ValueError("population_size >= 2 and NFE budget >= population_size are required")
        if self.min_aro_share < 0.47 or self.max_poa_share > 0.38 or self.max_elite_share > 0.15:
            raise ValueError("V3 architecture share guards were weakened")


@dataclass(frozen=True)
class Candidate:
    permutation: tuple[int, ...]
    solution: Solution
    evaluation: HCMCEvaluationResult

    @property
    def cost(self) -> float:
        return self.evaluation.fitness

    @property
    def routes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(route[1:-1]) for route in self.solution.routes)


@dataclass(frozen=True)
class TDCRRAPOV3Result:
    best_solution: Solution
    best_permutation: tuple[int, ...]
    best_evaluation: HCMCEvaluationResult
    convergence: list[float]
    runtime_seconds: float
    nfe: int
    spending: dict[str, int]
    metadata: dict[str, Any]

    @property
    def valid_architecture(self) -> bool:
        """Tên tương thích với kết quả TD-CR-RAPO cũ."""
        return self.metadata.get("architecture_status") == "PASS"

    @property
    def nfe_by_branch(self) -> dict[str, int]:
        """Tên tương thích cho thống kê chi tiêu NFE theo nhánh."""
        return self.spending


class EvaluationLedger:
    """One evaluator call is exactly one NFE."""

    def __init__(self, instance: TDVRPInstance, evaluator: HCMCTDVRPEvaluator, limit: int):
        self.instance, self.evaluator, self.limit = instance, evaluator, limit
        self.nfe = 0
        self.spending = {"initialization": 0, "aro": 0, "poa": 0, "elite": 0}
        self.recovery_nfe = 0
        self.best: Candidate | None = None
        self.convergence: list[float] = []

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.nfe)

    def evaluate(self, permutation: Iterable[int], phase: str, recovery: bool = False) -> Candidate | None:
        if not self.remaining:
            return None
        permutation = tuple(int(item) for item in permutation)
        solution = split_decoder(permutation, self.instance)
        evaluated = repair_validate_evaluate(solution, self.instance, self.evaluator)
        self.nfe += 1
        self.spending[phase] += 1
        if recovery:
            self.recovery_nfe += 1
        if evaluated is None:
            return None
        repaired, result = evaluated
        candidate = Candidate(tuple(item for route in repaired.routes for item in route[1:-1]), repaired, result)
        if self.best is None or candidate.cost < self.best.cost:
            self.best = candidate
        self.convergence.append(self.best.cost)
        return candidate


class DualEliteMemory:
    """Positive elite edge memory minus a decayed stagnation edge memory."""

    def __init__(self, n: int, decay: float, lam: float):
        self.n = n
        self.decay = decay
        self.lam = lam
        self.positive: dict[tuple[int, int], float] = {}
        self.negative: dict[tuple[int, int], float] = {}
        self.positive_updates = 0
        self.negative_updates = 0
        self.decay_events = 0

    @staticmethod
    def edges(permutation: tuple[int, ...]) -> list[tuple[int, int]]:
        return list(zip((0, *permutation), (*permutation, 0)))

    def decay_memory(self) -> None:
        self.positive = {edge: value * self.decay for edge, value in self.positive.items()}
        self.negative = {edge: value * self.decay for edge, value in self.negative.items()}
        self.decay_events += 1

    def update_positive(self, elites: list[Candidate]) -> None:
        self.decay_memory()
        for rank, candidate in enumerate(sorted(elites, key=lambda item: item.cost)):
            weight = 1.0 / (rank + 1)
            for edge in self.edges(candidate.permutation):
                self.positive[edge] = self.positive.get(edge, 0.0) + weight
        self.positive_updates += 1

    def update_negative(self, population: list[Candidate], stagnation: int, s1: int) -> None:
        if stagnation < s1:
            return
        for candidate in population:
            for edge in self.edges(candidate.permutation):
                self.negative[edge] = self.negative.get(edge, 0.0) + 1.0
        self.negative_updates += 1

    def guidance(self, edge: tuple[int, int]) -> float:
        return self.positive.get(edge, 0.0) - self.lam * self.negative.get(edge, 0.0)

    def best_edges(self, limit: int = 8) -> list[tuple[int, int]]:
        return [edge for edge, _ in sorted(self.positive.items(), key=lambda item: item[1], reverse=True)[:limit]]


def _hamming(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(a != b for a, b in zip(left, right))


def _critical_block_score(permutation: tuple[int, ...], start: int, length: int, criticality: dict[int, float]) -> float:
    block = permutation[start:start + length]
    segment = sum(criticality.get(customer, 0.0) for customer in block) / max(1, length)
    boundary = (criticality.get(permutation[start - 1], 0.0) if start else 0.0) + (criticality.get(permutation[min(len(permutation) - 1, start + length)], 0.0) if start + length < len(permutation) else 0.0)
    return 0.5 * segment + 0.5 * boundary / 2.0


def _block_move(permutation: tuple[int, ...], start: int, length: int, target: int, reverse: bool = False) -> tuple[int, ...]:
    values = list(permutation)
    block = values[start:start + length]
    del values[start:start + length]
    target = min(max(0, target), len(values))
    if reverse:
        block.reverse()
    values[target:target] = block
    return tuple(values)


def _approx_delta(permutation: tuple[int, ...], start: int, length: int, target: int, memory: DualEliteMemory) -> tuple[float, float, float]:
    moved = _block_move(permutation, start, length, target)
    old_edges = set(memory.edges(permutation)); new_edges = set(memory.edges(moved))
    memory_score = sum(memory.guidance(edge) for edge in new_edges - old_edges)
    novelty = len(new_edges - old_edges) / max(1, len(new_edges))
    # Travel is deliberately a cheap prescreening proxy. Exact TD cost is only
    # computed by EvaluationLedger after the top-q candidates are selected.
    delta = abs(target - start) + length * 0.01
    return delta, memory_score, novelty


class SearchRunTDCRRAPOV3:
    def __init__(self, instance: TDVRPInstance, graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]], traffic: TrafficOverlay, config: TDCRRAPOV3Config, seed: int, start_time: float = 0.0, departure_time_mode: str = "fixed", departure_time_step: float | None = None):
        self.instance, self.config, self.seed = instance, config, seed
        self.rng = random.Random(seed)
        self.evaluator = HCMCTDVRPEvaluator(
            instance,
            graph,
            traffic,
            start_time=start_time,
            departure_time_mode=departure_time_mode,
            departure_time_step=departure_time_step,
        )
        self.ledger = EvaluationLedger(instance, self.evaluator, config.max_function_evaluations)
        self.population: list[Candidate] = []
        self.elite_archive: list[Candidate] = []
        self.archive: Candidate | None = None
        self.memory = DualEliteMemory(instance.num_customers, config.memory_decay, config.memory_lambda)
        self.generation = 0
        self.stagnation_nfe = 0
        self.stagnation_max = 0
        self.s1 = max(1, round(config.s1_fraction * config.max_function_evaluations))
        self.s2 = max(2, round(config.s2_fraction * config.max_function_evaluations))
        self.recovery_cooldown = 0
        self.recovery_triggers = 0
        self.recovery_successes = 0
        self.max_recovery_budget = math.floor(config.max_recovery_share * config.max_function_evaluations)
        self.elite_size = max(2, math.ceil(0.20 * config.population_size))
        self.aos_names = ["relocate", "swap", "two_opt", "two_opt_star"]
        self.aos_scores = {name: 1.0 for name in self.aos_names}
        self.aos_usage = {name: 0 for name in self.aos_names}
        self.aos_success = {name: 0 for name in self.aos_names}
        self.improvement = {"aro": 0.0, "poa": 0.0, "elite": 0.0}
        self.diagnostics = {"criticality_calls": 0, "blocks_generated": 0, "blocks_evaluated": 0, "duplicate_candidates": 0, "recovery_triggers": 0, "recovery_successes": 0, "memory_guided_aro": 0}
        self.block_lengths: list[int] = []

    def initialize_population(self) -> None:
        base = list(range(1, self.instance.num_customers + 1))
        greedy_solution = build_greedy_solution(self.instance, (self.evaluator.nodes, self.evaluator.edges), self.evaluator.traffic, self.evaluator.start_time)
        greedy_candidate = self.ledger.evaluate(tuple(customer for route in greedy_solution.routes for customer in route[1:-1]), "initialization")
        if greedy_candidate:
            self.population.append(greedy_candidate)
        for index in range(1, self.config.population_size):
            permutation = base[:]
            self.rng.shuffle(permutation)
            candidate = self.ledger.evaluate(permutation, "initialization")
            if candidate:
                self.population.append(candidate)
        if len(self.population) < 2:
            raise RuntimeError("TD-CR-RAPO-V3 could not initialize a feasible population")
        self.archive = min(self.population, key=lambda item: item.cost)
        self.update_elites()

    def update_elites(self) -> None:
        ordered = sorted(self.population, key=lambda item: item.cost)
        distinct: list[Candidate] = []
        for candidate in ordered:
            if all(candidate.permutation != elite.permutation for elite in distinct):
                distinct.append(candidate)
            if len(distinct) >= self.elite_size:
                break
        self.elite_archive = distinct
        self.memory.update_positive(self.elite_archive)

    def _aro_explore(self, parent_index: int) -> bool:
        parent = self.population[parent_index]
        permutation = list(parent.permutation)
        n = len(permutation)
        if self.memory.positive and self.rng.random() < self.config.p_memory_guidance:
            self.diagnostics["memory_guided_aro"] += 1
            edges = self.memory.best_edges()
            edge = self.rng.choice(edges)
            permutation = [customer for customer in permutation if customer not in edge]
            position = self.rng.randrange(len(permutation) + 1)
            permutation[position:position] = list(edge)
        else:
            operation = self.rng.randrange(5)
            if operation == 0:
                left, right = self.rng.sample(range(n), 2); permutation[left], permutation[right] = permutation[right], permutation[left]
            elif operation == 1:
                guide = self.rng.choice(self.elite_archive or self.population).permutation
                width = self.rng.randint(max(2, n // 5), max(2, n // 2))
                start = self.rng.randrange(n - width + 1); block = list(guide[start:start + width]); permutation = [x for x in permutation if x not in block]; at = self.rng.randrange(len(permutation) + 1); permutation[at:at] = block
            elif operation == 2:
                for left, right in zip(self.rng.sample(range(n), min(n, 6))[::2], self.rng.sample(range(n), min(n, 6))[1::2]): permutation[left], permutation[right] = permutation[right], permutation[left]
            elif operation == 3:
                left, right = sorted(self.rng.sample(range(n), 2)); width = max(1, (right - left) // 2); permutation[left:left + width], permutation[right - width:right] = permutation[right - width:right], permutation[left:left + width]
            else:
                left, right = sorted(self.rng.sample(range(n), 2)); permutation[left:right + 1] = reversed(permutation[left:right + 1])
        candidate = self.ledger.evaluate(permutation, "aro")
        if candidate is None:
            return False
        improved = candidate.cost < parent.cost
        if improved:
            self.population[parent_index] = candidate; self.improvement["aro"] += parent.cost - candidate.cost
        if self.archive is None or candidate.cost < self.archive.cost:
            self.archive = candidate; self.stagnation_nfe = 0; self.update_elites()
        else:
            self.stagnation_nfe += 1; self.stagnation_max = max(self.stagnation_max, self.stagnation_nfe); self.memory.update_negative(self.population, self.stagnation_nfe, self.s1)
        return improved

    def _poa_block_displacement(self, parent_index: int, intensity: str, recovery: bool = False) -> bool:
        parent = self.population[parent_index]
        criticality = compute_customer_criticality(parent, self.instance, (self.evaluator.nodes, self.evaluator.edges), self.evaluator.traffic)
        self.diagnostics["criticality_calls"] += 1
        n = len(parent.permutation)
        if intensity == "small": low, high, top_q = 1, min(2, n), 2
        elif intensity == "medium": low, high, top_q = 2, min(4, n), 2
        else: low, high, top_q = max(2, n // 10), max(2, math.ceil(0.20 * n)), 3
        generated: list[tuple[tuple[float, float, float], tuple[int, ...]]] = []
        seen: set[tuple[int, ...]] = set()
        for _ in range(self.config.poa_candidates_k):
            length = self.rng.randint(low, max(low, high)); start = self.rng.randrange(n - length + 1); target = self.rng.randrange(n - length + 1)
            score = _critical_block_score(parent.permutation, start, length, criticality)
            if self.rng.random() > min(1.0, self.config.block_epsilon + score):
                start = self.rng.randrange(n - length + 1)
            candidate = _block_move(parent.permutation, start, length, target, reverse=intensity != "small" and self.rng.random() < 0.5)
            if candidate in seen:
                self.diagnostics["duplicate_candidates"] += 1; continue
            seen.add(candidate); generated.append((_approx_delta(parent.permutation, start, length, target, self.memory), candidate)); self.block_lengths.append(length); self.diagnostics["blocks_generated"] += 1
        generated.sort(key=lambda item: (item[0][0], -item[0][1], -item[0][2]))
        candidates: list[Candidate] = []
        for _, permutation in generated[:min(top_q, self.ledger.remaining)]:
            candidate = self.ledger.evaluate(permutation, "poa", recovery=recovery)
            if candidate: candidates.append(candidate); self.diagnostics["blocks_evaluated"] += 1
        if not candidates: return False
        best = min(candidates, key=lambda item: item.cost); improved = best.cost < parent.cost
        if improved:
            self.population[parent_index] = best; self.improvement["poa"] += parent.cost - best.cost
        if self.archive is None or best.cost < self.archive.cost:
            self.archive = best; self.stagnation_nfe = 0; self.update_elites()
        else:
            self.stagnation_nfe += len(candidates); self.stagnation_max = max(self.stagnation_max, self.stagnation_nfe)
        return improved

    def _recovery(self) -> bool:
        if self.ledger.remaining < 3 or self.ledger.recovery_nfe + 3 > self.max_recovery_budget: return False
        self.recovery_triggers += 1; self.diagnostics["recovery_triggers"] += 1
        index = min(range(len(self.population)), key=lambda i: self.population[i].cost)
        # One recovery event generates a large K-candidate batch and evaluates
        # at most three prescreened candidates, matching the V3 source.
        improved = self._poa_block_displacement(index, "large", recovery=True)
        if improved: self.recovery_successes += 1; self.diagnostics["recovery_successes"] += 1
        self.recovery_cooldown = max(10, round(0.05 * self.config.max_function_evaluations))
        return improved

    def _elite_aos_refine(self) -> bool:
        if not self.elite_archive: return False
        parent = self.elite_archive[0]; parent_index = self.population.index(parent) if parent in self.population else 0
        raw = [max(0.0, self.aos_scores[name]) for name in self.aos_names]; total = sum(raw) or 1.0
        probabilities = [(1 - len(raw) * self.config.aos_epsilon) * value / total + self.config.aos_epsilon for value in raw]
        operator = self.rng.choices(self.aos_names, weights=probabilities, k=1)[0]; self.aos_usage[operator] += 1
        route_solution = self.population[parent_index].solution
        functions = {"relocate": relocate, "swap": swap, "two_opt": reverse_two_opt, "two_opt_star": cross_route_swap}
        candidate_solution = functions[operator](route_solution, self.rng)
        candidate = self.ledger.evaluate(tuple(x for route in candidate_solution.routes for x in route[1:-1]), "elite")
        if candidate is None: return False
        reward = max(0.0, (parent.cost - candidate.cost) / max(abs(parent.cost), 1e-6)); global_best = self.archive is None or candidate.cost < self.archive.cost
        if global_best: reward *= 2
        self.aos_scores[operator] = (1 - self.config.aos_alpha) * self.aos_scores[operator] + self.config.aos_alpha * reward
        if candidate.cost < parent.cost:
            self.aos_success[operator] += 1; self.population[parent_index] = candidate; self.improvement["elite"] += parent.cost - candidate.cost
        if global_best: self.archive = candidate; self.stagnation_nfe = 0; self.update_elites()
        else: self.stagnation_nfe += 1
        return candidate.cost < parent.cost

    def _run_action(self, search_budget: int, min_aro: int, max_poa: int, max_elite: int, warmup_limit: int) -> None:
        used = self.ledger.nfe; remaining = self.ledger.remaining
        aro, poa, elite = self.ledger.spending["aro"], self.ledger.spending["poa"], self.ledger.spending["elite"]
        aro_needed = max(0, min_aro - aro); warmup = used < warmup_limit
        intensity = "small" if self.stagnation_nfe < self.s1 else ("medium" if self.stagnation_nfe < self.s2 else "large")
        poa_cost = 3 if intensity == "large" else 2
        can_poa = remaining >= poa_cost and remaining - poa_cost >= aro_needed and poa + poa_cost <= max_poa
        can_elite = not warmup and remaining >= 1 and remaining - 1 >= aro_needed and elite + 1 <= max_elite
        if not warmup and self.stagnation_nfe >= self.s2 and self.recovery_cooldown == 0 and can_poa:
            nfe_before_recovery = self.ledger.nfe
            self._recovery()
            # Recovery đã đánh giá ứng viên thì phải kết thúc action hiện tại,
            # kể cả khi không cải thiện, để không chi POA hai lần trong một lượt.
            if self.ledger.nfe > nfe_before_recovery:
                return
        if remaining <= aro_needed: action = "aro"
        elif warmup: action = "aro" if self.rng.random() < self.config.warmup_aro_target or not can_poa else "poa"
        else:
            efficiency = {"aro": self.improvement["aro"] / max(1, aro), "poa": self.improvement["poa"] / max(1, poa), "elite": self.improvement["elite"] / max(1, elite)}
            if not can_poa: efficiency["poa"] = -1
            if not can_elite: efficiency["elite"] = -1
            action = max(efficiency, key=efficiency.get)
            if self.stagnation_nfe >= self.s1 and can_poa: action = "poa"
        if action == "aro": self._aro_explore(self.rng.randrange(len(self.population)))
        elif action == "poa": self._poa_block_displacement(self.rng.randrange(len(self.population)), intensity)
        else: self._elite_aos_refine()

    def run(self) -> TDCRRAPOV3Result:
        started = perf_counter(); self.initialize_population(); init_nfe = self.ledger.nfe; search_budget = max(1, self.config.max_function_evaluations - init_nfe)
        min_aro = math.ceil(self.config.min_aro_share * search_budget); max_poa = math.floor(self.config.max_poa_share * search_budget); max_elite = math.floor(self.config.max_elite_share * search_budget); warmup = round(self.config.warmup_nfe_fraction * self.config.max_function_evaluations)
        while self.ledger.remaining and (self.config.max_iterations is None or self.generation < self.config.max_iterations):
            self.generation += 1
            if self.recovery_cooldown: self.recovery_cooldown -= 1
            self._run_action(search_budget, min_aro, max_poa, max_elite, warmup)
        best = self.archive or min(self.population, key=lambda item: item.cost); search_nfe = sum(self.ledger.spending[name] for name in ("aro", "poa", "elite")); aro_share = self.ledger.spending["aro"] / max(1, search_nfe); poa_share = self.ledger.spending["poa"] / max(1, search_nfe); elite_share = self.ledger.spending["elite"] / max(1, search_nfe)
        valid = validate_solution(best.solution, self.instance).feasible and best.evaluation.feasible
        architecture_valid = aro_share >= self.config.min_aro_share and poa_share <= self.config.max_poa_share and elite_share <= self.config.max_elite_share and self.ledger.recovery_nfe / max(1, self.ledger.nfe) <= self.config.max_recovery_share
        metadata = {"algorithm": "TD-CR-RAPO-V3", "objective": "total_route_duration", "encoding": "permutation + split decoder", "architecture_valid": architecture_valid, "architecture_status": "PASS" if architecture_valid else "FAIL", "feasible": valid, "completed_iterations": self.generation, "search_nfe": search_nfe, "aro_search_share": aro_share, "poa_search_share": poa_share, "elite_search_share": elite_share, "recovery_nfe": self.ledger.recovery_nfe, "recovery_share": self.ledger.recovery_nfe / max(1, self.ledger.nfe), "elite_archive_size": len(self.elite_archive), "mean_block_length": sum(self.block_lengths) / max(1, len(self.block_lengths)), "aos_usage": dict(self.aos_usage), "aos_success": dict(self.aos_success), "diagnostics": dict(self.diagnostics), "stagnation_max": self.stagnation_max}
        return TDCRRAPOV3Result(best.solution, best.permutation, best.evaluation, list(self.ledger.convergence), perf_counter() - started, self.ledger.nfe, dict(self.ledger.spending), metadata)


def optimize(instance: TDVRPInstance, graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]], traffic: TrafficOverlay, config: TDCRRAPOV3Config | dict[str, Any] | None = None, seed: int | None = None, start_time: float = 0.0, departure_time_mode: str = "fixed", departure_time_step: float | None = None) -> TDCRRAPOV3Result:
    cfg = config if isinstance(config, TDCRRAPOV3Config) else TDCRRAPOV3Config.from_mapping(config)
    cfg.validate()
    return SearchRunTDCRRAPOV3(
        instance,
        graph,
        traffic,
        cfg,
        42 if seed is None else int(seed),
        start_time=start_time,
        departure_time_mode=departure_time_mode,
        departure_time_step=departure_time_step,
    ).run()


class TDCRRAPOV3:
    name = "TD-CR-RAPO-V3"
    full_name = "Time-Dependent Criticality-Guided Rabbit-and-Pufferfish Optimization V3"
    hybrid_type = "ARO+POA+CR-V3"

    def optimize(self, instance, graph, traffic, config=None, seed=None, start_time=0.0, **kwargs):
        return optimize(instance, graph, traffic, config=config, seed=seed, start_time=start_time, **kwargs)
