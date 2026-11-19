"""Adapter cho các baseline GA/DPSO/VND từ bộ thực nghiệm Poryos.

Các baseline gốc dùng ``core.tdvrp_common.TDVRPProblem``. Module này giữ
nguyên logic tìm kiếm của chúng, nhưng chuyển kết quả về dạng
``HCMCEvaluationResult`` của project hiện tại để app và ExperimentRunner dùng
được chung route trace, objective và thống kê.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from algorithms.dpso import DPSO
from algorithms.ga import GA
from algorithms.ga_ls import GALS
from algorithms.ils_vnd import ILSVND
from algorithms.aro_original import AROOriginal
from algorithms.poa_original import POAOriginal
from algorithms.pso_vns import PSOVNS
from algorithms.td_rapo_pufferfish import TDRAPOPufferfish
from core.data.evaluator import HCMCEvaluationResult, HCMCTDVRPEvaluator
from core.data.models import RoadEdge, RoadNode, TDVRPInstance, TrafficOverlay
from core.models import Solution
from core.tdvrp_common import TDVRPProblem


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ALGORITHMS = (
    "GA",
    "GA-LS",
    "ARO-Original",
    "POA-Original",
    "DPSO",
    "ILS-VND",
    "PSO-VNS",
    "RAPO-Pufferfish",
)


@dataclass
class BaselineAdapterResult:
    """Kết quả chung để ExperimentRunner không cần biết backend cũ."""

    best_solution: Solution
    best_evaluation: HCMCEvaluationResult
    convergence: list[float]
    runtime_seconds: float
    evaluations: int
    metadata: dict[str, object]

    @property
    def nfe(self) -> int:
        return self.evaluations


@lru_cache(maxsize=16)
def _load_legacy_problem(instance_path: str, vehicle_limit: int) -> TDVRPProblem:
    """Tạo problem tương thích một lần cho mỗi instance và chính sách fleet."""
    return TDVRPProblem(
        instance_path,
        collection_root=ROOT / "data" / "TDVRP",
        vehicle_limit=vehicle_limit,
        search_route_limit=max(2 * vehicle_limit, vehicle_limit + 1),
    )


def run_baseline(
    algorithm: str,
    instance: TDVRPInstance,
    graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]],
    traffic: TrafficOverlay,
    *,
    instance_path: str | Path,
    population_size: int,
    nfe_budget: int,
    seed: int,
    initial_population: Iterable[Iterable[int]] | None = None,
    start_time: float = 0.0,
) -> BaselineAdapterResult:
    """Chạy một baseline với cùng objective và trả route trace chuẩn hóa."""
    if algorithm not in BASELINE_ALGORITHMS:
        raise ValueError(f"Baseline chưa được hỗ trợ: {algorithm}")

    # Dùng số route tối đa bằng số khách hàng để khớp fleet không giới hạn
    # của RAPO-2709 trong benchmark nghiên cứu hiện tại.
    legacy_problem = _load_legacy_problem(str(Path(instance_path).resolve()), instance.num_customers)
    config = {
        "population_size": population_size,
        "max_function_evaluations": nfe_budget,
    }
    implementations = {
        "GA": GA(),
        "GA-LS": GALS(),
        "ARO-Original": AROOriginal(),
        "POA-Original": POAOriginal(),
        "DPSO": DPSO(),
        "ILS-VND": ILSVND(),
        "PSO-VNS": PSOVNS(),
        "RAPO-Pufferfish": TDRAPOPufferfish(),
    }
    if algorithm == "ILS-VND":
        # ILS-VND là thuật toán trajectory-based, không dùng population size.
        config.pop("population_size")

    raw = implementations[algorithm].optimize(
        legacy_problem,
        config=config,
        seed=seed,
        initial_population=initial_population,
    )
    routes = [list(route) for route in raw.routes]
    solution = Solution(routes)
    # Baseline source dùng chính sách unlimited nonempty routes giống
    # RAPO-2709. Bản Poryos gốc vẫn lưu num_vehicles=4, nên dùng bản view
    # có đúng số route của nghiệm khi materialize trace; không sửa dữ liệu.
    trace_instance = replace(instance, num_vehicles=len(routes))
    evaluator = HCMCTDVRPEvaluator(
        trace_instance,
        graph,
        traffic,
        start_time=start_time,
        departure_time_mode="optimize",
        departure_time_step=instance.sample_step,
    )
    evaluation = evaluator.evaluate(solution)
    solution.fitness = evaluation.fitness
    solution.objective = evaluation.total_route_duration
    solution.total_distance = evaluation.total_distance
    solution.total_travel_time = evaluation.total_travel_time
    solution.is_feasible = evaluation.feasible
    metadata = dict(raw.metadata)
    metadata.update(
        {
            "algorithm": algorithm,
            "adapter": "poryos_baseline_adapter",
            "objective": "total_route_duration",
            "shared_exact_evaluator": True,
            "source_backend": "ThucNghiemV7",
        }
    )
    return BaselineAdapterResult(
        best_solution=solution,
        best_evaluation=evaluation,
        convergence=list(raw.convergence_curve),
        runtime_seconds=float(raw.runtime_seconds),
        evaluations=int(raw.nfe),
        metadata=metadata,
    )
