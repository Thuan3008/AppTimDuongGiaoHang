"""Verify ablations use the intended branches and the original NFE ledger."""

import json
from types import SimpleNamespace

import pytest

from algorithms import rapo_2709
from algorithms.rapo_ablation import ABLATION_VARIANTS, optimize_ablation
from experiments.rapo_ablation import VARIANTS, organize_existing, result_paths


class FakeEvaluator:
    def __init__(self, instance, graph, traffic, **kwargs):
        pass

    def _validate_permutation(self, order):
        if sorted(order) != [1, 2, 3]:
            raise ValueError("invalid permutation")

    def evaluate(self, order):
        order = tuple(order)
        self._validate_permutation(order)
        fitness = float(sum(index * customer for index, customer in enumerate(order, 1)))
        return rapo_2709.RAPO2709Candidate(order, (order,), (), fitness)

    def materialize(self, candidate):
        return (
            SimpleNamespace(routes=[candidate.permutation]),
            SimpleNamespace(fitness=candidate.fitness, feasible=True),
        )


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_ablation_branch_nfe_and_shared_initial_population(monkeypatch, variant):
    monkeypatch.setattr(rapo_2709, "_ExactSplitEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        rapo_2709,
        "poa_restructure",
        lambda parent, evaluator, config, rng: tuple(reversed(parent.permutation)),
    )
    initial = [(1, 2, 3), (1, 3, 2), (2, 1, 3), (3, 2, 1)]
    config = rapo_2709.RAPO2709Config(
        population_size=4,
        max_function_evaluations=40,
        elite_budget_fraction=0.5,
    )
    result = optimize_ablation(
        SimpleNamespace(num_customers=3), ({}, {}), None,
        variant=variant, config=config, seed=5, initial_population=initial,
    )

    assert result.evaluations == sum(result.nfe_by_branch.values()) == 40
    assert result.nfe_by_branch["initialization"] == 4
    assert result.convergence[:4] == [14.0, 13.0, 13.0, 10.0]
    if variant == "ARO-only":
        assert result.nfe_by_branch["ARO"] == 36
        assert result.nfe_by_branch["POA"] == result.nfe_by_branch["Elite"] == 0
    elif variant == "POA-only":
        assert result.nfe_by_branch["POA"] == 36
        assert result.nfe_by_branch["ARO"] == result.nfe_by_branch["Elite"] == 0
    else:
        assert result.nfe_by_branch["Elite"] > 0
        assert set(result.operator_scores.values()) == {1.0}
        assert result.metadata["elite_operator_selection"] == "uniform_fixed"


def test_original_solver_remains_callable_with_same_initial_population(monkeypatch):
    monkeypatch.setattr(rapo_2709, "_ExactSplitEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        rapo_2709,
        "poa_restructure",
        lambda parent, evaluator, config, rng: tuple(reversed(parent.permutation)),
    )
    result = rapo_2709.optimize(
        SimpleNamespace(num_customers=3), ({}, {}), None,
        config=rapo_2709.RAPO2709Config(population_size=4, max_function_evaluations=40),
        seed=5,
        initial_population=[(1, 2, 3), (1, 3, 2), (2, 1, 3), (3, 2, 1)],
    )
    assert result.evaluations == 40
    assert result.metadata["variant"] == "RAPO-2709"


def test_unknown_ablation_variant_is_rejected():
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        optimize_ablation(None, None, None, variant="ARO-Original")


def test_organize_existing_only_after_complete_run(tmp_path):
    output = tmp_path / "ablation.json"
    checkpoint = tmp_path / "ablation.checkpoint.jsonl"
    protocol = {"variants": list(VARIANTS), "seeds": [0]}
    rows = [
        {"variant": variant, "seed": 0, "fitness": 12.0, "nfe": 4, "runtime_seconds": 1.5}
        for variant in VARIANTS
    ]
    checkpoint.write_text(
        json.dumps({"type": "header", "protocol": protocol}) + "\n"
        + json.dumps({"type": "run", "row": rows[0]}) + "\n",
        encoding="utf-8",
    )
    output.write_text(json.dumps({"protocol": protocol, "runs": rows}), encoding="utf-8")
    for kind in ("runs", "summary", "paired"):
        (tmp_path / f"ablation_{kind}.csv").write_text("test\n", encoding="utf-8")

    with pytest.raises(ValueError, match="chưa hoàn tất"):
        organize_existing(output)
    assert checkpoint.is_file()
    assert not (tmp_path / "ablation").exists()

    with checkpoint.open("a", encoding="utf-8") as handle:
        for row in rows[1:]:
            handle.write(json.dumps({"type": "run", "row": row}) + "\n")
    directory = organize_existing(output)
    assert directory == tmp_path / "ablation"
    assert result_paths(output)[1].is_file()
    assert result_paths(output)[2].is_file()
    assert "[4/4]" in (directory / "run.log").read_text(encoding="utf-8")
    assert not checkpoint.exists()
