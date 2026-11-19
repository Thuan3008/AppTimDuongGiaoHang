"""Separate wall-time comparison; original 290-NFE protocol is untouched."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import traceback
from time import perf_counter
from importlib.metadata import version

from algorithms.timed_runtime import TimeBudget, TimeExpired
from algorithms.td_constructive import ALGORITHMS, TDHeuristicConfig, solve, _Search, _TimeLimit
from core.data.load_poryos import load_poryos, find_poryos_bks, load_poryos_bks
from core.tdvrp_common import TDVRPProblem
from experiments.time_checkpoint import CampaignStore, atomic_json, valid_result

ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / 'data/TDVRP/hong_kong/n=25/poryos-hong_kong-n25-hyb/bpr-heavy/poryos-hong_kong-n25-hyb-bpr-heavy.vrp.json'
STOCHASTIC = ('TD-ERAPO-Time', 'ARO-Original', 'POA-Original', 'GA', 'GA-LS')
ALL = (*STOCHASTIC, *ALGORITHMS, 'Greedy')


def timed_greedy(problem, seconds):
    """Original append-only greedy rule, with a singleton-completed fallback."""
    search = _Search(problem, TDHeuristicConfig(time_limit_seconds=seconds))
    search.singletons()
    pending = set(range(1, problem.dimension + 1))
    closed, current = [], ()
    reason = 'all_customers_assigned'
    try:
        while pending:
            search.check_time()
            choices = []
            old = search.costs[current][0] if current else 0
            for c in sorted(pending):
                candidate = current + (c,)
                value = search.evaluate_route(candidate)
                if value is not None:
                    choices.append((value[0] - old, value[0], c, candidate))
            if not choices:
                if not current:
                    raise ValueError('No feasible customer for Greedy')
                closed.append(current)
                current = ()
                continue
            _, _, c, current = min(choices)
            pending.remove(c)
            search.remember(tuple(closed) + (current,) + tuple((c,) for c in sorted(pending)), 'greedy')
    except _TimeLimit:
        reason = 'time_limit'
    return {'routes': [list(r) for r in search.best], 'duration': search.best_cost,
            'nfe': None, 'termination': reason, 'history': search.history,
            'statistics': {'route_evaluations': search.evaluations},
            'search_runtime_seconds': perf_counter() - search.start}


def run_job(algorithm, seed, seconds, instance_path=INSTANCE):
    payload = json.loads(instance_path.read_text(encoding='utf-8'))
    n = payload['num_customers']
    if algorithm == 'TD-ERAPO-Time':
        from algorithms.td_erapo_timed import optimize, TDERAPOTimedConfig
        instance, graph, traffic = load_poryos(instance_path)
        rng = random.Random(seed)
        p0 = [tuple(rng.sample(list(range(1, n + 1)), n)) for _ in range(10)]
        result = optimize(instance, graph, traffic,
                          TDERAPOTimedConfig(time_limit_seconds=seconds), seed=seed,
                          initial_population=p0)
        data = {'routes': [list(r[1:-1]) for r in result.best_solution.routes],
                'duration': result.best_evaluation.fitness, 'nfe': result.nfe,
                'termination': result.metadata['termination'],
                'history': result.metadata['time_history'], 'metadata': result.metadata,
                'runtime_seconds': result.runtime_seconds}
        problem = TDVRPProblem(instance_path, collection_root=ROOT / 'data/TDVRP', vehicle_limit=n)
    else:
        problem = TDVRPProblem(instance_path, collection_root=ROOT / 'data/TDVRP', vehicle_limit=n)
        if algorithm in ALGORITHMS:
            result = solve(problem, algorithm, TDHeuristicConfig(time_limit_seconds=seconds))
            data = result.to_dict()
            data['nfe'] = None
        elif algorithm == 'Greedy':
            data = timed_greedy(problem, seconds)
        else:
            from algorithms.ga_timed import GA
            from algorithms.ga_ls_timed import GALS
            from algorithms.aro_original_timed import AROOriginal
            from algorithms.poa_original_timed import POAOriginal
            implementations = {'GA': GA, 'GA-LS': GALS, 'ARO-Original': AROOriginal,
                               'POA-Original': POAOriginal}
            rng = random.Random(seed)
            p0 = [tuple(rng.sample(list(range(1, n + 1)), n)) for _ in range(10)]
            budget = TimeBudget(seconds)
            problem.time_budget = budget
            try:
                implementations[algorithm]().optimize(problem,
                    {'population_size': 10, 'max_function_evaluations': 10**9},
                    seed=seed, initial_population=p0)
                raise RuntimeError('Safety NFE guard reached before deadline; do not report as a timed run')
            except TimeExpired:
                pass
            if budget.best is None:
                raise RuntimeError('Deadline before any feasible solution was evaluated')
            data = {'routes': [list(r) for r in budget.best.routes], 'duration': budget.best.cost,
                    'nfe': budget.nfe, 'history': budget.history,
                    'termination': 'time_limit', 'search_runtime_seconds': perf_counter() - budget.start}
    validation_start = perf_counter()
    details = problem.validate(data['routes'])
    duration = math.fsum(e[0] for e in details)
    if not math.isclose(duration, data['duration'], abs_tol=1e-6, rel_tol=1e-12):
        raise ValueError('Recomputed Duration differs from reported value')
    bks = load_poryos_bks(find_poryos_bks(instance_path))
    within = [h for h in data['history'] if h['seconds'] <= seconds]
    data.update(algorithm=algorithm, seed=seed, feasible=True, status='completed',
                duration=duration, gap_to_bks_percent=100 * (duration / bks['cost'] - 1),
                departure_times=[e[1] for e in details], return_times=[e[2] for e in details],
                validation_seconds=perf_counter() - validation_start,
                best_duration_within_deadline=min((h['duration'] for h in within), default=None),
                time_limit_seconds=seconds)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--seeds', default='0-29', help='Inclusive range, e.g. 0-29 or 0-0')
    parser.add_argument('--algorithms', nargs='+', choices=ALL, default=list(ALL))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be finite and positive')
    bounds = [int(s) for s in args.seeds.split('-')]
    if len(bounds) != 2 or not 0 <= bounds[0] <= bounds[1]:
        parser.error('--seeds must be an increasing nonnegative range')
    seeds = list(range(bounds[0], bounds[1] + 1))
    output = args.output or ROOT / f'results/time_comparison_n25_{args.seconds:g}s_seeds{bounds[0]}_{bounds[1]}'
    source_files = [p for directory in ('algorithms', 'core') for p in sorted((ROOT / directory).rglob('*.py'))]
    source_files += [Path(__file__).resolve(), ROOT / 'experiments/time_checkpoint.py', INSTANCE, find_poryos_bks(INSTANCE)]
    payload = json.loads(INSTANCE.read_text(encoding='utf-8'))
    source_files += [ROOT / 'data/TDVRP' / payload['td'][key]['path'] for key in ('graph', 'traffic')]
    protocol = {'instance': str(INSTANCE), 'seconds': args.seconds, 'seeds': seeds,
                'algorithms': args.algorithms, 'population_size': 10,
                'stopping': 'cooperative time limit; no 290-NFE stop',
                'python': sys.version, 'platform': platform.platform(),
                'mamut_routing_lib': version('mamut-routing-lib'),
                'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    manifest = output / 'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Code, data or configuration changed. Use a new --output directory.')
    jobs = [(a, s) for s in seeds for a in args.algorithms if a in STOCHASTIC]
    jobs += [(a, None) for a in args.algorithms if a not in STOCHASTIC]
    print(f'n25 BPR-heavy | P=10 | {args.seconds:g}s/job | {len(jobs)} jobs | output={output}', flush=True)
    if args.check:
        # Verify both sidecar and ATF hashes and reprice BKS before a campaign.
        load_poryos(INSTANCE)
        problem = TDVRPProblem(INSTANCE, collection_root=ROOT / 'data/TDVRP', vehicle_limit=25)
        bks = load_poryos_bks(find_poryos_bks(INSTANCE))
        evaluation = problem.evaluate_candidate(bks['routes'])
        assert evaluation.feasible and abs(evaluation.base_cost - bks['cost']) < 1e-6
        print('CHECK PASSED. No experiment started.', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest, protocol)
    store = CampaignStore(output, protocol)
    store.log(f'SESSION START jobs={len(jobs)} seconds={args.seconds:g} seeds={args.seeds} completed_checkpoint={len(store.completed)}')
    rows = []
    failures = []
    for index, (algorithm, seed) in enumerate(jobs, 1):
        path = output / f'{algorithm}_seed{seed if seed is not None else "det"}.json'
        try:
            data = store.completed.get((algorithm, seed))
            if data is not None:
                # Recover missing/truncated result files from the checkpoint.
                atomic_json(path, data)
                store.log(f'[{index}/{len(jobs)}] RESUME SKIP {algorithm} seed={seed}')
            else:
                if path.exists():
                    try:
                        data = json.loads(path.read_text(encoding='utf-8'))
                    except (ValueError, UnicodeDecodeError):
                        data = None
                    if not valid_result(data, algorithm, seed, args.seconds):
                        store.log(f'RECOVERY invalid uncheckpointed result will be retried: {path.name}')
                        data = None
                if data is not None:
                    store.save_completed(data)
                    store.log(f'[{index}/{len(jobs)}] IMPORT completed result {algorithm} seed={seed}')
                else:
                    store.event('started', algorithm=algorithm, seed=seed)
                    store.log(f'[{index}/{len(jobs)}] START {algorithm} seed={seed}')
                    data = run_job(algorithm, seed, args.seconds)
                    if not valid_result(data, algorithm, seed, args.seconds):
                        raise ValueError('Run returned an invalid completed result')
                    # Commit the checkpoint before writing derivative result files.
                    store.save_completed(data)
                    atomic_json(path, data)
        except KeyboardInterrupt:
            store.event('interrupted', algorithm=algorithm, seed=seed)
            store.log(f'INTERRUPTED {algorithm} seed={seed}; rerun same command to resume completed jobs')
            raise SystemExit(130)
        except Exception as error:
            store.event('error', algorithm=algorithm, seed=seed, error_type=type(error).__name__, message=str(error))
            store.log(f'[{index}/{len(jobs)}] ERROR {algorithm} seed={seed}: {traceback.format_exc()}')
            failures.append((algorithm, seed))
            continue
        rows.append(data)
        store.log(f'[{index}/{len(jobs)}] OK {algorithm} seed={seed} Duration={data["duration"]:.6f} Gap={data["gap_to_bks_percent"]:.4f}% NFE={data["nfe"]}')
        summary = []
        for name in args.algorithms:
            group = [r for r in rows if r['algorithm'] == name]
            if group:
                values = [r['duration'] for r in group]
                summary.append({'algorithm': name, 'runs': len(group), 'deterministic': name not in STOCHASTIC,
                                'best': min(values), 'mean': statistics.fmean(values),
                                'sd': statistics.stdev(values) if len(values) > 1 else None,
                                'mean_gap_percent': statistics.fmean(r['gap_to_bks_percent'] for r in group)})
        atomic_json(output / 'summary.json', summary)
        with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    store.event('session_finished', completed=len(rows), failed=len(failures), requested=len(jobs))
    store.log(f'SESSION END completed={len(rows)}/{len(jobs)} failed={len(failures)}')
    if failures:
        raise SystemExit('Some jobs failed. See run.log; rerun same command to retry them.')


if __name__ == '__main__':
    main()
