"""Run TD-ERAPO on Hong Kong HYB Wave-heavy n25, seeds 0..29."""
from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

from core.data.load_poryos import find_poryos_bks, load_poryos_bks
from experiments.runner import ExperimentConfig, ExperimentRunner

ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / 'data/TDVRP/hong_kong/n=25/poryos-hong_kong-n25-hyb/wave-heavy/poryos-hong_kong-n25-hyb-wave-heavy.vrp.json'
OUTPUT = ROOT / 'results/wave_heavy_n25_nfe290_seeds00_29'


def fingerprint():
    payload = json.loads(INSTANCE.read_text(encoding='utf-8'))
    inputs = [INSTANCE, find_poryos_bks(INSTANCE)]
    for kind in ('graph', 'traffic'):
        ref = payload['td'][kind]
        path = ROOT / 'data/TDVRP' / ref['path']
        if not path.is_file():
            raise FileNotFoundError(f'Missing official {kind} sidecar: {path}')
        data = path.read_bytes()
        digest = hashlib.sha256(gzip.decompress(data) if path.suffix == '.gz' else data).hexdigest()
        if digest != ref['sha256']:
            raise ValueError(f'SHA256 mismatch: {path}')
        inputs.append(path)
    for directory in ('algorithms', 'core', 'experiments'):
        inputs.extend(sorted((ROOT / directory).rglob('*.py')))
    bks = load_poryos_bks(find_poryos_bks(INSTANCE))
    if bks['instance_name'] != payload['instance_name']:
        raise ValueError('BKS belongs to a different instance')
    return {
        'instance_name': payload['instance_name'], 'display_algorithm': 'TD-ERAPO',
        'implementation_algorithm': 'RAPO-2709', 'population_size': 10,
        'nfe_budget': 290, 'seeds': list(range(30)), 'exact_split': True,
        'earliest_departure_seconds': 0, 'allow_proxy': False,
        'verify_checksums': True, 'bks_cost': bks['cost'],
        'sha256_files': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in inputs},
    }


def verify_report(report, output):
    from core.tdvrp_common import TDVRPProblem

    problem = TDVRPProblem(INSTANCE, collection_root=ROOT / 'data/TDVRP')
    bks = load_poryos_bks(find_poryos_bks(INSTANCE))
    bks_eval = problem.evaluate_candidate(bks['routes'])
    if not bks_eval.feasible or abs(bks_eval.base_cost - bks['cost']) > 1e-6:
        raise ValueError('BKS could not be reproduced by the reference evaluator')
    checks = []
    for row in report.runs:
        if row['status'] != 'completed':
            continue
        routes = [r[1:-1] if r[0] == r[-1] == 0 else r for r in row['solution_routes']]
        evaluation = problem.evaluate_candidate(routes)
        valid = (row['feasible'] and evaluation.feasible and row['nfe'] == 290
                 and abs(evaluation.base_cost - row['fitness']) <= 1e-6)
        checks.append({'seed': row['seed'], 'valid': bool(valid),
                       'duration_recomputed': evaluation.base_cost,
                       'absolute_difference': abs(evaluation.base_cost - row['fitness'])})
    valid_seeds = {r['seed'] for r in checks if r['valid']}
    complete = len(checks) == 30 and valid_seeds == set(range(30))
    validation = {'complete_30_seeds': complete, 'bks_recomputed': bks_eval.base_cost,
                  'bks_stored': bks['cost'], 'runs': checks}
    (output / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    if not complete:
        raise RuntimeError('Not all 30 seeds passed. Reports saved; rerun the same command to retry failed seeds.')
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check files/configuration without running optimization')
    args = parser.parse_args()
    current = fingerprint()
    provenance_path = OUTPUT / 'provenance.json'
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding='utf-8'))
        if previous['experiment'] != current:
            raise ValueError('Source code or data changed since this experiment started. Keep the same version to resume.')
    elif (OUTPUT / 'checkpoint.jsonl').exists():
        raise ValueError('Checkpoint exists without provenance; do not mix unverified results.')
    print('TD-ERAPO | Hong Kong HYB n25 | Wave-heavy', flush=True)
    print('P=10 | NFE=290 | seeds=0..29 | Exact Split | departure >= 0 | sequential', flush=True)
    print(f'BKS={current["bks_cost"]:.9f} | output={OUTPUT}', flush=True)
    if args.check:
        print('CHECK PASSED: files, sidecar SHA256, BKS identity and resume fingerprint. No optimization started.', flush=True)
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not provenance_path.exists():
        provenance_path.write_text(json.dumps({'experiment': current}, indent=2), encoding='utf-8')
    session = {'started': datetime.now().astimezone().isoformat(), 'python': sys.version,
               'platform': platform.platform(), 'execution': 'sequential'}
    with (OUTPUT / 'sessions.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(session) + '\n')

    def progress(event):
        if event['status'] == 'completed':
            gap = 100 * (event['fitness'] / current['bks_cost'] - 1)
            detail = f'Duration={event["fitness"]:.6f} | Gap={gap:.4f}% | NFE={event["nfe"]} | {event["runtime_seconds"]:.2f}s'
        else:
            detail = f'ERROR: {event["error_message"]}'
        line = f'[{datetime.now():%H:%M:%S}] [{event["completed"]}/30] seed={event["seed"]} {event["status"]} | {detail}'
        print(line, flush=True)
        with (OUTPUT / 'run.log').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')

    config = ExperimentConfig(instance_path=INSTANCE, seeds=tuple(range(30)),
                              population_size=10, nfe_budget=290, algorithms=('RAPO-2709',),
                              checkpoint_path=OUTPUT / 'checkpoint.jsonl', resume=True)
    try:
        report = ExperimentRunner(config, progress_callback=progress).run()
    except KeyboardInterrupt:
        print('\nStopped. Completed seeds are checkpointed; rerun the same command to continue.', flush=True)
        return 130
    report.save(OUTPUT / 'report.json')
    print('Rechecking all saved routes and BKS with the reference evaluator...', flush=True)
    verify_report(report, OUTPUT)
    gaps = [r['gap_to_bks_percent'] for r in report.runs]
    summary = {**report.summary[0], 'gap_standard_deviation_percentage_points': statistics.stdev(gaps),
               'route_count_mean': statistics.fmean(len(r['solution_routes']) for r in report.runs),
               'verified_30_seeds': True}
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('DONE: 30/30 seeds verified. Summary:', flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
