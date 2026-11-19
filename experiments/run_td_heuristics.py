"""CLI for deterministic TD constructive/descent baselines (no NFE padding)."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

from algorithms.td_constructive import ALGORITHMS, TDHeuristicConfig, solve
from core.data.load_poryos import find_poryos_bks, load_poryos_bks
from core.tdvrp_common import TDVRPProblem

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'data/TDVRP/hong_kong/n=25/poryos-hong_kong-n25-hyb/bpr-heavy/poryos-hong_kong-n25-hyb-bpr-heavy.vrp.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', type=Path, default=DEFAULT)
    parser.add_argument('--collection-root', type=Path, default=ROOT / 'data/TDVRP')
    parser.add_argument('--algorithms', nargs='+', choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument('--time-limit', type=float, default=None, help='Seconds per method, including insertion initialization; cooperative limit')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    config = TDHeuristicConfig(time_limit_seconds=args.time_limit)
    instance_path = args.instance.resolve()
    output = args.output or ROOT / 'results/td_heuristics' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    payload = json.loads(instance_path.read_text(encoding='utf-8'))
    source_files = [ROOT / 'algorithms/td_constructive.py', ROOT / 'core/tdvrp_common.py',
                    Path(__file__).resolve(), instance_path]
    bks_path = find_poryos_bks(instance_path)
    if bks_path.is_file():
        source_files.append(bks_path)
    protocol = {'instance': str(instance_path), 'instance_name': payload['instance_name'],
                'python': sys.version, 'platform': platform.platform(),
                'mamut_routing_lib': version('mamut-routing-lib'),
                'time_limit_seconds': args.time_limit, 'execution': 'sequential, one deterministic run per method',
                'algorithms': args.algorithms, 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    bks_cost = None
    rows = []
    for algorithm in args.algorithms:
        print(f'Loading fresh evaluator for {algorithm}...', flush=True)
        # A fresh problem/cache for every method. Loading is outside solver timing.
        problem = TDVRPProblem(instance_path, collection_root=args.collection_root,
                               vehicle_limit=payload['num_customers'])
        if bks_path.is_file():
            bks = load_poryos_bks(bks_path)
            if bks['instance_name'] != payload['instance_name']:
                raise ValueError('BKS belongs to another instance')
            verified = problem.evaluate_candidate(bks['routes'])
            if not verified.feasible or abs(verified.base_cost - bks['cost']) > 1e-6:
                raise ValueError('BKS does not match the evaluator')
            bks_cost = bks['cost']
            # BKS validation must not warm the timed search route cache.
            problem.route_cost.cache_clear()
        print(f'Starting {algorithm}; output={output}', flush=True)
        result = solve(problem, algorithm, config)
        data = result.to_dict()
        data['bks_cost'] = bks_cost
        data['gap_to_bks_percent'] = 100 * (result.duration / bks_cost - 1) if bks_cost else None
        (output / f'{algorithm}.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
        row = {'algorithm': algorithm, 'duration': result.duration, 'gap_to_bks_percent': data['gap_to_bks_percent'],
               'runtime_seconds': result.runtime_seconds, 'num_routes': len(result.routes),
               'feasible': result.feasible, 'termination': result.termination, **result.statistics}
        rows.append(row)
        with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
    print(f'DONE: {output}', flush=True)


if __name__ == '__main__':
    main()
