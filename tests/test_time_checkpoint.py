import json
import sys

import pytest

from experiments.time_checkpoint import CampaignStore, atomic_json


def result(algorithm='GA', seed=0):
    return {'status': 'completed', 'algorithm': algorithm, 'seed': seed,
            'time_limit_seconds': 1., 'duration': 1., 'feasible': True,
            'routes': [[1]], 'gap_to_bks_percent': 0., 'nfe': 10}


def test_completed_job_survives_error_and_reopen(tmp_path):
    protocol = {'seconds': 1., 'seeds': [0]}
    store = CampaignStore(tmp_path, protocol)
    store.save_completed(result())
    store.event('error', algorithm='GA', seed=0, message='later session interrupted')
    store.log('START test')
    reopened = CampaignStore(tmp_path, protocol)
    assert reopened.completed[('GA', 0)] == result()
    assert 'START test' in (tmp_path / 'run.log').read_text()


def test_recovers_only_incomplete_final_append(tmp_path):
    protocol = {'seconds': 1.}
    store = CampaignStore(tmp_path, protocol)
    store.save_completed(result())
    with store.path.open('ab') as stream:
        stream.write(b'{"type":"started","algorithm":')
    restored = CampaignStore(tmp_path, protocol)
    assert restored.recovered_tail
    assert ('GA', 0) in restored.completed
    restored.event('started', algorithm='GA', seed=1)
    assert ('GA', 0) in CampaignStore(tmp_path, protocol).completed


def test_corrupt_middle_record_is_not_silently_discarded(tmp_path):
    store = CampaignStore(tmp_path, {'seconds': 1.})
    with store.path.open('ab') as stream:
        stream.write(b'broken\n{}\n')
    before = store.path.read_bytes()
    with pytest.raises(ValueError, match='Corrupt'):
        CampaignStore(tmp_path, {'seconds': 1.})
    assert store.path.read_bytes() == before


def test_protocol_mismatch_does_not_modify_checkpoint(tmp_path):
    store = CampaignStore(tmp_path, {'seconds': 1.})
    with store.path.open('ab') as stream:
        stream.write(b'{')
    before = store.path.read_bytes()
    with pytest.raises(ValueError, match='mismatch'):
        CampaignStore(tmp_path, {'seconds': 2.})
    assert store.path.read_bytes() == before


def test_atomic_json_is_complete_and_removes_temporary(tmp_path):
    path = tmp_path / 'result.json'
    atomic_json(path, result())
    assert json.loads(path.read_text()) == result()
    assert not path.with_suffix('.json.tmp').exists()


def test_campaign_retries_failure_skips_completed_and_restores_result(tmp_path, monkeypatch):
    from experiments import run_time_comparison as runner
    calls = []
    def job(algorithm, seed, seconds):
        calls.append((algorithm, seed))
        if algorithm == 'TD-Insertion' and calls.count((algorithm, seed)) == 1:
            raise RuntimeError('simulated failure')
        return result(algorithm, seed)
    monkeypatch.setattr(runner, 'run_job', job)
    monkeypatch.setattr(sys, 'argv', ['run', '--seconds', '1', '--seeds', '0-0',
                                    '--algorithms', 'GA', 'TD-Insertion', '--output', str(tmp_path)])
    with pytest.raises(SystemExit, match='Some jobs failed'):
        runner.main()
    runner.main()
    assert calls == [('GA', 0), ('TD-Insertion', None), ('TD-Insertion', None)]
    # Checkpoint is authoritative if the derivative JSON disappears.
    (tmp_path / 'GA_seed0.json').unlink()
    runner.main()
    assert len(calls) == 3
    assert json.loads((tmp_path / 'GA_seed0.json').read_text())['status'] == 'completed'
    assert len(json.loads((tmp_path / 'summary.json').read_text())) == 2
    log = (tmp_path / 'run.log').read_text()
    assert 'START' in log and 'ERROR' in log and 'RESUME SKIP' in log and 'SESSION END' in log


def test_keyboard_interrupt_records_event_and_keeps_previous_results(tmp_path, monkeypatch):
    from experiments import run_time_comparison as runner
    def job(algorithm, seed, seconds):
        if algorithm == 'TD-Insertion':
            raise KeyboardInterrupt
        return result(algorithm, seed)
    monkeypatch.setattr(runner, 'run_job', job)
    monkeypatch.setattr(sys, 'argv', ['run', '--seconds', '1', '--seeds', '0-0',
                                    '--algorithms', 'GA', 'TD-Insertion', '--output', str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 130
    assert (tmp_path / 'GA_seed0.json').is_file()
    assert 'INTERRUPTED' in (tmp_path / 'run.log').read_text()
    events = [json.loads(line) for line in (tmp_path / 'checkpoint.jsonl').read_text().splitlines()]
    assert events[-1]['type'] == 'interrupted'
