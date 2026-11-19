"""Durable run-level checkpoint and timestamped log for timed campaigns."""
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def valid_result(data, algorithm, seed, seconds):
    return (isinstance(data, dict) and data.get('status') == 'completed'
            and data.get('algorithm') == algorithm and data.get('seed') == seed
            and data.get('time_limit_seconds') == seconds and data.get('feasible') is True
            and isinstance(data.get('duration'), (int, float))
            and math.isfinite(data['duration']) and data['duration'] >= 0
            and isinstance(data.get('routes'), list) and bool(data['routes']))


class CampaignStore:
    def __init__(self, output, protocol):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / 'checkpoint.jsonl'
        self.log_path = self.output / 'run.log'
        self.signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        self.completed = {}
        self.recovered_tail = False
        if self.path.exists():
            raw = self.path.read_bytes()
            lines = raw.splitlines(keepends=True)
            consumed = 0
            records = []
            for index, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if index == len(lines) - 1 and not line.endswith(b'\n') and index > 0:
                        # Only an incomplete final append is discarded, never a
                        # malformed middle record or an unverified header.
                        with self.path.open('r+b') as stream:
                            stream.truncate(consumed)
                            stream.flush()
                            os.fsync(stream.fileno())
                        self.recovered_tail = True
                        break
                    raise ValueError(f'Corrupt checkpoint record {index + 1}; no results discarded')
                records.append(record)
                if index == 0 and (record.get('type') != 'header' or record.get('signature') != self.signature):
                    raise ValueError('Checkpoint protocol/code/data mismatch')
                consumed += len(line)
            if not records or records[0].get('type') != 'header' or records[0].get('signature') != self.signature:
                raise ValueError('Checkpoint protocol/code/data mismatch')
            for record in records[1:]:
                if record.get('type') == 'completed':
                    data = record['result']
                    if not valid_result(data, record['algorithm'], record['seed'], protocol['seconds']):
                        raise ValueError('Invalid completed result in checkpoint')
                    self.completed[(record['algorithm'], record['seed'])] = data
            if raw and not raw.endswith(b'\n') and not self.recovered_tail:
                with self.path.open('ab') as stream:
                    stream.write(b'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
        else:
            self.event('header', signature=self.signature, protocol=protocol, version=1)
        if self.recovered_tail:
            self.log('RECOVERY discarded an incomplete final checkpoint record; completed jobs retained')

    def log(self, message):
        line = f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] {message}'
        print(line, flush=True)
        with self.log_path.open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def event(self, kind, **fields):
        record = {'type': kind, 'timestamp': datetime.now().astimezone().isoformat(), **fields}
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def save_completed(self, data):
        key = data['algorithm'], data['seed']
        self.event('completed', algorithm=key[0], seed=key[1], result=data)
        self.completed[key] = data
