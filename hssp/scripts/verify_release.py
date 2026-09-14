"""核验冻结划分；默认只检查发布清单，提供数据根目录时核验真实文件。"""
import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    freeze = json.loads((root / 'manifests/frozen_files.json').read_text(encoding='utf-8'))
    archive = root / 'manifests/membership.jsonl.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == freeze['membership_sha256'], 'Membership SHA mismatch'
    expected = {(r['dataset'], int(r['subset_id']), r['split']): r for r in csv.DictReader((root / 'manifests/expected_files_20260815.csv').open(encoding='utf-8'))}
    positions = Counter()
    ids = defaultdict(set)
    images = defaultdict(set)
    train_seen = defaultdict(set)
    unique_records = defaultdict(set)
    with gzip.open(archive, 'rt', encoding='utf-8') as handle:
        for line in handle:
            r = json.loads(line)
            key = (r['dataset'], r['subset_id'], r['split'])
            assert r['position'] == positions[key], 'Noncontiguous row order'
            positions[key] += 1
            identity = (r['dataset'], r['id'])
            assert identity not in ids[key], 'Duplicate ID within subset'
            ids[key].add(identity)
            images[(r['dataset'], r['split'])].add(r['image'])
            unique_records[r['split']].add((r['dataset'], r['record_sha256']))
            if r['split'] == 'train':
                assert identity not in train_seen[r['dataset']], 'Training subsets overlap'
                train_seen[r['dataset']].add(identity)
    for ds in train_seen:
        assert not images[(ds, 'train')] & images[(ds, 'test')], 'Train/test image paths overlap'
    for entry in freeze['files']:
        key = (entry['dataset'], entry['subset_id'], entry['split'])
        assert positions[key] == entry['row_count'] == int(expected[key]['row_count'])
        assert entry['sha256'] == expected[key]['sha256']
        if args.data_root:
            actual = args.data_root / entry['path']
            assert hashlib.sha256(actual.read_bytes()).hexdigest() == entry['sha256'], str(actual)
    tasks = list(csv.DictReader((root / 'manifests/tasks.csv').open(encoding='utf-8')))
    report = {'status': 'PASS', 'verification': 'actual_file_bytes' if args.data_root else 'manifest_internal_consistency',
              'task_units': len(tasks), 'clients': len({r['client_id'] for r in tasks}),
              'sources': len({r['dataset'] for r in tasks}), 'files': len(freeze['files']),
              'scheduled_rows': {sp: sum(positions[(r['dataset'], int(r['subset_id']), sp)] for r in tasks) for sp in ['train', 'test']},
              'unique_record_hashes': {sp: len(v) for sp, v in unique_records.items()},
              'raw_to_frozen_rebuild_verified': False, 'all_baseline_fairness_verified': False}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
