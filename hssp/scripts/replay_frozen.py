"""从已有标准化候选记录恢复冻结顺序；候选必须覆盖清单中的精确内容。"""
import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def digest(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', type=Path, required=True, help='Directory of normalized JSON arrays')
    ap.add_argument('--output-root', type=Path)
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    if not args.check_only and args.output_root is None:
        ap.error('--output-root is required without --check-only')
    if args.output_root and args.output_root.exists():
        raise FileExistsError(args.output_root)
    package = Path(__file__).resolve().parents[1]
    required = defaultdict(list)
    with gzip.open(package / 'manifests/membership.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            required[f"dataset/{r['dataset']}/{r['split']}/dataset-{r['subset_id']}.json"].append(r['record_sha256'])
    needed = {h for rows in required.values() for h in rows}
    found = {}
    datasets = sorted({rel.split('/')[1] for rel in required})
    paths = sorted({path for ds in datasets for sp in ('train', 'test') for path in (args.candidates / ds / sp).glob('dataset-*.json')})
    for path in paths:
        rows = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(rows, list):
            raise ValueError(f'Expected normalized record array: {path}')
        for row in rows:
            h = digest(row)
            if h in needed:
                found[h] = row
    missing = needed - found.keys()
    if missing:
        raise ValueError(f'Missing {len(missing)} exact candidate records; no output written')
    expected = {r['path']: r['sha256'] for r in json.loads((package / 'manifests/frozen_files.json').read_text(encoding='utf-8'))['files']}
    payloads = {}
    for rel, hashes in required.items():
        raw = json.dumps([found[h] for h in hashes], ensure_ascii=False, indent=4).encode('utf-8')
        if hashlib.sha256(raw).hexdigest() != expected[rel]:
            raise ValueError(f'Records match but serialized bytes differ: {rel}')
        payloads[rel] = raw
    if args.check_only:
        print(json.dumps({'status':'PASS', 'files':len(payloads), 'exact_bytes_reconstructable':True, 'raw_download_to_candidates_verified':False}))
        return
    args.output_root.mkdir(parents=True)
    for rel, raw in payloads.items():
        dest = args.output_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    print('Frozen record content and order restored. Images must be provided separately. Run verify_release.py for byte identity.')


if __name__ == '__main__':
    main()
