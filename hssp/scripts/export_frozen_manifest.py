"""只读导出已冻结数据的编号、顺序与哈希，不复制图像和问答文本。"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--scenario', type=Path, required=True)
    ap.add_argument('--expected', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scenario = json.loads(args.scenario.read_text(encoding='utf-8'))
    expected = {(r['dataset'], int(r['subset_id']), r['split']): r for r in csv.DictReader(args.expected.open(encoding='utf-8-sig'))}
    keys = sorted({(t['dataset'], int(t['subset_id']), sp) for c in scenario for t in c['datasets'] for sp in ('train', 'test')})
    report = {'files': [], 'mismatches': [], 'sample_rows': 0, 'missing_images': 0}
    # 清单保存样本位置及内容摘要；不会把原始问答和机器绝对路径写入发布物。
    with gzip.open(args.output / 'membership.jsonl.gz', 'wt', encoding='utf-8') as out:
        for ds, sub, split in keys:
            rel = f'dataset/{ds}/{split}/dataset-{sub}.json'
            path = args.root / rel
            raw = path.read_bytes()
            rows = json.loads(raw)
            sha = digest(raw)
            old = expected[(ds, sub, split)]
            matched = sha == old['sha256'] and len(rows) == int(old['row_count'])
            entry = {'dataset': ds, 'subset_id': sub, 'split': split, 'path': rel, 'sha256': sha, 'row_count': len(rows), 'matches_20260815': matched}
            report['files'].append(entry)
            if not matched:
                report['mismatches'].append(rel)
            for pos, row in enumerate(rows):
                image = str(row['image']).replace('\\', '/')
                if not (args.root / image).is_file():
                    report['missing_images'] += 1
                rec = {'dataset': ds, 'subset_id': sub, 'split': split, 'position': pos, 'id': str(row['id']), 'image': image,
                       'record_sha256': digest(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8'))}
                out.write(json.dumps(rec, ensure_ascii=False, separators=(',', ':')) + '\n')
                report['sample_rows'] += 1
    report['status'] = 'PASS' if not report['mismatches'] and not report['missing_images'] else 'MISMATCH'
    report['membership_sha256'] = digest((args.output / 'membership.jsonl.gz').read_bytes())
    (args.output / 'frozen_files.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, ensure_ascii=False))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
