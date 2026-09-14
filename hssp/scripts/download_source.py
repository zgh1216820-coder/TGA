"""按明确的Hugging Face提交下载单个来源；不会把当前版本冒充历史版本。"""
import argparse
import json
import re
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--revision', required=True, help='Full 40-character commit SHA')
    ap.add_argument('--raw-root', type=Path, required=True)
    args = ap.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        raise ValueError('Use a pinned commit SHA, not main')
    registry = json.loads((Path(__file__).resolve().parents[1] / 'configs/sources.json').read_text(encoding='utf-8'))
    source = next(r for r in registry if r['name'] == args.source)
    if not source['hf_repo_id']:
        raise ValueError('No verified Hugging Face ID recorded; use the source URL')
    dest = args.raw_root / args.source
    if dest.exists():
        raise FileExistsError(dest)
    snapshot_download(repo_id=source['hf_repo_id'], repo_type='dataset', revision=args.revision, local_dir=str(dest))
    (dest / 'HSSP_DOWNLOAD.json').write_text(json.dumps({'repo_id': source['hf_repo_id'], 'revision': args.revision, 'historical_revision_verified': False}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
