"""对保存的单任务预测重新评分，不加载模型权重、不生成新预测。"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hssp.metrics import token_set_f1
from hssp.choice_parser import parse_choice_list, infer_hssp_choice_options


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--mode', choices=['open', 'choice'], required=True)
    ap.add_argument('--tokenizer', help='Exact receiver tokenizer, required for open-ended evaluation')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    rows = json.loads(args.predictions.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty array of prediction records')
    for row in rows:
        if not {'sentence', 'gt_sentence'} <= row.keys():
            raise ValueError('Every row must contain sentence and gt_sentence; remove aggregate-only records')
    if args.mode == 'open':
        if not args.tokenizer:
            ap.error('--tokenizer is required')
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        report = token_set_f1([r['sentence'] for r in rows], [r['gt_sentence'] for r in rows], tok.encode)
        report['tokenizer'] = args.tokenizer
    else:
        correct = 0
        for row in rows:
            choices = [s.strip() for s in parse_choice_list(row['input'], hssp_format=True)]
            if choices and "'" in choices[0] and "'" not in row['gt_sentence']:
                choices = [s.strip("'") for s in choices]
            pred = infer_hssp_choice_options(row['sentence'].strip(), choices)
            gold = infer_hssp_choice_options(row['gt_sentence'], choices)
            correct += bool(pred and gold and set(pred) == set(gold))
        report = {'accuracy': correct / len(rows), 'correct': correct}
    report['samples'] = len(rows)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
