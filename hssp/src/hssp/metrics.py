"""保持论文历史评测口径，不将词元集合F1改成样本F1均值。"""


def token_set_f1(predictions, references, encode):
    if len(predictions) != len(references) or not predictions:
        raise ValueError('Expected nonempty aligned predictions and single references')
    matched = predicted = gold = 0
    for pred, ref in zip(predictions, references):
        pred_ids = set(encode(pred.strip()))
        ref_ids = set(encode(ref))
        matched += len(pred_ids & ref_ids)
        predicted += len(pred_ids)
        gold += len(ref_ids)
    precision = matched / predicted if predicted else 0.0
    recall = matched / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def macro_score(task_scores):
    if not task_scores:
        raise ValueError('No tasks')
    return sum(task_scores) / len(task_scores)
