"""从现有训练实现提取的来源选择及权重函数，保留原逻辑。"""
import random
import torch

def _tg_is_random_matched_policy(training_args):
    policy = str(getattr(training_args, "fedmosaic_tg_source_policy", "similarity") or "similarity").strip().lower()
    return policy in {"random", "random_matched", "random_matched_coverage"}

def _tg_ranked_candidates(sim_row, candidates, training_args):
    min_sim = float(getattr(training_args, "fedmosaic_tg_min_sim", -1.0))
    eligible = [cid for cid in candidates if float(sim_row[cid]) >= min_sim]
    if not eligible:
        eligible = list(candidates)
    return sorted(eligible, key=lambda cid: float(sim_row[cid]), reverse=True)

def _tg_random_matched_sources(client_id, candidates, reference_sources, training_args, extra_state_dict_dict):
    if not reference_sources:
        return []
    homo_ids = _tg_homo_client_ids(client_id, extra_state_dict_dict)
    same_needed = sum(1 for cid in reference_sources if cid in homo_ids)
    cross_needed = len(reference_sources) - same_needed
    same_pool = [cid for cid in candidates if cid in homo_ids and cid != int(client_id)]
    cross_pool = [cid for cid in candidates if cid not in homo_ids and cid != int(client_id)]
    curr_round = int((extra_state_dict_dict or {}).get("curr_round", 0))
    seed = int(getattr(training_args, "seed", 1) or 1)
    rng = random.Random(seed + 1009 * curr_round + 9176 * int(client_id))
    selected = []
    if same_needed > 0 and same_pool:
        selected.extend(rng.sample(same_pool, k=min(same_needed, len(same_pool))))
    if cross_needed > 0 and cross_pool:
        selected.extend(rng.sample(cross_pool, k=min(cross_needed, len(cross_pool))))
    if len(selected) < len(reference_sources):
        rest = [cid for cid in candidates if cid != int(client_id) and cid not in selected]
        if rest:
            selected.extend(rng.sample(rest, k=min(len(reference_sources) - len(selected), len(rest))))
    return selected[: len(reference_sources)]

def _tg_select_source_ids(sim_row, allowed_ids, client_id, training_args, extra_state_dict_dict=None):
    candidates = [int(cid) for cid in allowed_ids if int(cid) != int(client_id)]
    if not candidates:
        return []

    topk = int(getattr(training_args, "fedmosaic_tg_topk", 2))
    ranked_all = _tg_ranked_candidates(sim_row, candidates, training_args)
    budget = topk if topk > 0 else len(ranked_all)

    hetero_quota = int(getattr(training_args, "fedmosaic_tg_hetero_quota", 0) or 0)
    reference_sources = _tg_select_with_hetero_quota(
        client_id,
        ranked_all,
        budget,
        hetero_quota,
        extra_state_dict_dict or {},
    )

    if _tg_is_random_matched_policy(training_args):
        return _tg_random_matched_sources(
            client_id,
            candidates,
            reference_sources,
            training_args,
            extra_state_dict_dict or {},
        )

    source_scope = str(getattr(training_args, "fedmosaic_tg_source_scope", "all") or "all").strip().lower()
    eligible = ranked_all
    if source_scope in {"homo", "same"}:
        homo_ids = _tg_homo_client_ids(client_id, extra_state_dict_dict or {})
        eligible = [cid for cid in eligible if cid in homo_ids]
    elif source_scope in {"hetero", "cross"}:
        homo_ids = _tg_homo_client_ids(client_id, extra_state_dict_dict or {})
        eligible = [cid for cid in eligible if cid not in homo_ids]

    if hetero_quota > 0:
        eligible = _tg_select_with_hetero_quota(
            client_id,
            eligible,
            budget,
            hetero_quota,
            extra_state_dict_dict or {},
        )
    elif topk > 0:
        eligible = eligible[:topk]

    return eligible[:budget] if budget > 0 else eligible

def _tg_resolve_weight_policy(training_args):
    policy = str(
        getattr(training_args, "fedmosaic_tg_weight_policy", "auto") or "auto"
    ).strip().lower()
    if policy == "auto":
        return "uniform" if _tg_is_random_matched_policy(training_args) else "similarity_softmax"
    if policy not in {"similarity_softmax", "uniform"}:
        raise ValueError(
            "fedmosaic_tg_weight_policy must be one of "
            "auto|similarity_softmax|uniform"
        )
    return policy

def _tg_sparse_softmax_weights(
    sim_row,
    allowed_ids,
    client_id,
    training_args,
    extra_state_dict_dict=None,
    selected_sources=None,
):
    num_clients = sim_row.numel()
    if selected_sources is None:
        selected_sources = _tg_select_source_ids(
            sim_row,
            allowed_ids,
            client_id,
            training_args,
            extra_state_dict_dict,
        )
    weights = torch.zeros(num_clients, dtype=torch.float32)
    if not selected_sources:
        return weights
    if _tg_resolve_weight_policy(training_args) == "uniform":
        for cid in selected_sources:
            weights[cid] = 1.0 / len(selected_sources)
        return weights

    masked = torch.full((num_clients,), -1e9, dtype=torch.float32)
    for cid in selected_sources:
        masked[cid] = float(sim_row[cid])
    temp = float(getattr(training_args, "softmax_temp", 0.5))
    temp = temp if temp > 0 else 1.0
    weights = (masked / temp).softmax(dim=0)
    if not torch.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = torch.zeros(num_clients, dtype=torch.float32)
        for cid in selected_sources:
            weights[cid] = 1.0 / len(selected_sources)
    return weights

def _tg_homo_client_ids(client_id, extra_state_dict_dict):
    model_ids = extra_state_dict_dict.get("model_ids", {}) if extra_state_dict_dict is not None else {}
    for _, homo_ids in model_ids.items():
        ids = {int(cid) for cid in homo_ids}
        if int(client_id) in ids:
            return ids
    return {int(client_id)}

def _tg_select_with_hetero_quota(client_id, eligible, budget, hetero_quota, extra_state_dict_dict):
    budget = min(int(budget), len(eligible))
    if budget <= 0:
        return []

    hetero_quota = max(0, min(int(hetero_quota), budget))
    if hetero_quota <= 0:
        return eligible[:budget]

    homo_ids = _tg_homo_client_ids(client_id, extra_state_dict_dict)
    selected = []
    selected_set = set()

    for cid in eligible:
        if cid in homo_ids:
            continue
        selected.append(cid)
        selected_set.add(cid)
        if len(selected) >= hetero_quota:
            break

    for cid in eligible:
        if len(selected) >= budget:
            break
        if cid in selected_set:
            continue
        selected.append(cid)
        selected_set.add(cid)

    return selected
