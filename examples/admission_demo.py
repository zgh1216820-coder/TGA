"""在合成相关性分数上展示两个来源池的选择，不启动训练。"""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from tga.admission import _tg_select_source_ids, _tg_sparse_softmax_weights

def main():
    args = SimpleNamespace(**json.loads((Path(__file__).resolve().parents[1] / "configs/tga.json").read_text()))
    scores = torch.tensor([1.0, 0.8, 0.9, 0.7, 0.6])
    state = {"model_ids": {"1B": [0, 1, 4], "3B": [2, 3]}, "curr_round": 1}
    for family, pool in [("cross_size_mappable", [1, 2, 3, 4]), ("same_model", [1, 4])]:
        selected = _tg_select_source_ids(scores, pool, 0, args, state)
        weights = _tg_sparse_softmax_weights(scores, pool, 0, args, state, selected_sources=selected)
        print(json.dumps({"family": family, "selected": selected, "weights": weights.tolist()}))

if __name__ == "__main__":
    main()
