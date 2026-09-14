import os
import json
import random
import io
import math
import zipfile
import tarfile
import shutil
import tempfile
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==========================================
#  全局配置区
# ==========================================
RAW_BASE_DIR = None
TARGET_BASE_DIR = None

#  修正：这仅仅是“最大容量上限”，绝不强求，更不复制
MAX_TRAIN_SAMPLES = 4000
MAX_TEST_SAMPLES = 1000
DIRICHLET_ALPHA = 0.5  # Non-IID 倾斜度
SHARE_TEST_ACROSS_SUBSETS = True
MIN_TRAIN_SAMPLES_PER_SPLIT = 500

# 客户端分配矩阵映射
DATASET_CONFIG = {
    "vqa-rad": {"splits": 1, "type": "parquet"},
    "malaria-microscopy-vqa": {"splits": 4, "type": "parquet"},
    "path-vqa": {"splits": 5, "type": "parquet"},
    "DocumentVQA": {"splits": 4, "type": "parquet"},
    "ChartQA": {"splits": 4, "type": "parquet"},
    "InfoVQA": {"splits": 4, "type": "parquet"},
    "ChemVQA-2K": {"splits": 4, "type": "chemvqa"},
    "MMAD": {"splits": 4, "type": "mmad"},
    "DrivingVQA": {"splits": 1, "type": "drivingvqa"},
    "InspecSafe-V1": {"splits": 2, "type": "inspecsafe"},
    "Traffic-VQA": {"splits": 7, "type": "trafficvqa"}
}

# ==========================================
#  核心：严格无重叠的 Dirichlet 轮盘赌分配
# ==========================================
def dirichlet_split_and_save(data_list, out_dir, num_splits, max_samples, split_name, alpha=DIRICHLET_ALPHA, top_k=20, shared_across_splits=False, min_samples_per_split=1):
    if not data_list:
        print(f" {split_name} 数据为空，跳过切分。")
        return

    if shared_across_splits:
        shared_data = data_list[:max_samples]
        for i in range(num_splits):
            out_file = os.path.join(out_dir, f"dataset-{i}.json")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(shared_data, f, ensure_ascii=False, indent=4)
            print(f"  OK {split_name} [Subset {i}] - shared eval set: {len(shared_data)} samples")
        return

    if len(data_list) < num_splits * min_samples_per_split:
        raise ValueError(
            f"{split_name} has only {len(data_list)} samples for {num_splits} splits. "
            f"Need at least {num_splits * min_samples_per_split} samples."
        )

    print(f"\n 开始基于 Dirichlet(α={alpha}) 切分 [{split_name}]...")
    print(f"   - 规则：绝对无重复，每个 Client 上限 {max_samples} 条。")

    # 1. 提取伪标签
    answers = [item["conversations"][1]["value"].lower().strip() for item in data_list]
    counter = Counter(answers)
    top_answers = {ans for ans, _ in counter.most_common(top_k)}
    ans_to_class = {ans: i for i, ans in enumerate(sorted(top_answers))}

    class_indices = defaultdict(list)
    for idx, item in enumerate(data_list):
        ans = item["conversations"][1]["value"].lower().strip()
        c = ans_to_class.get(ans, top_k)
        class_indices[c].append(idx)

    # 2. 为每个 Client 设置剩余容量 (只做上限限制)
    client_capacities = [max_samples] * num_splits
    client_buckets = [[] for _ in range(num_splits)]

    # 3. 生成 Dirichlet 概率矩阵
    num_classes = top_k + 1
    class_dist = np.random.dirichlet([alpha] * num_splits, num_classes)

    # 4. 严格无放回的轮盘赌分配
    for c, indices in class_indices.items():
        random.shuffle(indices) # 打乱同类样本，避免连续性
        for idx in indices:
            probs = class_dist[c].copy()

            # 将已经达到 max_samples 上限的 Client 概率清零
            for i in range(num_splits):
                if client_capacities[i] <= 0:
                    probs[i] = 0.0

            sum_probs = probs.sum()
            if sum_probs > 0:
                # 重新归一化概率
                probs = probs / sum_probs
                chosen_client = np.random.choice(num_splits, p=probs)

                # 真实分配数据，并扣除容量
                client_buckets[chosen_client].append(idx)
                client_capacities[chosen_client] -= 1
            else:
                # 所有 Client 都达到了最大容量上限，剩余数据直接丢弃，绝不塞入
                break

    # 5. Repair extreme Dirichlet draws so every referenced train split is usable.
    for i in range(num_splits):
        while len(client_buckets[i]) < min_samples_per_split:
            donor = max(range(num_splits), key=lambda j: len(client_buckets[j]))
            if donor == i or len(client_buckets[donor]) <= min_samples_per_split:
                break
            client_buckets[i].append(client_buckets[donor].pop())

    # 6. 打乱每个 Client 内部顺序并保存
    for i, bucket in enumerate(client_buckets):
        chunk = [data_list[idx] for idx in bucket]
        random.shuffle(chunk)
        if not chunk: continue
        out_file = os.path.join(out_dir, f"dataset-{i}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=4)
        print(f"   {split_name} [Client {i}] - 生成: dataset-{i}.json (真实获取: {len(chunk)} 条)")

# ==========================================
#  通用工具函数
# ==========================================
def get_target_dirs(dataset_name):
    base = os.path.join(TARGET_BASE_DIR, dataset_name)
    dirs = {"base": base, "images": os.path.join(base, "images"), "train": os.path.join(base, "train"), "test": os.path.join(base, "test")}
    for d in dirs.values():
        if d != base: os.makedirs(d, exist_ok=True)
    return dirs

def safe_extract_string(val):
    if pd.isna(val) is True if isinstance(pd.isna(val), bool) else pd.isna(val).all(): return None
    if val is None: return None
    if isinstance(val, (list, np.ndarray)):
        return str(val[0]) if len(val) > 0 else None
    return str(val)

def safe_extract_answer(row):
    for key in ['answer', 'answers', 'label']:
        val = row.get(key)
        is_null = False
        try:
            if val is None or pd.isna(val) is True: is_null = True
        except ValueError: pass
        if not is_null:
            if isinstance(val, (list, np.ndarray)) and len(val) > 0: return str(val[0])
            elif isinstance(val, (str, int, float)): return str(val)
    return None


def normalize_choice_pairs(options):
    if isinstance(options, dict):
        return [(str(k).strip(), str(v).strip()) for k, v in options.items() if str(k).strip()]
    if isinstance(options, np.ndarray):
        options = options.tolist()
    if isinstance(options, list):
        return [(chr(ord('A') + i), str(v).strip()) for i, v in enumerate(options) if str(v).strip()]
    text = str(options).strip()
    if not text:
        return []
    return [(chr(ord('A') + i), part.strip()) for i, part in enumerate(text.split(',')) if part.strip()]


def format_choice_prompt(question, options, suffix="Please answer with the correct option letter."):
    pairs = normalize_choice_pairs(options)
    if not pairs:
        return question, []
    choice_list = " | ".join([f"{key}. {value}" for key, value in pairs])
    return f"{question}\nChoice list:[{choice_list}]\n{suffix}", pairs


def normalize_choice_answer(answer, pairs):
    text = str(answer).strip()
    if not pairs:
        return text
    for key, value in pairs:
        if text.lower() == key.lower() or text.lower() == value.lower():
            return key
    for key, value in pairs:
        if text.lower() in value.lower() or value.lower() in text.lower():
            return key
    return text

# ==========================================
#  解析器
# ==========================================
def process_parquet(ds_name, splits):
    print(f"\n [Parquet 解析器] 开始处理: {ds_name}")
    raw_dir = os.path.join(RAW_BASE_DIR, ds_name, "data")
    if not os.path.exists(raw_dir): return
    dirs = get_target_dirs(ds_name)

    standard_data = {"train": [], "test": []}

    for f in sorted(os.listdir(raw_dir)):
        if not f.endswith(".parquet"): continue
        split_type = "train" if "train" in f.lower() else ("test" if "test" in f.lower() or "val" in f.lower() else "other")
        if split_type == "other": continue

        df = pd.read_parquet(os.path.join(raw_dir, f))
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"转换 {split_type}"):
            sample_id = f"{ds_name}_{split_type}_{len(standard_data[split_type])}"

            img_data = row.get('image')
            img_bytes = img_data['bytes'] if isinstance(img_data, dict) and 'bytes' in img_data else (img_data if isinstance(img_data, bytes) else None)
            if not img_bytes: continue

            q_text = safe_extract_string(row.get('question') if 'question' in row else row.get('query'))
            a_text = safe_extract_answer(row)

            if ds_name == "malaria-microscopy-vqa":
                choices = row.get('choices', [])
                if isinstance(choices, (np.ndarray, list)) and len(choices) > 0 and q_text:
                    q_text, choice_pairs = format_choice_prompt(q_text, choices)
                    a_text = normalize_choice_answer(row.get('correct_answer', ''), choice_pairs)

            if not q_text or not a_text: continue

            img_filename = f"{sample_id}.jpg"
            try:
                Image.open(io.BytesIO(img_bytes)).convert('RGB').save(os.path.join(dirs["images"], img_filename), format="JPEG")
            except: continue

            standard_data[split_type].append({
                "id": sample_id, "image": f"dataset/{ds_name}/images/{img_filename}",
                "conversations": [{"from": "human", "value": f"<image>\n{q_text.strip()}"}, {"from": "gpt", "value": a_text.strip()}]
            })

    dirichlet_split_and_save(standard_data["train"], dirs["train"], splits, MAX_TRAIN_SAMPLES, "Train", min_samples_per_split=MIN_TRAIN_SAMPLES_PER_SPLIT)
    dirichlet_split_and_save(standard_data["test"], dirs["test"], splits, MAX_TEST_SAMPLES, "Test", shared_across_splits=SHARE_TEST_ACROSS_SUBSETS)


def process_medxpertqa(ds_name, splits):
    print(f"\n [JSONL+ZIP 解析器] 开始处理: {ds_name}")
    raw_dir, dirs = os.path.join(RAW_BASE_DIR, ds_name), get_target_dirs(ds_name)
    zip_path = os.path.join(raw_dir, "images.zip")
    if not os.path.exists(zip_path): return

    with zipfile.ZipFile(zip_path, 'r') as z:
        name_map = {os.path.basename(n): n for n in z.namelist() if not n.endswith('/')}

        def process_split(jsonl_path, split_type):
            if not os.path.exists(jsonl_path): return []
            data_list = []
            with open(jsonl_path, 'r', encoding='utf-8') as f: lines = f.readlines()
            for idx, line in tqdm(enumerate(lines), total=len(lines), desc=f"解析 {split_type}"):
                if not line.strip(): continue
                row = json.loads(line)
                img_files = row.get("images", [])
                if not img_files or img_files[0] not in name_map: continue

                try:
                    img_bytes = z.read(name_map[img_files[0]])
                    img_filename = f"{ds_name}_{split_type}_{idx}.jpg"
                    Image.open(io.BytesIO(img_bytes)).convert('RGB').save(os.path.join(dirs["images"], img_filename), format="JPEG")
                except: continue

                q_text = str(row.get("question", "")).strip()
                opts = row.get("options", {})
                choice_pairs = []
                if isinstance(opts, dict) and opts:
                    q_text, choice_pairs = format_choice_prompt(q_text, opts)
                answer_text = normalize_choice_answer(row.get("label", ""), choice_pairs)

                data_list.append({
                    "id": f"{ds_name}_{split_type}_{idx}", "image": f"dataset/{ds_name}/images/{img_filename}",
                    "conversations": [{"from": "human", "value": f"<image>\n{q_text}"}, {"from": "gpt", "value": answer_text} ]
                })
            return data_list

        train_data = process_split(os.path.join(raw_dir, "MM", "dev.jsonl"), "train")
        test_data = process_split(os.path.join(raw_dir, "MM", "test.jsonl"), "test")

    dirichlet_split_and_save(train_data, dirs["train"], splits, MAX_TRAIN_SAMPLES, "Train", min_samples_per_split=MIN_TRAIN_SAMPLES_PER_SPLIT)
    dirichlet_split_and_save(test_data, dirs["test"], splits, MAX_TEST_SAMPLES, "Test", shared_across_splits=SHARE_TEST_ACROSS_SUBSETS)


def process_chemvqa(ds_name, splits):
    print(f"\n [CSV+ZIP 解析器] 开始处理: {ds_name}")
    raw_dir, dirs = os.path.join(RAW_BASE_DIR, ds_name), get_target_dirs(ds_name)
    csv_path, zip_path = os.path.join(raw_dir, "ChemVQA_2K_full.csv"), os.path.join(raw_dir, "ChemVQA_2K_images.zip")
    if not os.path.exists(csv_path) or not os.path.exists(zip_path): return

    df = pd.read_csv(csv_path)
    standard_data = []

    with zipfile.ZipFile(zip_path, 'r') as z:
        name_map = {os.path.basename(n): n for n in z.namelist() if not n.endswith('/')}
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="解析 ChemVQA"):
            img_name = str(row['image_name'])
            target_zip_path = name_map.get(img_name) or name_map.get(f"{img_name}.png") or name_map.get(f"{img_name}.jpg")
            if not target_zip_path: continue

            try:
                img_bytes = z.read(target_zip_path)
                img_filename = f"{ds_name}_{idx}.jpg"
                Image.open(io.BytesIO(img_bytes)).convert('RGB').save(os.path.join(dirs["images"], img_filename), format="JPEG")
            except: continue

            standard_data.append({
                "id": f"{ds_name}_{idx}", "image": f"dataset/{ds_name}/images/{img_filename}",
                "conversations": [{"from": "human", "value": f"<image>\n{str(row['question']).strip()}"}, {"from": "gpt", "value": str(row['answer']).strip()}]
            })

    random.shuffle(standard_data)
    split_idx = int(len(standard_data) * 0.8)
    dirichlet_split_and_save(standard_data[:split_idx], dirs["train"], splits, MAX_TRAIN_SAMPLES, "Train", min_samples_per_split=MIN_TRAIN_SAMPLES_PER_SPLIT)
    dirichlet_split_and_save(standard_data[split_idx:], dirs["test"], splits, MAX_TEST_SAMPLES, "Test", shared_across_splits=SHARE_TEST_ACROSS_SUBSETS)


def process_mmad(ds_name, splits):
    print(f"\n [多重ZIP挂载解析器] 开始处理: {ds_name}")
    raw_dir, dirs = os.path.join(RAW_BASE_DIR, ds_name), get_target_dirs(ds_name)
    csv_path = os.path.join(raw_dir, "metadata.csv")
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)

    zip_files = [os.path.join(raw_dir, f) for f in sorted(os.listdir(raw_dir)) if f.endswith(".zip")]
    zip_objs = {zf: zipfile.ZipFile(zf, 'r') for zf in zip_files}
    global_name_map = {}
    for z_obj in zip_objs.values():
        for name in z_obj.namelist():
            if not name.endswith('/'):
                global_name_map[os.path.basename(name)] = (z_obj, name)
                global_name_map[name.replace('\\', '/')] = (z_obj, name)

    standard_data = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="解析 MMAD"):
        query_img = str(row.get('query_image', ''))
        target_info = global_name_map.get(query_img) or global_name_map.get(os.path.basename(query_img))
        if not target_info: continue
        z_obj, internal_path = target_info

        try:
            img_bytes = z_obj.read(internal_path)
            img_filename = f"{ds_name}_{idx}.jpg"
            Image.open(io.BytesIO(img_bytes)).convert('RGB').save(os.path.join(dirs["images"], img_filename), format="JPEG")
        except: continue

        q_text = str(row.get('question', '')).strip()
        opts = row.get('options')
        if pd.notna(opts) and str(opts).strip():
            q_text, choice_pairs = format_choice_prompt(q_text, opts)
            answer_text = normalize_choice_answer(row.get('answer', ''), choice_pairs)
        else:
            answer_text = str(row.get('answer', '')).strip()

        standard_data.append({
            "id": f"{ds_name}_{idx}", "image": f"dataset/{ds_name}/images/{img_filename}",
            "conversations": [{"from": "human", "value": f"<image>\n{q_text}"}, {"from": "gpt", "value": answer_text}]
        })

    for z in zip_objs.values(): z.close()
    random.shuffle(standard_data)
    split_idx = int(len(standard_data) * 0.8)
    dirichlet_split_and_save(standard_data[:split_idx], dirs["train"], splits, MAX_TRAIN_SAMPLES, "Train", min_samples_per_split=MIN_TRAIN_SAMPLES_PER_SPLIT)
    dirichlet_split_and_save(standard_data[split_idx:], dirs["test"], splits, MAX_TEST_SAMPLES, "Test", shared_across_splits=SHARE_TEST_ACROSS_SUBSETS)


def process_drivingvqa(ds_name, splits):
    print(f"\n [JSON+ZIP 解析器] 开始处理: {ds_name}")
    raw_dir, dirs = os.path.join(RAW_BASE_DIR, ds_name), get_target_dirs(ds_name)
    valid_jsons = [f for f in sorted(os.listdir(raw_dir)) if f in ['train.json', 'test.json', 'val.json', 'drivingvqa.json']]
    zip_path = os.path.join(raw_dir, "images.zip")
    if not valid_jsons or not os.path.exists(zip_path): return

    standard_data = {"train": [], "test": []}
    with zipfile.ZipFile(zip_path, 'r') as z:
        name_map = {os.path.basename(n): n for n in z.namelist() if not n.endswith('/')}
        for file_name in valid_jsons:
            split_type = "train" if "train" in file_name.lower() else "test"
            with open(os.path.join(raw_dir, file_name), 'r', encoding='utf-8') as f: raw_data = json.load(f)
            items = raw_data.items() if isinstance(raw_data, dict) else enumerate(raw_data)

            for sample_id, item in tqdm(items, desc=f"解析 {file_name}"):
                if not isinstance(item, dict): continue
                img_basename = os.path.basename(item.get('img_filename', ''))
                if img_basename not in name_map: continue

                try:
                    img_bytes = z.read(name_map[img_basename])
                    img_filename = f"{ds_name}_{sample_id}.jpg"
                    Image.open(io.BytesIO(img_bytes)).convert('RGB').save(os.path.join(dirs["images"], img_filename), format="JPEG")
                except: continue

                q_raw = item.get('questions', '')
                q_text = " ".join([str(q) for q in q_raw]) if isinstance(q_raw, list) else str(q_raw)
                opts = item.get('possible_answers', {})
                if opts:
                    q_text, choice_pairs = format_choice_prompt(q_text, opts, "Please answer with the correct option letter.")
                else:
                    choice_pairs = []

                ans_raw = item.get('true_answers', [])
                ans_text = ", ".join([str(a) for a in ans_raw]) if isinstance(ans_raw, list) else str(ans_raw)
                ans_text = normalize_choice_answer(ans_text, choice_pairs)

                standard_data[split_type].append({
                    "id": f"{ds_name}_{sample_id}", "image": f"dataset/{ds_name}/images/{img_filename}",
                    "conversations": [{"from": "human", "value": f"<image>\n{q_text.strip()}"}, {"from": "gpt", "value": ans_text.strip()}]
                })

    dirichlet_split_and_save(standard_data["train"], dirs["train"], splits, MAX_TRAIN_SAMPLES, "Train", min_samples_per_split=MIN_TRAIN_SAMPLES_PER_SPLIT)
    dirichlet_split_and_save(standard_data["test"], dirs["test"], splits, MAX_TEST_SAMPLES, "Test", shared_across_splits=SHARE_TEST_ACROSS_SUBSETS)


def process_inspecsafe(ds_name, splits):
    print(f"\n [TAR 极速分离器] 开始处理: {ds_name}")
    raw_dir, dirs = os.path.join(RAW_BASE_DIR, ds_name), get_target_dirs(ds_name)
    train_tar, test_tar = os.path.join(raw_dir, "train.tar"), os.path.join(raw_dir, "test.tar")
    if not os.path.exists(train_tar) or not os.path.exists(test_tar): return

    def parse_tar_fast(tar_path, split_name):
        data_list = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(tar_path, 'r') as tar:
                for member in tar.getmembers():
                    dest = os.path.realpath(os.path.join(tmpdir, member.name))
                    if os.path.commonpath([os.path.realpath(tmpdir), dest]) != os.path.realpath(tmpdir) or member.issym() or member.islnk():
                        raise ValueError('Unsafe archive member')
                tar.extractall(path=tmpdir)
            file_map = {f: os.path.join(root, f) for root, _, files in os.walk(tmpdir) for f in files}
            json_names = sorted(f for f in file_map.keys() if f.endswith('.json'))

            for j_name in tqdm(json_names, desc=f"解析 {split_name}"):
                try:
                    with open(file_map[j_name], 'r', encoding='utf-8') as f: item = json.load(f)
                except: continue

                base_name = os.path.splitext(j_name)[0]
                src_img_path = file_map.get(f"{base_name}.jpg") or file_map.get(f"{base_name}.png")
                if not src_img_path and item.get("imagePath"): src_img_path = file_map.get(os.path.basename(item["imagePath"]))
                if not src_img_path: continue

                img_filename = f"{ds_name}_{split_name}_{base_name}.jpg"
                try: Image.open(src_img_path).convert('RGB').save(os.path.join(dirs["images"], img_filename), format="JPEG")
                except: continue

                labels = {shape.get("label", "").strip() for shape in item.get("shapes", []) if shape.get("label", "").strip()}
                ans_text = f"Based on the image, the visible components or observations are: {', '.join(sorted(labels))}." if labels else "I cannot identify any specific industrial components or anomalies in this image."

                data_list.append({
                    "id": f"{ds_name}_{split_name}_{base_name}", "image": f"dataset/{ds_name}/images/{img_filename}",
                    "conversations": [{"from": "human", "value": f"<image>\nWhat industrial components or anomalies are visible in this image? Please list them."}, {"from": "gpt", "value": ans_text}]
                })
        return data_list

    dirichlet_split_and_save(parse_tar_fast(train_tar, "train"), dirs["train"], splits, MAX_TRAIN_SAMPLES, "Train", min_samples_per_split=MIN_TRAIN_SAMPLES_PER_SPLIT)
    dirichlet_split_and_save(parse_tar_fast(test_tar, "test"), dirs["test"], splits, MAX_TEST_SAMPLES, "Test", shared_across_splits=SHARE_TEST_ACROSS_SUBSETS)


def process_trafficvqa(ds_name, splits):
    print(f"\n [全目录雷达扫描器] 开始处理: {ds_name}")
    raw_dir, dirs = os.path.join(RAW_BASE_DIR, ds_name), get_target_dirs(ds_name)

    # 建立全目录图片雷达映射 (以备不时之需)
    img_map = {}
    for root, walk_dirs, files in os.walk(raw_dir):
        walk_dirs.sort()
        files.sort()
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_map[f] = os.path.join(root, f)
                img_map[os.path.splitext(f)[0]] = os.path.join(root, f)

    valid_jsons = [f for f in sorted(os.listdir(raw_dir)) if f.endswith('.json') and f in ['train_dataset.json', 'test_dataset.json', 'train.json', 'test.json']]
    standard_data = {"train": [], "test": []}

    for json_name in valid_jsons:
        split_type = "train" if "train" in json_name.lower() else "test"
        with open(os.path.join(raw_dir, json_name), 'r', encoding='utf-8') as f:
            try: data = json.load(f)
            except: continue

        # 兼容列表或字典存储的 JSON
        items = data if isinstance(data, list) else (list(data.values()) if isinstance(data, dict) else [])

        for idx, item in tqdm(enumerate(items), desc=f"解析 {json_name}"):
            if not isinstance(item, dict): continue

            #  精准打击：根据真实的键名提取数据
            img_val = item.get('optical_image_path') or item.get('image')
            q_text = str(item.get('question', '')).strip()
            a_text = str(item.get('gt', '')).strip()  # 真实答案藏在 'gt' 里！

            if not img_val or not q_text or not a_text:
                continue

            # 从雷达中找到原图绝对路径
            img_basename = os.path.basename(str(img_val))
            src_img_path = img_map.get(img_basename) or img_map.get(os.path.splitext(img_basename)[0])
            if not src_img_path:
                continue

            # 规范化命名与保存
            img_filename = f"{ds_name}_{split_type}_{idx}.jpg"
            tgt_img_path = os.path.join(dirs["images"], img_filename)

            try:
                # 统一转换为 RGB JPG 格式，保障联邦训练不出错
                Image.open(src_img_path).convert('RGB').save(tgt_img_path, format="JPEG")
            except:
                continue

            # 构建标准 LLaVA 会话格式
            standard_data[split_type].append({
                "id": f"{ds_name}_{split_type}_{idx}",
                "image": f"dataset/{ds_name}/images/{img_filename}",  # repo-root relative image path
                "conversations": [
                    {"from": "human", "value": f"<image>\n{q_text}"},
                    {"from": "gpt", "value": a_text}
                ]
            })

    # 进行 Dirichlet 严苛重采样与容量切分
    dirichlet_split_and_save(standard_data["train"], dirs["train"], splits, MAX_TRAIN_SAMPLES, "Train", min_samples_per_split=MIN_TRAIN_SAMPLES_PER_SPLIT)
    dirichlet_split_and_save(standard_data["test"], dirs["test"], splits, MAX_TEST_SAMPLES, "Test", shared_across_splits=SHARE_TEST_ACROSS_SUBSETS)

# ==========================================
#  主引擎调度
# ==========================================
def main():
    """生成新的确定性划分，不声称重现历史随机划分。"""
    import argparse
    from pathlib import Path
    global RAW_BASE_DIR, TARGET_BASE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--datasets', nargs='+', choices=sorted(DATASET_CONFIG), default=list(DATASET_CONFIG))
    args = ap.parse_args()
    RAW_BASE_DIR = str(args.raw_root.resolve())
    if args.output_root.exists():
        raise FileExistsError('Output must be a new directory')
    for name in args.datasets:
        if not (args.raw_root / name).is_dir():
            raise FileNotFoundError(args.raw_root / name)
    args.output_root.mkdir(parents=True)
    TARGET_BASE_DIR = str(args.output_root / 'dataset')
    random.seed(args.seed)
    np.random.seed(args.seed)
    for name in args.datasets:
        cfg = DATASET_CONFIG[name]
        globals()['process_' + cfg['type']](name, cfg['splits'])
        for split in ('train', 'test'):
            for idx in range(cfg['splits']):
                target = Path(TARGET_BASE_DIR) / name / split / f'dataset-{idx}.json'
                if not target.is_file():
                    raise RuntimeError(f'Missing output: {target}')
    (args.output_root / 'BUILD.json').write_text(json.dumps({'mode':'new_seeded_partition', 'historical_identity':False, 'seed':args.seed, 'datasets':args.datasets}, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
