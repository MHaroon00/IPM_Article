# -*- coding: utf-8 -*-
"""
Protocol B (deduplicated) experiments for the R3 revision of
"An Automated Confidence-Aware Framework for Sarcasm Detection in App Reviews
Using Inference-Optimized Transformer Language Models".

Command-line version of the Colab notebook R3_ProtocolB_Colab.ipynb (same code).
Usage: python rerun_dedup_experiments.py --data_dir DATA --out_dir OUT [--quick_check]

Steps:
  1. Load the two released benchmark files and deduplicate them before any split.
     Reviews are matched on a normalised key (lower case, collapsed whitespace,
     non-ASCII bytes removed so that mis-encoded copies of the same review match).
     A text that carries two different labels is removed; a text that is repeated
     with the same label is kept once.
  2. Draw stratified 70:15:15 partitions with seeds 42, 123 and 2024 (a stratified
     70:30 split followed by a stratified halving of the 30% part) and save them.
  3. Fine-tune BERT, RoBERTa, DistilBERT, TinyBERT and MiniLM on both tasks with the
     hyperparameters of the manuscript (eager attention, FP16), keeping the epoch with
     the best validation Macro F1, and store class probabilities for every validation
     and test review.
  4. Compute accuracy, Macro F1, Weighted F1 and per-class scores, seed means with
     t-based 95% confidence intervals, paired bootstrap and exact McNemar tests for all
     model pairs, and the Holm correction over the 60 comparisons.
  5. Validate confidence-based routing: expected calibration error before and after
     temperature scaling, a threshold sweep, and an independent second-opinion stage.
  6. Recompute the TF-IDF, surface-feature and majority-class baselines.
  7. Write every value the manuscript needs to results/rerun_macros.tex and the two
     tables to results/tab_dedup_main.tex and results/tab_routing.tex.
"""
import hashlib, itertools, json, math, os, random, re, time

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
# Replace these identifiers with the exact checkpoints of the original grid if they differ.
MODELS = {
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
    "DistilBERT": "distilbert-base-uncased",
    "TinyBERT": "huawei-noah/TinyBERT_General_4L_312D",
    "MiniLM": "microsoft/MiniLM-L12-H384-uncased",
}
# TinyBERT and MiniLM are BERT architectures whose repositories ship only vocab.txt, so
# they are loaded with the BERT classes directly instead of relying on Auto* inference.
BERT_FAMILY = {"TinyBERT", "MiniLM"}
MODEL_ORDER = ["BERT", "RoBERTa", "DistilBERT", "TinyBERT", "MiniLM"]
SEEDS = [42, 123, 2024]
HP = dict(batch_size=16, epochs=3, lr=2e-5, warmup_ratio=0.1, max_len=128,
          weight_decay=0.01, fp16=True)
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
TAU = 0.7
N_BOOT = 2500

# Mean Macro F1 under Protocol A (the aggregated binary and sarcasm-type tables of the manuscript).
ORIGINAL_MACRO_F1 = {
    "binary": {"BERT": 0.9941, "RoBERTa": 0.9889, "DistilBERT": 0.9941, "TinyBERT": 0.9774, "MiniLM": 0.9911},
    "type": {"BERT": 0.8301, "RoBERTa": 0.8078, "DistilBERT": 0.8273, "TinyBERT": 0.7058, "MiniLM": 0.8092},
}
TYPE_CLASSES = ["Humor", "Irony", "Flattery", "Self-Deprecation", "Passive-Aggression"]
BINARY_CLASSES = ["Non-Sarcastic", "Sarcastic"]
CLASSES_OF = {"binary": BINARY_CLASSES, "type": TYPE_CLASSES}
BINARY_FILE = "annotated_comments_binary_1.csv"
TYPE_FILE = "Sarcasm_multiclassification_1.csv"

# Values the manuscript reports for the released files and for Protocol B. The data check
# stops the run if the files on Drive do not reproduce them.
EXPECTED = {
    "md5": {BINARY_FILE: "295c7ff729e74ec68abf9111356e5efb", TYPE_FILE: "55f55c3db7df373584f5178eb04e39b3"},
    "raw_counts": {
        "binary": {"Non-Sarcastic": 6539, "Sarcastic": 3460},
        "type": {"Humor": 1709, "Irony": 536, "Flattery": 513, "Self-Deprecation": 439, "Passive-Aggression": 263},
    },
    "dedup_counts": {
        "binary": {"Non-Sarcastic": 6475, "Sarcastic": 3451},
        "type": {"Humor": 1555, "Irony": 385, "Flattery": 489, "Self-Deprecation": 297, "Passive-Aggression": 215},
    },
    "conflicting_texts": {"binary": 0, "type": 230},
    "split_sizes": {"binary": (6948, 1489, 1489), "type": (2058, 441, 442)},
    "oversampled_train": {"binary": 9064, "type": 5440},
}


# --------------------------------------------------------------------------------------
# Data loading, deduplication and partitioning
# --------------------------------------------------------------------------------------
def norm_key(s):
    s = re.sub(r"[^\x00-\x7f]", "", str(s))
    return re.sub(r"\s+", " ", s).strip().lower()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tasks(data_dir):
    b = pd.read_csv(os.path.join(data_dir, BINARY_FILE))
    m = pd.read_csv(os.path.join(data_dir, TYPE_FILE))
    b = b[["comment_Text", "Have_Sarcasm"]].rename(columns={"comment_Text": "text", "Have_Sarcasm": "label"})
    m = m[["comment_Text", "Sarcasm_Type"]].rename(columns={"comment_Text": "text", "Sarcasm_Type": "label"})
    b["label"] = b["label"].astype(str).str.strip().str.capitalize().map({"Yes": "Sarcastic", "No": "Non-Sarcastic"})
    m["label"] = m["label"].astype(str).str.strip()
    return {"binary": b, "type": m}


def deduplicate(d):
    d = d.copy()
    d["key"] = d["text"].map(norm_key)
    n_labels = d.groupby("key")["label"].nunique()
    conflict = set(n_labels[n_labels > 1].index)
    out = d[~d["key"].isin(conflict)].drop_duplicates("key", keep="first").reset_index(drop=True)
    stats = dict(rows=len(d), unique_texts=int(d["key"].nunique()),
                 redundant_rows=int(len(d) - d["key"].nunique()),
                 conflicting_texts=len(conflict), rows_in_conflicting=int(d["key"].isin(conflict).sum()),
                 kept=len(out))
    return out, stats


def make_splits(d, seed):
    from sklearn.model_selection import train_test_split
    tr, tmp = train_test_split(d, test_size=0.3, stratify=d["label"], random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp["label"], random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def oversample_index(labels, seed):
    from imblearn.over_sampling import RandomOverSampler
    idx, _ = RandomOverSampler(random_state=seed).fit_resample(np.arange(len(labels)).reshape(-1, 1), labels)
    return idx.ravel()


def verify_data(data_dir, strict=True):
    """Checks the files on Drive against the values reported in the manuscript."""
    problems, notes = [], []
    for f in (BINARY_FILE, TYPE_FILE):
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{f} not found in {data_dir}")
        h = md5(path)
        if h == EXPECTED["md5"][f]:
            notes.append(f"{f}: identical to the file used for the manuscript (MD5 {h}).")
        else:
            notes.append(f"{f}: bytes differ from the manuscript copy (MD5 {h}); contents are checked below.")
    raw = load_tasks(data_dir)
    dedup, stats = {}, {}
    for task in ("binary", "type"):
        if raw[task]["label"].isna().any():
            problems.append(f"{task}: {int(raw[task]['label'].isna().sum())} rows have an unrecognised label")
        counts = raw[task]["label"].value_counts().to_dict()
        if counts != EXPECTED["raw_counts"][task]:
            problems.append(f"{task}: class counts {counts} differ from {EXPECTED['raw_counts'][task]}")
        dedup[task], stats[task] = deduplicate(raw[task])
        dcounts = dedup[task]["label"].value_counts().to_dict()
        if dcounts != EXPECTED["dedup_counts"][task]:
            problems.append(f"{task}: deduplicated counts {dcounts} differ from {EXPECTED['dedup_counts'][task]}")
        if stats[task]["conflicting_texts"] != EXPECTED["conflicting_texts"][task]:
            problems.append(f"{task}: {stats[task]['conflicting_texts']} conflicting texts, expected {EXPECTED['conflicting_texts'][task]}")
        for seed in SEEDS:
            tr, va, te = make_splits(dedup[task], seed)
            sizes = (len(tr), len(va), len(te))
            if sizes != EXPECTED["split_sizes"][task]:
                problems.append(f"{task} seed {seed}: split sizes {sizes} differ from {EXPECTED['split_sizes'][task]}")
            if set(te["key"]) & set(tr["key"]) or set(va["key"]) & set(tr["key"]) or set(va["key"]) & set(te["key"]):
                problems.append(f"{task} seed {seed}: a review text is shared between partitions")
            n_os = len(oversample_index(tr["label"], seed))
            if n_os != EXPECTED["oversampled_train"][task]:
                problems.append(f"{task} seed {seed}: {n_os} oversampled training reviews, expected {EXPECTED['oversampled_train'][task]}")
    if problems and strict:
        raise ValueError("The dataset does not reproduce the manuscript values:\n  " + "\n  ".join(problems))
    return raw, dedup, stats, notes, problems


# --------------------------------------------------------------------------------------
# Preprocessing. If the original grid used a different function, paste it here so that
# Protocol A and Protocol B differ only in the deduplication step.
# --------------------------------------------------------------------------------------
def preprocess(text):
    t = str(text)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    try:
        import contractions
        t = contractions.fix(t)
    except Exception:
        pass
    try:
        import emoji
        t = emoji.demojize(t, delimiters=(" ", " "))
    except Exception:
        pass
    t = t.lower()
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def set_seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_key, num_labels):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              BertForSequenceClassification, BertTokenizerFast)
    name = MODELS[model_key]
    if model_key in BERT_FAMILY:
        tok = BertTokenizerFast.from_pretrained(name)
        model_cls = BertForSequenceClassification
    else:
        tok = AutoTokenizer.from_pretrained(name)
        model_cls = AutoModelForSequenceClassification
    try:
        model = model_cls.from_pretrained(name, num_labels=num_labels, attn_implementation="eager")
    except (TypeError, ValueError):
        model = model_cls.from_pretrained(name, num_labels=num_labels)
    attn = getattr(model.config, "_attn_implementation", None) or "eager"
    return model, tok, attn


def make_scaler(enabled):
    import torch
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def predict_proba(model, tok, texts, device, max_len, batch_size=64):
    import torch
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True, padding="max_length",
                      max_length=max_len, return_tensors="pt").to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=HP["fp16"] and device == "cuda"):
                logits = model(**enc).logits
            out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def run_done(run_dir, task, model_key, seed):
    return os.path.exists(os.path.join(run_dir, f"{task}_{model_key}_seed{seed}_meta.json"))


def train_one(model_key, task, seed, tr, va, te, run_dir, hp=None):
    import torch
    from sklearn.metrics import f1_score
    from transformers import get_linear_schedule_with_warmup
    hp = dict(HP, **(hp or {}))
    classes = CLASSES_OF[task]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = hp["fp16"] and device == "cuda"
    set_seed(seed)
    lab2id = {c: i for i, c in enumerate(classes)}

    tr_os = tr.iloc[oversample_index(tr["label"], seed)].reset_index(drop=True)
    model, tok, attn = load_model_and_tokenizer(model_key, len(classes))
    model.to(device)

    x_tr = [preprocess(t) for t in tr_os["text"]]; y_tr = np.array([lab2id[l] for l in tr_os["label"]])
    x_va = [preprocess(t) for t in va["text"]]; y_va = np.array([lab2id[l] for l in va["label"]])
    x_te = [preprocess(t) for t in te["text"]]

    steps_per_epoch = math.ceil(len(x_tr) / hp["batch_size"])
    total = steps_per_epoch * hp["epochs"]
    no_decay = ["bias", "LayerNorm.weight"]
    params = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": hp["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=hp["lr"])
    sch = get_linear_schedule_with_warmup(opt, int(hp["warmup_ratio"] * total), total)
    scaler = make_scaler(use_amp)

    best_f1, best_state, best_epoch, history = -1.0, None, -1, []
    rng = np.random.RandomState(seed)
    t0 = time.time()
    for epoch in range(hp["epochs"]):
        model.train()
        order = rng.permutation(len(x_tr))
        for s in range(steps_per_epoch):
            b = order[s * hp["batch_size"]:(s + 1) * hp["batch_size"]]
            enc = tok([x_tr[i] for i in b], truncation=True, padding="max_length",
                      max_length=hp["max_len"], return_tensors="pt").to(device)
            yb = torch.tensor(y_tr[b], device=device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = model(**enc, labels=yb).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
        pv = predict_proba(model, tok, x_va, device, hp["max_len"])
        f1 = f1_score(y_va, pv.argmax(1), average="macro")
        history.append(float(f1))
        print(f"    epoch {epoch + 1}/{hp['epochs']}: validation Macro F1 {f1:.4f}  ({time.time() - t0:.0f} s)")
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0
    model.load_state_dict(best_state)

    pv = predict_proba(model, tok, x_va, device, hp["max_len"])
    pt = predict_proba(model, tok, x_te, device, hp["max_len"])
    for split, frame, p in [("val", va, pv), ("test", te, pt)]:
        df = pd.DataFrame({"key": frame["key"], "gold": frame["label"]})
        for j, c in enumerate(classes):
            df[f"p_{c}"] = p[:, j]
        df["pred"] = [classes[k] for k in p.argmax(1)]
        df.to_csv(os.path.join(run_dir, f"{task}_{model_key}_seed{seed}_{split}.csv"), index=False)
    # The meta file is written last and atomically: its presence marks a finished run.
    _atomic_json(dict(best_epoch=best_epoch, best_val_macro_f1=float(best_f1), val_history=history,
                      train_time_s=train_time, checkpoint=MODELS[model_key], attn_implementation=attn,
                      device=device, fp16=use_amp, hp=hp, train_size_after_oversampling=len(x_tr)),
                 os.path.join(run_dir, f"{task}_{model_key}_seed{seed}_meta.json"))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return best_f1, train_time


def run_grid(dedup, out_dir, hp=None, subsample=None):
    """Trains every model on every task and seed, skipping runs that are already finished."""
    run_dir = os.path.join(out_dir, "runs"); split_dir = os.path.join(out_dir, "splits")
    os.makedirs(run_dir, exist_ok=True); os.makedirs(split_dir, exist_ok=True)
    jobs = [(t, s, m) for t in ("binary", "type") for s in SEEDS for m in MODEL_ORDER]
    todo = [(t, s, m) for (t, s, m) in jobs if not run_done(run_dir, t, m, s)]
    print(f"{len(jobs) - len(todo)} of {len(jobs)} runs already finished; {len(todo)} to train.")
    t_start, done_now = time.time(), 0
    for task in ("binary", "type"):
        data = dedup[task]
        if subsample:
            from sklearn.model_selection import train_test_split
            data, _ = train_test_split(data, train_size=subsample, stratify=data["label"], random_state=0)
            data = data.reset_index(drop=True)
        for seed in SEEDS:
            tr, va, te = make_splits(data, seed)
            assert not set(te["key"]) & set(tr["key"]) and not set(va["key"]) & set(tr["key"])
            for name, frame in [("train", tr), ("val", va), ("test", te)]:
                frame.to_csv(os.path.join(split_dir, f"{task}_seed{seed}_{name}.csv"), index=False)
            for model_key in MODEL_ORDER:
                if run_done(run_dir, task, model_key, seed):
                    continue
                k = jobs.index((task, seed, model_key)) + 1
                print(f"[{k}/{len(jobs)}] {model_key} | {task} | seed {seed}")
                best, secs = train_one(model_key, task, seed, tr, va, te, run_dir, hp)
                done_now += 1
                left = len(todo) - done_now
                eta = (time.time() - t_start) / done_now * left
                print(f"    finished in {secs / 60:.1f} min, best validation Macro F1 {best:.4f}; "
                      f"{left} runs left, about {eta / 60:.0f} min remaining")
    print("All runs finished.")


# --------------------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------------------
def load_run(run_dir, task, model, seed, split):
    return pd.read_csv(os.path.join(run_dir, f"{task}_{model}_seed{seed}_{split}.csv"))


def proba(df, classes):
    return df[[f"p_{c}" for c in classes]].to_numpy()


def t_ci(values):
    from scipy import stats
    v = np.asarray(values, dtype=float); n = len(v); m = v.mean(); sd = v.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return m, sd, m - h, m + h


def ece(p, y, bins=15):
    conf = p.max(1); pred = p.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(acc[sel].mean() - conf[sel].mean())
    return e


def fit_temperature(p_val, y_val):
    from scipy.optimize import minimize_scalar
    logit = np.log(np.clip(p_val, 1e-12, 1))
    def nll(T):
        z = logit / T; z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y_val)), y_val].mean()
    return minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x


def apply_temperature(p, T):
    z = np.log(np.clip(p, 1e-12, 1)) / T; z = z - z.max(1, keepdims=True)
    e = np.exp(z); return e / e.sum(1, keepdims=True)


def holm(pvals):
    p = np.asarray(pvals, dtype=float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx]); adj[idx] = min(1.0, running)
    return adj


def macro_f1_fast(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean()


def paired_tests(ya, yb, y, k, seed):
    from scipy.stats import binomtest
    rng = np.random.RandomState(seed); n = len(y)
    delta = macro_f1_fast(y, yb, k) - macro_f1_fast(y, ya, k)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s = rng.randint(0, n, n); boots[i] = macro_f1_fast(y[s], yb[s], k) - macro_f1_fast(y[s], ya[s], k)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_boot = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    ca, cb = ya == y, yb == y
    b_, c_ = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p_mc = 1.0 if b_ + c_ == 0 else binomtest(b_, b_ + c_, 0.5).pvalue
    return dict(delta_macro_f1=delta, ci_low=lo, ci_high=hi, bootstrap_p=p_boot,
                a_correct_b_wrong=b_, a_wrong_b_correct=c_, mcnemar_p=p_mc)


def surface(t):
    s = str(t); n = max(len(s), 1); w = s.split()
    return [np.log1p(len(s)), np.log1p(len(w)), s.count("!") / n * 100, s.count("?") / n * 100,
            sum(ch.isupper() for ch in s) / n, sum(ord(ch) > 127 for ch in s) / n,
            s.count(".") / n * 100, s.count('"') + s.count("'"), sum(ch.isdigit() for ch in s) / n]


def lexical_baselines(tasks_raw, tasks_dedup):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = []
    for task in ["binary", "type"]:
        for version, d in [("released", tasks_raw[task]), ("deduplicated", tasks_dedup[task])]:
            d = d.copy(); d["key"] = d["text"].map(norm_key)
            for seed in SEEDS:
                tr, va, te = make_splits(d, seed)
                tf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                   LogisticRegression(max_iter=3000, class_weight="balanced", C=4))
                tf.fit(tr["text"], tr["label"]); p = tf.predict(te["text"])
                rows.append(dict(task=task, data=version, model="TF-IDF + LR", seed=seed,
                                 acc=accuracy_score(te["label"], p), macro_f1=f1_score(te["label"], p, average="macro")))
                sf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
                sf.fit(np.array([surface(t) for t in tr["text"]]), tr["label"])
                p = sf.predict(np.array([surface(t) for t in te["text"]]))
                rows.append(dict(task=task, data=version, model="Surface features + LR", seed=seed,
                                 acc=accuracy_score(te["label"], p), macro_f1=f1_score(te["label"], p, average="macro")))
                maj = tr["label"].value_counts().idxmax(); p = [maj] * len(te)
                rows.append(dict(task=task, data=version, model="Majority class", seed=seed,
                                 acc=accuracy_score(te["label"], p), macro_f1=f1_score(te["label"], p, average="macro")))
    return pd.DataFrame(rows)


def fmt(x, d=4):
    return f"{x:.{d}f}"


def signed(x, d=4):
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.{d}f}"


def describe_environment(e):
    gpu = e["gpu"]
    article = "a" if gpu.lower().startswith(("tesla", "geforce", "quadro")) else "an"
    return (f"{article} {gpu} GPU with Python {e['python']}, PyTorch {e['torch']} with CUDA {e['cuda']} "
            f"and Transformers {e['transformers']}")


def record_environment(out_dir):
    import platform, torch, transformers
    path = os.path.join(out_dir, "environment.json")
    if os.path.exists(path):
        return json.load(open(path))
    e = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
             python=platform.python_version(), torch=torch.__version__, cuda=str(torch.version.cuda),
             transformers=transformers.__version__)
    _atomic_json(e, path)
    return e


# --------------------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------------------
def analyse(out_dir, tasks_raw, tasks_dedup, dedup_stats, quick=False):
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    run_dir = os.path.join(out_dir, "runs"); res_dir = os.path.join(out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    missing = [(t, m, s) for t in ("binary", "type") for m in MODEL_ORDER for s in SEEDS if not run_done(run_dir, t, m, s)]
    if missing:
        raise RuntimeError(f"{len(missing)} runs are not finished yet, for example {missing[0]}. Run the training cell again.")
    metas = {(t, m, s): json.load(open(os.path.join(run_dir, f"{t}_{m}_seed{s}_meta.json")))
             for t in ("binary", "type") for m in MODEL_ORDER for s in SEEDS}

    # ---- per-run metrics ----
    rows, per_class = [], []
    for task in ["binary", "type"]:
        cls = CLASSES_OF[task]
        for model in MODEL_ORDER:
            for seed in SEEDS:
                te = load_run(run_dir, task, model, seed, "test")
                y = te["gold"]; p = te["pred"]
                rows.append(dict(task=task, model=model, seed=seed, acc=accuracy_score(y, p),
                                 macro_f1=f1_score(y, p, average="macro", labels=cls, zero_division=0),
                                 weighted_f1=f1_score(y, p, average="weighted", labels=cls, zero_division=0)))
                pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=cls, zero_division=0)
                for c, a, b_, f_, s_ in zip(cls, pr, rc, f1, sup):
                    per_class.append(dict(task=task, model=model, seed=seed, cls=c, precision=a, recall=b_, f1=f_, support=s_))
    runs = pd.DataFrame(rows); runs.to_csv(os.path.join(res_dir, "per_run_metrics.csv"), index=False)
    pd.DataFrame(per_class).to_csv(os.path.join(res_dir, "per_class_metrics.csv"), index=False)

    agg = {}
    for (task, model), g in runs.groupby(["task", "model"]):
        agg[(task, model)] = dict(acc=t_ci(g["acc"]), f1=t_ci(g["macro_f1"]), wf1=t_ci(g["weighted_f1"]))
    agg_df = pd.DataFrame([dict(task=t, model=m, acc_mean=v["acc"][0], acc_sd=v["acc"][1], f1_mean=v["f1"][0], f1_sd=v["f1"][1],
                                f1_ci_low=v["f1"][2], f1_ci_high=v["f1"][3], wf1_mean=v["wf1"][0], wf1_sd=v["wf1"][1],
                                f1_protocol_a=ORIGINAL_MACRO_F1[t][m], delta_vs_protocol_a=v["f1"][0] - ORIGINAL_MACRO_F1[t][m])
                           for (t, m), v in agg.items()])
    agg_df.to_csv(os.path.join(res_dir, "aggregate_mean_sd_ci95.csv"), index=False)

    # ---- main deduplicated table ----
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Binary task} & \multicolumn{4}{c}{Sarcasm-type task} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
             r"Model & Accuracy & Macro F1 & $\Delta$ & Accuracy & Macro F1 & Weighted F1 & $\Delta$ \\", r"\midrule"]
    for model in MODEL_ORDER:
        b = agg[("binary", model)]; t = agg[("type", model)]
        db = b["f1"][0] - ORIGINAL_MACRO_F1["binary"][model]; dt = t["f1"][0] - ORIGINAL_MACRO_F1["type"][model]
        lines.append(f"{model} & {fmt(b['acc'][0])} $\\pm$ {fmt(b['acc'][1])} & {fmt(b['f1'][0])} $\\pm$ {fmt(b['f1'][1])} & {signed(db)} & "
                     f"{fmt(t['acc'][0])} $\\pm$ {fmt(t['acc'][1])} & {fmt(t['f1'][0])} $\\pm$ {fmt(t['f1'][1])} & "
                     f"{fmt(t['wf1'][0])} $\\pm$ {fmt(t['wf1'][1])} & {signed(dt)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(res_dir, "tab_dedup_main.tex"), "w").write("\n".join(lines) + "\n")

    # ---- paired tests with Holm ----
    tests = []
    for task in ["binary", "type"]:
        cls = CLASSES_OF[task]; lab2id = {c: i for i, c in enumerate(cls)}
        for seed in SEEDS:
            frames = {m: load_run(run_dir, task, m, seed, "test") for m in MODEL_ORDER}
            y = frames["BERT"]["gold"].map(lab2id).to_numpy()
            for a, b in itertools.combinations(MODEL_ORDER, 2):
                assert (frames[a]["key"].values == frames[b]["key"].values).all(), "test partitions differ between models"
                r = paired_tests(frames[a]["pred"].map(lab2id).to_numpy(), frames[b]["pred"].map(lab2id).to_numpy(), y, len(cls), seed)
                r.update(task=task, seed=seed, model_a=a, model_b=b); tests.append(r)
    tests = pd.DataFrame(tests); tests["mcnemar_p_holm"] = holm(tests["mcnemar_p"])
    tests.to_csv(os.path.join(res_dir, "paired_bootstrap_mcnemar_holm.csv"), index=False)
    involves_tiny = (tests["model_a"] == "TinyBERT") | (tests["model_b"] == "TinyBERT")
    holm_tiny = int((tests.loc[involves_tiny, "mcnemar_p_holm"] < 0.05).sum())
    holm_top = int((tests.loc[~involves_tiny, "mcnemar_p_holm"] < 0.05).sum())
    raw_top = int((tests.loc[~involves_tiny, "mcnemar_p"] < 0.05).sum())

    # ---- routing validation ----
    # Calibration is computed per seed and averaged. The threshold sweep and the second-opinion
    # analysis pool the three test partitions, because only a handful of reviews fall below the
    # threshold in each seed and a mean of per-seed accuracies would weight them unevenly.
    routing_rows, macros_routing, sweep = [], {}, {}
    for task in ["binary", "type"]:
        cls = CLASSES_OF[task]; lab2id = {c: i for i, c in enumerate(cls)}
        val_means = {m: np.mean([metas[(task, m, s)]["best_val_macro_f1"] for s in SEEDS]) for m in MODEL_ORDER}
        sel = max(val_means, key=val_means.get)
        others = [m for m in MODEL_ORDER if m not in (sel, "TinyBERT")]
        per_seed, pooled = [], dict(conf=[], pred=[], y=[], new=[])
        for seed in SEEDS:
            va = load_run(run_dir, task, sel, seed, "val"); te = load_run(run_dir, task, sel, seed, "test")
            yv = va["gold"].map(lab2id).to_numpy(); yt = te["gold"].map(lab2id).to_numpy()
            pv = proba(va, cls); pt = proba(te, cls)
            T = fit_temperature(pv, yv); pts = apply_temperature(pt, T)
            conf = pt.max(1); pred = pt.argmax(1)
            assert (pts.argmax(1) == pred).all()
            ens = np.mean([proba(load_run(run_dir, task, m, seed, "test"), cls) for m in others], axis=0)
            routed = conf < TAU
            new_pred = pred.copy(); new_pred[routed] = ens[routed].argmax(1)
            per_seed.append(dict(seed=seed, T=T, ece=ece(pt, yt), ece_ts=ece(pts, yt), routed_n=int(routed.sum()),
                                 errors=int((pred != yt).sum()), errors_routed=int(((pred != yt) & routed).sum()),
                                 f1_before=f1_score(yt, pred, average="macro", labels=list(range(len(cls))), zero_division=0),
                                 f1_after=f1_score(yt, new_pred, average="macro", labels=list(range(len(cls))), zero_division=0)))
            for k, v in (("conf", conf), ("pred", pred), ("y", yt), ("new", new_pred)):
                pooled[k].append(v)
        conf, pred, y, new = (np.concatenate(pooled[k]) for k in ("conf", "pred", "y", "new"))
        ok = pred == y
        sw = dict(n=len(y), errors=int((~ok).sum()))
        for tau in THRESHOLDS:
            r = conf < tau
            sw[f"routed_n_{tau}"] = int(r.sum()); sw[f"routed_{tau}"] = 100 * r.mean()
            sw[f"acc_accepted_{tau}"] = ok[~r].mean() if (~r).any() else np.nan
            sw[f"acc_routed_{tau}"] = ok[r].mean() if r.any() else np.nan
            sw[f"errors_routed_{tau}"] = int((~ok & r).sum())
        r = conf < TAU
        sw["ens_before"] = ok[r].mean() if r.any() else np.nan
        sw["ens_after"] = (new[r] == y[r]).mean() if r.any() else np.nan
        sw["ens_fixed"] = int(((pred != y) & (new == y)).sum())
        sw["ens_broken"] = int(((pred == y) & (new != y)).sum())
        ps = pd.DataFrame(per_seed); ps["task"] = task; ps["model"] = sel; routing_rows.append(ps)
        mean = ps.mean(numeric_only=True)
        sweep[task] = sw; macros_routing[task] = (sel, mean)
    pd.concat(routing_rows).to_csv(os.path.join(res_dir, "routing_validation_per_seed.csv"), index=False)
    pd.DataFrame([dict(task=t, model=macros_routing[t][0], **sweep[t]) for t in sweep]).to_csv(
        os.path.join(res_dir, "routing_threshold_sweep_pooled.csv"), index=False)

    def cell(v, pct=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        return f"{v:.1f}" if pct else f"{v:.4f}"
    def routed_cell(s, tau):
        return f"{s[f'routed_n_{tau}']} ({s[f'routed_{tau}']:.2f})"
    lines = [r"\begin{tabular}{cccccccc}", r"\toprule",
             r" & \multicolumn{4}{c}{Binary task (" + macros_routing["binary"][0] + r", " + f"{sweep['binary']['n']:,}" +
             r" reviews)} & \multicolumn{4}{c}{Sarcasm-type task (" + macros_routing["type"][0] + r", " + f"{sweep['type']['n']:,}" + r" reviews)} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"$\tau$ & Routed, $n$ (\%) & Acc.\ accepted & Acc.\ routed & Errors routed & Routed, $n$ (\%) & Acc.\ accepted & Acc.\ routed & Errors routed \\", r"\midrule"]
    lines[0] = r"\begin{tabular}{ccccccccc}"
    for tau in THRESHOLDS:
        b = sweep["binary"]; t = sweep["type"]
        lines.append(f"{tau:.1f} & {routed_cell(b, tau)} & {cell(b[f'acc_accepted_{tau}'])} & {cell(b[f'acc_routed_{tau}'])} & "
                     f"{b[f'errors_routed_{tau}']} of {b['errors']} & {routed_cell(t, tau)} & {cell(t[f'acc_accepted_{tau}'])} & "
                     f"{cell(t[f'acc_routed_{tau}'])} & {t[f'errors_routed_{tau}']} of {t['errors']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(res_dir, "tab_routing.tex"), "w").write("\n".join(lines) + "\n")

    # ---- baselines ----
    base = lexical_baselines(tasks_raw, tasks_dedup)
    base.to_csv(os.path.join(res_dir, "lexical_baselines_per_seed.csv"), index=False)
    bsum = base.groupby(["task", "data", "model"]).agg(acc=("acc", "mean"), acc_sd=("acc", "std"),
                                                          f1=("macro_f1", "mean"), f1_sd=("macro_f1", "std")).round(4)
    bsum.to_csv(os.path.join(res_dir, "lexical_baselines_summary.csv"))

    # ---- macros ----
    best = {t: max(MODEL_ORDER, key=lambda m: agg[(t, m)]["f1"][0]) for t in ["binary", "type"]}
    shifts = {t: [agg[(t, m)]["f1"][0] - ORIGINAL_MACRO_F1[t][m] for m in MODEL_ORDER] for t in ["binary", "type"]}
    def direction(v):
        if all(abs(x) < 0.001 for x in v): return "was essentially unchanged for every architecture"
        if all(x > 0 for x in v): return "increased for every architecture"
        if all(x < 0 for x in v): return "decreased for every architecture"
        return "changed in different directions across architectures"
    M = {
        "RRbinBestModel": best["binary"], "RRbinBestF": fmt(agg[("binary", best["binary"])]["f1"][0]),
        "RRtypeBestModel": best["type"], "RRtypeBestF": fmt(agg[("type", best["type"])]["f1"][0]),
        "RRbinTinyF": fmt(agg[("binary", "TinyBERT")]["f1"][0]), "RRtypeTinyF": fmt(agg[("type", "TinyBERT")]["f1"][0]),
        "RRbinShiftMin": signed(min(shifts["binary"])), "RRbinShiftMax": signed(max(shifts["binary"])),
        "RRtypeShiftMin": signed(min(shifts["type"])), "RRtypeShiftMax": signed(max(shifts["type"])),
        "RRbinDirection": direction(shifts["binary"]), "RRtypeDirection": direction(shifts["type"]),
        "RRholmTiny": str(holm_tiny), "RRholmTop": str(holm_top), "RRrawTop": str(raw_top),
    }
    top_sig = tests[(~involves_tiny) & (tests["mcnemar_p_holm"] < 0.05)]
    M["RRholmTopList"] = "; ".join(f"{r.model_a} and {r.model_b} on the {'binary' if r.task == 'binary' else 'sarcasm-type'} task at seed {r.seed}"
                                   for r in top_sig.itertuples()) or "none"
    for task, short in [("binary", "bin"), ("type", "type")]:
        M[f"RR{short}HolmTiny"] = str(int((tests.loc[involves_tiny & (tests["task"] == task), "mcnemar_p_holm"] < 0.05).sum()))
    for task, short in [("binary", "bin"), ("type", "type")]:
        sel, mean = macros_routing[task]; sw = sweep[task]
        M.update({
            f"RR{short}RouteModel": sel, f"RR{short}ECE": fmt(mean["ece"]), f"RR{short}ECEts": fmt(mean["ece_ts"]),
            f"RR{short}Temp": f"{mean['T']:.2f}", f"RR{short}N": f"{sw['n']:,}", f"RR{short}Errors": str(sw["errors"]),
            f"RR{short}RoutedN": str(sw[f"routed_n_{TAU}"]), f"RR{short}Routed": f"{sw[f'routed_{TAU}']:.2f}",
            f"RR{short}ErrRouted": str(sw[f"errors_routed_{TAU}"]),
            f"RR{short}AccAccepted": cell(sw[f"acc_accepted_{TAU}"]), f"RR{short}AccRouted": cell(sw[f"acc_routed_{TAU}"]),
            f"RR{short}EnsBefore": cell(sw["ens_before"]), f"RR{short}EnsAfter": cell(sw["ens_after"]),
            f"RR{short}EnsFixed": str(sw["ens_fixed"]), f"RR{short}EnsBroken": str(sw["ens_broken"]),
            f"RR{short}FBefore": fmt(mean["f1_before"]), f"RR{short}FAfter": fmt(mean["f1_after"]),
        })
    env_path = os.path.join(out_dir, "environment.json")
    if os.path.exists(env_path):
        M["RRenvironment"] = describe_environment(json.load(open(env_path)))
    with open(os.path.join(res_dir, "rerun_macros.tex"), "w") as f:
        f.write("% Generated by the Protocol B notebook. Do not edit by hand.\n")
        for k, v in M.items():
            f.write(f"\\renewcommand{{\\{k}}}{{{v}}}\n")

    # ---- checks of the sentences written in the manuscript ----
    attn_used = sorted({v["attn_implementation"] for v in metas.values()})
    b, t = sweep["binary"], sweep["type"]
    required = {
        "All runs used eager attention (Section 5.3)": attn_used == ["eager"],
        "TinyBERT has the lowest mean Macro F1 on the binary task (Section 6.5.1)":
            min(MODEL_ORDER, key=lambda m: agg[("binary", m)]["f1"][0]) == "TinyBERT",
        "TinyBERT has the lowest mean Macro F1 on the sarcasm-type task (Section 6.5.1)":
            min(MODEL_ORDER, key=lambda m: agg[("type", m)]["f1"][0]) == "TinyBERT",
        "Fewer than 1% of binary test reviews are routed at tau = 0.7 (Section 6.6)": b[f"routed_{TAU}"] < 1.0,
        "Routed sarcasm-type predictions are less accurate than accepted ones at every threshold (Section 6.6)":
            all(t[f"acc_routed_{tau}"] < t[f"acc_accepted_{tau}"] for tau in THRESHOLDS if t[f"routed_n_{tau}"] > 0),
        "Most sarcasm-type errors are made with confidence of at least 0.7 (Section 6.6)": t[f"errors_routed_{TAU}"] < t["errors"] / 2,
        "Expected calibration error is below 0.03 on both tasks before scaling (Section 6.6)":
            all(macros_routing[k][1]["ece"] < 0.03 for k in ("binary", "type")),
        "Temperature scaling does not lower the mean expected calibration error (Section 6.6)":
            all(macros_routing[k][1]["ece_ts"] >= macros_routing[k][1]["ece"] for k in ("binary", "type")),
        "Raising tau never routes fewer reviews (Section 6.6)":
            all(s_[f"routed_n_{x}"] <= s_[f"routed_n_{y}"] for s_ in (b, t) for x, y in zip(THRESHOLDS, THRESHOLDS[1:])),
    }
    informational = {
        "No comparison among BERT, RoBERTa, DistilBERT and MiniLM survives Holm": holm_top == 0,
        "Some TinyBERT comparison survives Holm on each task": all(
            (tests.loc[involves_tiny & (tests["task"] == tk), "mcnemar_p_holm"] < 0.05).any() for tk in ["binary", "type"]),
        "Binary best Macro F1 within 0.01 of Protocol A (0.9941)": abs(agg[("binary", best["binary"])]["f1"][0] - 0.9941) <= 0.01,
    }
    summary = dict(quick_check=quick, dedup=dedup_stats, macros=M, attn_implementations=attn_used,
                   required_checks={k: bool(v) for k, v in required.items()},
                   informational_checks={k: bool(v) for k, v in informational.items()})
    _atomic_json(summary, os.path.join(res_dir, "summary.json"))

    print("================ CHECKS AGAINST THE MANUSCRIPT TEXT ================")
    if quick:
        print("QUICK CHECK MODE: models were trained on a small sample for one epoch, so CHECK lines")
        print("are expected here and say nothing about the real run.\n")
    print("Sentences that depend on the result (CHECK means the sentence must be reworded):")
    for k, v in required.items():
        print(("  PASS   " if v else "  CHECK  ") + k)
    print("\nFor information only (no sentence depends on these):")
    for k, v in informational.items():
        print(("  yes    " if v else "  no     ") + k)
    return dict(aggregate=agg_df, tests=tests, routing=pd.concat(routing_rows),
                sweep=pd.DataFrame([dict(task=k, model=macros_routing[k][0], **sweep[k]) for k in sweep]),
                baselines=bsum, macros=M, summary=summary)



# --------------------------------------------------------------------------------------
# Command-line entry point
# --------------------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Protocol B re-run for the R3 revision")
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--out_dir", default="protocol_b")
    ap.add_argument("--analysis_only", action="store_true")
    ap.add_argument("--quick_check", action="store_true", help="small sample, one epoch; not for the paper")
    a = ap.parse_args()
    out_dir = a.out_dir.rstrip("/") + ("_quickcheck" if a.quick_check else "")
    os.makedirs(out_dir, exist_ok=True)
    raw, dedup, stats, notes, _ = verify_data(a.data_dir, strict=True)
    print("\n".join(notes))
    if not a.analysis_only:
        record_environment(out_dir)
        if a.quick_check:
            run_grid(dedup, out_dir, hp=dict(epochs=1, max_len=64), subsample=400)
        else:
            run_grid(dedup, out_dir)
    analyse(out_dir, raw, dedup, stats, quick=a.quick_check)


if __name__ == "__main__":
    main()
