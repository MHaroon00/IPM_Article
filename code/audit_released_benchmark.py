# -*- coding: utf-8 -*-
"""
Audit of the released benchmark files used for the R3 revision. Runs on CPU in about a minute.

Produces, in --out_dir:
  dedup_summary.csv               rows, unique texts, redundant rows, conflicting texts per task
  conflict_pairs.csv              which sarcasm types co-occur on the same text
  crossfile_audit.csv             label of each sarcasm-type review in the binary file
  type_column_agreement.csv       agreement between the type file and the type column of the binary file
  class_cue_profile.csv           share of reviews with "worst" and with developer-directed requests
  label_check_sample.csv          30 random reviews from each of the two disputed categories
  original_split_overlap.csv      train/test text overlap in the released-benchmark partitions
  paired_tests_with_holm.csv      the 60 paired tests of the manuscript with Holm-adjusted p-values
  table_paired_tests_summary.csv  the per-pair summary reported in the paired-test table

Usage:
  python audit_released_benchmark.py --data_dir . --tests paired_bootstrap_mcnemar.csv --out_dir audit
"""
import argparse, os, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def norm_key(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\x00-\x7f]", "", str(s))).strip().lower()


def holm(p):
    p = np.asarray(p, dtype=float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--tests", default="paired_bootstrap_mcnemar.csv")
    ap.add_argument("--out_dir", default="audit")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)

    braw = pd.read_csv(os.path.join(a.data_dir, "annotated_comments_binary_1.csv"))
    mraw = pd.read_csv(os.path.join(a.data_dir, "Sarcasm_multiclassification_1.csv"))
    b = braw[["comment_Text", "Have_Sarcasm"]].rename(columns={"comment_Text": "text", "Have_Sarcasm": "label"})
    m = mraw[["comment_Text", "Sarcasm_Type"]].rename(columns={"comment_Text": "text", "Sarcasm_Type": "label"})
    b["label"] = b["label"].str.strip().str.capitalize(); m["label"] = m["label"].str.strip()
    for d in (b, m):
        d["key"] = d["text"].map(norm_key)

    # deduplication summary
    rows = []
    for task, d in [("binary", b), ("type", m)]:
        n = d.groupby("key")["label"].nunique(); conflict = set(n[n > 1].index)
        kept = d[~d["key"].isin(conflict)].drop_duplicates("key")
        rows.append(dict(task=task, rows=len(d), unique_texts=d["key"].nunique(),
                         redundant_rows=len(d) - d["key"].nunique(),
                         exact_rows_in_duplicate_groups=int(d["text"].duplicated(keep=False).sum()),
                         conflicting_texts=len(conflict), rows_in_conflicting=int(d["key"].isin(conflict).sum()),
                         kept=len(kept), kept_class_counts=kept["label"].value_counts().to_dict()))
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "dedup_summary.csv"), index=False)
    g = m.groupby("key")["label"].agg(lambda s: " + ".join(sorted(set(s))))
    g[g.str.contains(r"\+")].value_counts().rename("texts").to_csv(os.path.join(a.out_dir, "conflict_pairs.csv"))

    # cross-file consistency and label mapping evidence
    bmap = b.drop_duplicates("key").set_index("key")["label"]
    m["label_in_binary_file"] = m["key"].map(bmap).fillna("not present")
    pd.crosstab(m["label"], m["label_in_binary_file"], margins=True).to_csv(os.path.join(a.out_dir, "crossfile_audit.csv"))
    braw["key"] = braw["comment_Text"].map(norm_key)
    tmap = braw[braw["Have_Sarcasm"].str.strip().str.capitalize() == "Yes"].drop_duplicates("key").set_index("key")["Sarcasm_Type"]
    pd.crosstab(m["label"], m["key"].map(tmap).fillna("none")).to_csv(os.path.join(a.out_dir, "type_column_agreement.csv"))
    cue = pd.DataFrame({
        "label": m["label"],
        "contains_worst": m["text"].str.lower().str.contains("worst"),
        "developer_directed": m["text"].str.lower().str.contains(r"\b(?:dev|devs|developer|developers|you guys|fix|please)\b"),
    })
    cue.groupby("label").mean().round(3).to_csv(os.path.join(a.out_dir, "class_cue_profile.csv"))
    pd.concat([m[m["label"] == l].sample(30, random_state=7)[["label", "text"]]
               for l in ["Self-Deprecation", "Passive-Aggression"]]).to_csv(os.path.join(a.out_dir, "label_check_sample.csv"), index=False)

    # overlap in the released-benchmark partitions
    rows = []
    for task, d in [("binary", b), ("type", m)]:
        for seed in [42, 123, 2024]:
            tr, tmp = train_test_split(d, test_size=0.3, stratify=d["label"], random_state=seed)
            va, te = train_test_split(tmp, test_size=0.5, stratify=tmp["label"], random_state=seed)
            trlab = tr.groupby("key")["label"].agg(set)
            hit = te[te["key"].isin(trlab.index)]
            same = int(sum(r.label in trlab[r.key] for r in hit.itertuples()))
            rows.append(dict(task=task, seed=seed, train=len(tr), val=len(va), test=len(te),
                             exact_text_overlap=int(te["text"].isin(set(tr["text"])).sum()),
                             normalised_overlap=len(hit), same_label_in_train=same,
                             only_different_label_in_train=len(hit) - same))
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "original_split_overlap.csv"), index=False)

    # Holm correction of the 60 paired tests
    if os.path.exists(a.tests):
        t = pd.read_csv(a.tests)
        t["mcnemar_p_holm"] = holm(t["mcnemar_p_value"])
        t["pair"] = t["model_a"] + " vs. " + t["model_b"]
        t.to_csv(os.path.join(a.out_dir, "paired_tests_with_holm.csv"), index=False)
        s = t.groupby(["task", "pair"], sort=False).agg(
            abs_mean_delta=("delta_macro_f1", lambda x: abs(x.mean())),
            mcnemar_raw=("mcnemar_p_value", lambda x: int((x < 0.05).sum())),
            mcnemar_holm=("mcnemar_p_holm", lambda x: int((x < 0.05).sum())))
        s["ci_contains_zero"] = t.assign(c=(t["bootstrap_ci95_low"] <= 0) & (t["bootstrap_ci95_high"] >= 0)).groupby(["task", "pair"], sort=False)["c"].sum().values
        s.round(4).to_csv(os.path.join(a.out_dir, "table_paired_tests_summary.csv"))
    print("Audit written to", a.out_dir)


if __name__ == "__main__":
    main()
