"""Runs the three predictions of Hu, Wilcox et al. (2026) (their notebooks/analyses.ipynb) with a Wikipedia
5-gram next to their released GPT-2 and Llama-3-70B results, on all six of their English datasets.

  P1a  correlation of acceptable and unacceptable log-probabilities within pairs (BLiMP, SCaMP, SyntaxGym)
  P1b  the same in the closest and farthest decile of within-pair cosine distance
  P2a  correlation of the model's within-pair difference with the human Likert difference (filtered LI)
  P2b  the same in the closest and farthest cosine-distance decile
  P3   AUC separating pooled acceptable from unacceptable strings, per dataset and pooled
  plus minimal-pair accuracy for each scoring measure.

Inputs: `bash fetch_data.sh --all` (their result files and cosine distances), and either the KenLM model
(first run, about 14.6 GB, see score_ngram.py) or the cached 5-gram scores in
scores/hu_all_5gram_wiki_scores.csv.gz written by that first run. Without either, only GPT-2 and Llama-3 run.

Output: results/hu_predictions_ngram.csv (model, test, dataset, value)

Status: the committed output was produced on Nov 18 2026 by the original version of this script, whose
5-gram score cache was not kept. This port has been checked on the GPT-2 and Llama-3 rows only.
"""
import math
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

from common import HU, RESULTS, SCORES, word_edit_distance

LN10 = math.log(10)
DS = {"blimp": "blimp", "scamp_plausible": "scamp/plausible", "scamp_implausible": "scamp/implausible",
      "syntaxgym": "syntaxgym", "li": "li", "cola": "cola"}
MINIMAL_PAIR = ["blimp", "scamp_plausible", "scamp_implausible", "syntaxgym"]
METRICS = ["sum_logprob", "mean_logprob", "slor", "bf_l1"]
NGRAM_CACHE = SCORES / "hu_all_5gram_wiki_scores.csv.gz"
MODELS = {"5-gram (Wikipedia)": None, "GPT-2": "gpt2", "Llama-3-70B": "Meta-Llama-3-70B"}


def npz(ds, model):
    return dict(np.load(HU / f"results/{DS[ds]}/{model}.npz", allow_pickle=True))


def load_released(ds, model):
    d = npz(ds, model)
    if ds == "cola":
        for side in ("good", "bad"):
            d[side + "_bf_l1"] = d[side + "_slor"].astype(float) * (d[side + "_sum_logprob"].astype(float) / d[side + "_mean_logprob"].astype(float))
        return d
    df = pd.DataFrame({k: v for k, v in d.items() if v.ndim == 1})
    for side in ("good", "bad"):
        df[side + "_bf_l1"] = df[side + "_slor"] * (df[side + "_sum_logprob"] / df[side + "_mean_logprob"])
    return df


def ngram_scores(model_path=None):
    """sentence -> (sum_logprob, mean_logprob, slor, bf_l1), natural log, with an end-of-sentence token."""
    if NGRAM_CACHE.exists():
        c = pd.read_csv(NGRAM_CACHE).set_index("sentence")
        return {s: tuple(r) for s, r in zip(c.index, c[METRICS].values)}
    if model_path is None:
        return None
    import kenlm
    import spacy
    m = kenlm.Model(model_path)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"])
    sentences = set()
    for ds in DS:
        d = npz(ds, "gpt2")
        sentences |= set(map(str, d["good_sentence"])) | set(map(str, d["bad_sentence"]))
    out = {}
    for s in sentences:
        toks = [t.text for t in nlp.make_doc(s.strip())]
        n = len(toks) + 1
        lp = m.score(" ".join(toks), bos=True, eos=True) * LN10
        uni = (sum(m.score(t, bos=False, eos=False) for t in toks) + m.score("</s>", bos=False, eos=False)) * LN10
        out[s] = (lp, lp / n, (lp - uni) / n, lp - uni)
    pd.DataFrame([(s, *v) for s, v in out.items()], columns=["sentence"] + METRICS).to_csv(NGRAM_CACHE, index=False)
    return out


def ngram_df(ds, sc):
    d = npz(ds, "gpt2")
    out = {}
    for side in ("good", "bad"):
        rows = [sc[str(s)] for s in d[side + "_sentence"]]
        for j, name in enumerate(METRICS):
            out[side + "_" + name] = [r[j] for r in rows]
    if ds == "cola":
        return {k: np.array(v) for k, v in out.items()}
    df = pd.DataFrame(out)
    df["good_sentence"], df["bad_sentence"] = d["good_sentence"], d["bad_sentence"]
    if ds == "li":
        df["LS_grammatical"] = d["LS_grammatical"].astype(float)
        df["LS_ungrammatical"] = d["LS_ungrammatical"].astype(float)
    return df


def cosine(ds):
    c = pd.DataFrame(pickle.load(open(HU / f"data/cosine_distance/{ds}_all-mpnet-base-v2.pkl", "rb")))
    return c.drop_duplicates(["good_sentence", "bad_sentence"])


def by_decile(df, x, y):
    df = df.merge(cosine_cache[df.attrs["ds"]], on=["good_sentence", "bad_sentence"], how="left").dropna(subset=["cosine_distance"])
    df["bin"] = pd.qcut(df.cosine_distance, 10, labels=False, duplicates="drop")
    r = df.groupby("bin").apply(lambda g: stats.pearsonr(g[x], g[y])[0])
    return r.iloc[0], r.iloc[-1]


cosine_cache = {}


def main(model_path=None):
    sc = ngram_scores(model_path)
    models = {k: v for k, v in MODELS.items() if v is not None or sc is not None}
    for ds in MINIMAL_PAIR + ["li"]:
        cosine_cache[ds] = cosine(ds)
    rows = []

    def get(ds, name):
        return ngram_df(ds, sc) if MODELS[name] is None else load_released(ds, MODELS[name])

    for name in models:
        pooled = {m: ([], []) for m in METRICS}
        for ds in MINIMAL_PAIR:
            df = get(ds, name)
            df = df[df.good_sentence != df.bad_sentence]
            df.attrs["ds"] = ds
            rows.append(dict(model=name, test="P1a r(gram, ungram logprob)", dataset=ds,
                             value=stats.pearsonr(df.good_sum_logprob, df.bad_sum_logprob)[0]))
            near, far = by_decile(df, "good_sum_logprob", "bad_sum_logprob")
            rows += [dict(model=name, test="P1b r in closest decile", dataset=ds, value=near),
                     dict(model=name, test="P1b r in farthest decile", dataset=ds, value=far)]
            for m in METRICS:
                rows.append(dict(model=name, test="minimal-pair accuracy, " + m, dataset=ds,
                                 value=float((df["good_" + m] > df["bad_" + m]).mean())))
        # Prediction 2 on the filtered LI set
        df = get("li", name)
        df = df[df.good_sentence != df.bad_sentence].copy()
        df["LS_diff"] = df.LS_grammatical.astype(float) - df.LS_ungrammatical.astype(float)
        df["logprob_diff"] = df.good_sum_logprob - df.bad_sum_logprob
        df["ed"] = [word_edit_distance(a, b) for a, b in zip(df.good_sentence, df.bad_sentence)]
        f = df[df.ed <= df.ed.describe()["75%"]].drop_duplicates(["good_sentence", "bad_sentence"])
        f.attrs["ds"] = "li"
        rows.append(dict(model=name, test="P2a r(model delta, human delta)", dataset=f"li (filtered, n={len(f)})",
                         value=stats.pearsonr(f.LS_diff, f.logprob_diff)[0]))
        rows.append(dict(model=name, test="minimal-pair accuracy, sum_logprob", dataset="li (filtered)",
                         value=float((f.logprob_diff > 0).mean())))
        near, far = by_decile(f, "LS_diff", "logprob_diff")
        rows += [dict(model=name, test="P2b r in closest decile", dataset="li", value=near),
                 dict(model=name, test="P2b r in farthest decile", dataset="li", value=far)]
        # Prediction 3: AUC per dataset and pooled
        for ds in ["li", "cola"] + MINIMAL_PAIR:
            d = get(ds, name)
            for m in METRICS:
                g = np.array(d["good_" + m], dtype=float)
                b = np.array(d["bad_" + m], dtype=float)
                g, b = g[np.isfinite(g)], b[np.isfinite(b)]
                auc = metrics.roc_auc_score([1] * len(g) + [0] * len(b), np.concatenate([g, b]))
                rows.append(dict(model=name, test="P3 AUC, " + m, dataset=ds, value=auc))
                pooled[m][0].extend(g)
                pooled[m][1].extend(b)
        for m in METRICS:
            g, b = pooled[m]
            rows.append(dict(model=name, test="P3 AUC, " + m, dataset="POOLED",
                             value=metrics.roc_auc_score([1] * len(g) + [0] * len(b), np.array(g + b))))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / ("hu_predictions_ngram.csv" if sc is not None else "hu_predictions_released_only.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(out.pivot_table(index=["test", "dataset"], columns="model", values="value", sort=False).round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
