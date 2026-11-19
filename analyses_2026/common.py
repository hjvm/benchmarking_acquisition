"""Shared paths and helpers for the 2026 analyses.

All paths are relative to this folder (analyses_2026/), so every script can be
run from anywhere with `python analyses_2026/<script>.py`.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"          # downloaded third-party data (git-ignored), see fetch_data.sh
SCORES = HERE / "scores"      # sentence-level model scores (committed)
RESULTS = HERE / "results"    # analysis outputs (committed)
FIGURES = RESULTS / "figures"

# Third-party data, fetched by fetch_data.sh
HU = DATA / "hu_wilcox_2026"                       # github.com/jennhu/probability-grammaticality
HU_STIMULI = HU / "data/stimuli/li/stimuli.csv"
SPROUSE_LI_ME = DATA / "sprouse/SSA.data/ME experiment/LI.me.results.csv"     # Sprouse, Schutze & Almeida (2013)
SPROUSE_LI_LS = DATA / "sprouse/SSA.data/LS experiment/LI.ls.results.csv"
SPROUSE_ADGER_ME = DATA / "sprouse/SA2012.data/ME.results.csv"                 # Sprouse & Almeida (2012)

# 2023 scores (committed in scores/)
LI_ADGER_2023 = SCORES / "li-adger_2023_mlm_scores.csv.gz"      # = output/LI-Adger/all_models_li-adger_mlm.csv (2023)
LI_ADGER_TRIGRAM = SCORES / "li-adger_bnc_trigram_scores.csv"   # BNC trigram (Sprouse et al. 2018), from bert_mlm_data.csv
# New scores on the Hu, Wilcox et al. sentences (committed in scores/)
HU_PLL = SCORES / "hu_li_pll_scores.csv.gz"                     # score_pll.py
HU_NGRAM = SCORES / "hu_li_5gram_wiki_scores.csv"               # score_ngram.py

MODEL_ORDER = ["BabyBERTa-1", "BabyBERTa-2", "BabyBERTa-3", "roberta-base-10M-1", "roberta-base-10M-2",
               "roberta-base", "roberta-large", "bert-base-cased", "bert-large-cased",
               "5-gram-Wiki-LogProb", "5-gram-Wiki-SLOR"]


def read_sprouse(path, skip=5):
    """Sprouse's result files: latin-1, old Mac line endings, 5 comment lines."""
    text = open(path, encoding="latin-1", newline="").read().replace("\r\n", "\n").replace("\r", "\n")
    return pd.read_csv(io.StringIO(text), skiprows=skip)


def levenshtein(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev = d[:]
        d[0] = i
        for j, y in enumerate(b, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (x != y))
    return d[-1]


def word_edit_distance(s1, s2):
    """Word-level edit distance after stripping a final period or question mark,
    as in Hu, Wilcox et al. (2026), notebooks/analyses.ipynb."""
    s1 = str(s1).rstrip(".").rstrip("?")
    s2 = str(s2).rstrip(".").rstrip("?")
    return levenshtein(s1.split(), s2.split())


def hu_stimuli_with_filter():
    """Hu, Wilcox et al.'s LI stimuli (Sprouse 2013 + Mahowald 2016), with their filter:
    keep pairs whose word edit distance is <= the 75th percentile of the dataset (= 4 words)."""
    st = pd.read_csv(HU_STIMULI)
    st["edit_distance"] = [word_edit_distance(a, b) for a, b in zip(st.sentence_grammatical, st.sentence_ungrammatical)]
    threshold = st.edit_distance.describe()["75%"]
    st["kept"] = st.edit_distance <= threshold
    return st, threshold


def hu_sentence_scores():
    """Sentence x scorer table of raw scores on the Hu, Wilcox et al. LI sentences."""
    pll = pd.read_csv(HU_PLL).pivot_table(index="sentence", columns="model", values="pseudoLogProb")
    ng = pd.read_csv(HU_NGRAM).set_index("sentence")
    pll["5-gram-Wiki-LogProb"] = ng.logprob
    pll["5-gram-Wiki-SLOR"] = ng.slor
    return pll[[c for c in MODEL_ORDER if c in pll.columns]]


def zscore(df):
    return (df - df.mean()) / df.std(ddof=0)


def lower_triangle_heatmap(ax, corr, cmap, title, annot_size=8):
    import seaborn as sns
    import matplotlib.pyplot as plt
    corr = corr.iloc[1:, :-1]                     # drop the diagonal, as in the 2023 figure
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap, vmin=0, vmax=1, cbar=False, square=True,
                linewidths=.5, linecolor="white", annot_kws={"size": annot_size}, ax=ax)
    ax.set_title(title, loc="right", fontsize=12, y=0.82, x=0.98)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=8, length=0)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
