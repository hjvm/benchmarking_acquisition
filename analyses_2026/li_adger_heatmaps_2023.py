"""Rebuilds the two LI-Adger correlation heatmaps of the 2023 GenBench paper (Figure: mean and standard
deviation of judgments per sentence type, correlated between humans and models).

The code that made the published figure was lost before it was pushed. This is a reconstruction from the
paper's description, checked against the published figure (figures/ in the 2023 working copy):

  1. Every scorer (human or model) is z-scored over all sentences.
  2. Sentences are grouped into sentence types using the `condition` column of Sprouse's official
     result files (8 tokens per type), for LI (Sprouse, Schutze & Almeida 2013) and Adger (Sprouse & Almeida 2012).
  3. For each type, the mean and the standard deviation of the z-scores are taken.
  4. Pearson correlations of those per-type means (left) and per-type SDs (right) across types.

Human scores are the mean per-item z-score from Sprouse's per-participant magnitude-estimation files.

Check against the published figure (human row): the neural models reproduce within about .01 for the means
and about .02 for the SDs. The two BNC trigram entries do NOT reproduce (means: .10 and .30 here against
.15 and .47 published), although the trigram's correlations with the models do (e.g. trigram log-prob vs
Baby-CHI .885 here, .88 published). The cause is unknown; see README.md.

Usage: python li_adger_heatmaps_2023.py      (needs data/sprouse from fetch_data.sh)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import (FIGURES, LI_ADGER_2023, LI_ADGER_TRIGRAM, RESULTS, SPROUSE_ADGER_ME, SPROUSE_LI_ME,
                    lower_triangle_heatmap, read_sprouse, zscore)

LABELS = {  # 2023 figure labels
    "human": "human", "BabyBERTa_AO-CHILDES": "Baby-CHI", "BabyBERTa_AO-Newsela": "Baby-News",
    "BabyBERTa_Wikipedia-1": "Baby-Wiki", "BabyBERTa_AO-CHILDES+AO-Newselsa+Wikipedia-1": "Baby-CHI+News+Wiki",
    "roberta-base-10M-1": "RoBERTa-10M-1", "roberta-base-10M-2": "RoBERTa-10M-2", "roberta-base": "RoBERTa-base",
    "roberta-large": "RoBERTa-large", "bert-base-cased": "BERT-base-cased", "bert-large-cased": "BERT-large-cased",
    "BNClogProb": "Trigram-BNC-LogProb", "BNCslor": "Trigram-BNC-SLOR"}
# Published human-row values, for the automatic check printed at the end
PUBLISHED_MEAN = {"Baby-CHI": .12, "Baby-News": .22, "Baby-Wiki": .18, "Baby-CHI+News+Wiki": .31, "RoBERTa-10M-1": .37,
                  "RoBERTa-10M-2": .39, "RoBERTa-base": .68, "RoBERTa-large": .69, "BERT-base-cased": .62,
                  "BERT-large-cased": .64, "Trigram-BNC-LogProb": .15, "Trigram-BNC-SLOR": .47}
PUBLISHED_SD = {"Baby-CHI": .10, "Baby-News": .093, "Baby-Wiki": .083, "Baby-CHI+News+Wiki": .081, "RoBERTa-10M-1": .047,
                "RoBERTa-10M-2": .07, "RoBERTa-base": .10, "RoBERTa-large": .15, "BERT-base-cased": .16,
                "BERT-large-cased": .15, "Trigram-BNC-LogProb": .033, "Trigram-BNC-SLOR": .17}


def main():
    official = pd.concat([read_sprouse(SPROUSE_LI_ME)[["item", "condition", "zscores"]],
                          read_sprouse(SPROUSE_ADGER_ME)[["item", "condition", "zscores"]]])
    condition = official.drop_duplicates("item").set_index("item").condition
    human = official.groupby("item").zscores.mean()

    d = pd.read_csv(LI_ADGER_2023)
    long = pd.concat([d[["model", "sentence_good_id", "pseudoLogProb_good"]].set_axis(["model", "id", "s"], axis=1),
                      d[["model", "sentence_bad_id", "pseudoLogProb_bad"]].set_axis(["model", "id", "s"], axis=1)])
    wide = long.drop_duplicates(["model", "id"]).pivot(index="id", columns="model", values="s")
    tri = pd.read_csv(LI_ADGER_TRIGRAM).set_index("id")
    wide["BNClogProb"], wide["BNCslor"] = tri.BNClogProb, tri.BNCslor
    wide["human"] = human
    wide = wide[list(LABELS)].dropna()
    wide = wide[wide.index.isin(condition.index)]

    z = zscore(wide)
    g = z.groupby(z.index.map(condition))
    mean, sd = g.mean(), g.std(ddof=1)
    mean, sd = mean.rename(columns=LABELS), sd.rename(columns=LABELS)
    print(f"{len(wide)} sentences in {len(mean)} sentence types")

    mc, sc = mean.corr(), sd.corr()
    mc.to_csv(RESULTS / "li_adger_2023_type_mean_corr.csv")
    sc.to_csv(RESULTS / "li_adger_2023_type_sd_corr.csv")
    # Mean within-type SD (the paper's variability measure), overall and per source (Adger item IDs start with "ch")
    source = pd.Series(["Adger" if str(t).startswith("ch") else "LI" for t in sd.index], index=sd.index)
    variability = pd.DataFrame({"all": sd.mean(), "LI": sd[source == "LI"].mean(), "Adger": sd[source == "Adger"].mean()})
    # The paper's human figure (.288 for LI) is on Sprouse's own per-participant z-score scale, i.e. the item
    # means are NOT re-z-scored. Reported here as an extra row: LI .292, Adger .277.
    raw = wide.human.groupby(wide.index.map(condition)).std(ddof=1)
    variability.loc["human (Sprouse z-scale, not re-z-scored)"] = [raw.mean(), raw[source == "LI"].mean(), raw[source == "Adger"].mean()]
    variability.round(3).to_csv(RESULTS / "li_adger_2023_within_type_sd.csv")

    check = pd.DataFrame({"mean_rebuilt": mc.human.drop("human"), "mean_published": pd.Series(PUBLISHED_MEAN),
                          "sd_rebuilt": sc.human.drop("human"), "sd_published": pd.Series(PUBLISHED_SD)}).round(3)
    check.to_csv(RESULTS / "li_adger_2023_check_against_published.csv")
    print("Human row, rebuilt vs published 2023 figure:\n", check.to_string())
    print("Mean within-type SD:\n", variability.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    lower_triangle_heatmap(axes[0], mc, sns.light_palette("#c0392b", as_cmap=True),
                           "LI-Adger: Mean Human and Model\nJudgment Correlation Matrix (rebuilt)")
    lower_triangle_heatmap(axes[1], sc, sns.light_palette("#2e6f8e", as_cmap=True),
                           "LI-Adger: Std. Dev. Human and Model\nJudgment Correlation Matrix (rebuilt)")
    fig.tight_layout()
    fig.savefig(FIGURES / "li_adger_2023_heatmaps_rebuilt.png", dpi=200)


if __name__ == "__main__":
    main()
