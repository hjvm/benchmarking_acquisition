"""Reruns the three 2023 analyses on the LI data released by Hu, Wilcox et al. (2026).

Data: their stimuli.csv as released (Sprouse et al. 2013 + Mahowald et al. 2016, 2,270 pairs), with their
filter (word edit distance <= 4, their 75th percentile): 1,884 pairs, 936 Sprouse and 948 Mahowald.
Human scores: their z-scored Likert ratings (LS_* columns).
Models: the nine 2023 models available locally (pseudo-log-likelihood, scores/hu_li_pll_scores.csv.gz) and a
public Wikipedia 5-gram (log-probability and SLOR, scores/hu_li_5gram_wiki_scores.csv).

Analyses
  1. Accuracy under the BLiMP criterion (acceptable sentence scores higher).
  2. Their Prediction 2 with our models: correlation of the model's within-pair difference with the human
     Likert difference (the 2021 ADC correlation; see the acceptability_delta_criterion repository).
  3. The 2023 variability and heatmap analysis: z-score every scorer, group sentences into conditions,
     take each condition's mean and SD, correlate across conditions; conditions with < 4 sentences dropped.
     Sprouse conditions come from his official `condition` column (LS experiment); Mahowald conditions are
     the ID minus its item number, as released by Hu, Wilcox et al.

Outputs in results/: hu_rerun_accuracy.csv, hu_rerun_delta_corr.csv, and for each subset
(all, kept, kept_sprouse, kept_mahowald) hu_rerun_<subset>_{mean_corr,sd_corr,variability}.csv
"""
import numpy as np
import pandas as pd

from common import RESULTS, SPROUSE_LI_LS, hu_sentence_scores, hu_stimuli_with_filter, read_sprouse, zscore

pd.set_option("display.width", 250)


def side_scores(scores, df):
    good = scores.reindex(df.sentence_grammatical.astype(str)).reset_index(drop=True)
    bad = scores.reindex(df.sentence_ungrammatical.astype(str)).reset_index(drop=True)
    return good, bad


def accuracy(scores, df):
    good, bad = side_scores(scores, df)
    return (good > bad).mean()


def delta_corr(scores, df):
    good, bad = side_scores(scores, df)
    human = df.LS_diff.reset_index(drop=True)
    return (good - bad).apply(lambda c: np.corrcoef(c, human)[0, 1])


def sentence_table(df, official_condition):
    a = df[["Good ID", "sentence_grammatical", "LS_grammatical"]].set_axis(["id", "sentence", "human"], axis=1)
    b = df[["Bad ID", "sentence_ungrammatical", "LS_ungrammatical"]].set_axis(["id", "sentence", "human"], axis=1)
    t = pd.concat([a, b]).drop_duplicates("id")
    t["condition"] = t.id.map(official_condition).fillna(t.id.str.rsplit(".", n=1).str[0])
    return t


def condition_stats(scores, t, min_n=4):
    x = scores.reindex(t.sentence.astype(str)).reset_index(drop=True)
    x["Human"] = t.human.values
    z = zscore(x)
    z["condition"] = t.condition.values
    g = z.groupby("condition")
    n = g.size()
    keep = n[n >= min_n].index
    return g.mean().loc[keep], g.std(ddof=1).loc[keep], n


def main():
    st, threshold = hu_stimuli_with_filter()
    kept = st[st.kept].drop_duplicates(["sentence_grammatical", "sentence_ungrammatical"])
    print(f"pairs {len(st)}, threshold {threshold}, kept {len(kept)} {kept.phenomenon.value_counts().to_dict()}")
    scores = hu_sentence_scores()

    acc = pd.DataFrame({
        f"all pairs (n={len(st)})": accuracy(scores, st),
        f"kept by their filter (n={len(kept)})": accuracy(scores, kept),
        f"kept, Sprouse (n={(kept.phenomenon == 'LI-Sprouse').sum()})": accuracy(scores, kept[kept.phenomenon == "LI-Sprouse"]),
        f"kept, Mahowald (n={(kept.phenomenon == 'LI-Mahowald').sum()})": accuracy(scores, kept[kept.phenomenon == "LI-Mahowald"]),
        f"dropped (n={(~st.kept).sum()})": accuracy(scores, st[~st.kept])}).round(3)
    print("\nACCURACY (BLiMP criterion)\n", acc.to_string())
    acc.to_csv(RESULTS / "hu_rerun_accuracy.csv")

    dc = pd.DataFrame({"all pairs": delta_corr(scores, st), "kept by their filter": delta_corr(scores, kept)}).round(3)
    print("\nDELTA CORRELATION with the human Likert difference\n", dc.to_string())
    dc.to_csv(RESULTS / "hu_rerun_delta_corr.csv")

    official = read_sprouse(SPROUSE_LI_LS)[["item", "condition"]].drop_duplicates().set_index("item").condition
    subsets = {"all": st, "kept": kept, "kept_sprouse": kept[kept.phenomenon == "LI-Sprouse"],
               "kept_mahowald": kept[kept.phenomenon == "LI-Mahowald"]}
    for name, df in subsets.items():
        mean, sd, n = condition_stats(scores, sentence_table(df, official))
        print(f"\n== {name}: conditions {len(n)}, used (>= 4 sentences) {len(mean)}, dropped {int((n < 4).sum())}")
        print(pd.DataFrame({"within-condition SD": sd.mean(), "r(means) with Human": mean.corr().Human,
                            "r(SDs) with Human": sd.corr().Human}).round(3).to_string())
        mean.corr().to_csv(RESULTS / f"hu_rerun_{name}_mean_corr.csv")
        sd.corr().to_csv(RESULTS / f"hu_rerun_{name}_sd_corr.csv")
        sd.mean().rename("mean_within_condition_sd").to_csv(RESULTS / f"hu_rerun_{name}_variability.csv")


if __name__ == "__main__":
    main()
