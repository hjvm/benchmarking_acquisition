"""Lower-triangle heatmaps (condition means, condition SDs) for the rerun on the Hu, Wilcox et al. data,
in the style of the 2023 figure. Run rerun_2023_on_hu.py first.

Usage: python plot_hu_heatmaps.py [subset]     subset = kept (default) | all | kept_sprouse | kept_mahowald
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import FIGURES, RESULTS, lower_triangle_heatmap

LABELS = {"Human": "human", "BabyBERTa-1": "BabyBERTa-1", "BabyBERTa-2": "BabyBERTa-2", "BabyBERTa-3": "BabyBERTa-3",
          "roberta-base-10M-1": "RoBERTa-10M-1", "roberta-base-10M-2": "RoBERTa-10M-2", "roberta-base": "RoBERTa-base",
          "roberta-large": "RoBERTa-large", "bert-base-cased": "BERT-base-cased", "bert-large-cased": "BERT-large-cased",
          "5-gram-Wiki-LogProb": "5-gram-Wiki-LogProb", "5-gram-Wiki-SLOR": "5-gram-Wiki-SLOR"}


def main(subset="kept"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax, kind, cmap, title in ((axes[0], "mean", sns.light_palette("#c0392b", as_cmap=True), "Mean Human and Model\nJudgment Correlation Matrix"),
                                  (axes[1], "sd", sns.light_palette("#2e6f8e", as_cmap=True), "Std. Dev. Human and Model\nJudgment Correlation Matrix")):
        c = pd.read_csv(RESULTS / f"hu_rerun_{subset}_{kind}_corr.csv", index_col=0)
        order = [k for k in LABELS if k in c.index]
        c = c.loc[order, order].rename(index=LABELS, columns=LABELS)
        lower_triangle_heatmap(ax, c, cmap, title)
    fig.suptitle(f"Hu, Wilcox et al. (2026) LI data, subset: {subset}", fontsize=12, y=0.99)
    fig.tight_layout()
    out = FIGURES / f"hu_rerun_{subset}_heatmaps.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
