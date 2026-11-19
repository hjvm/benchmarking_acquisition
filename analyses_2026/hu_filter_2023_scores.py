"""How does the Hu, Wilcox et al. (2026) pair filter affect the 2023 accuracy results?

Hu, Wilcox et al. use Sprouse et al. (2013) LI pairs (plus Mahowald et al. 2016, not in our 2023 data) and keep
pairs whose word edit distance is <= 4 (their 75th percentile). This script uses the 2023 scores as they are
(no re-scoring) and recomputes BLiMP-criterion accuracy on the LI pairs their filter keeps and drops.

Output: results/hu_filter_2023_accuracy.csv, results/hu_filter_2023_pairs.csv
"""
import pandas as pd

from common import LI_ADGER_2023, LI_ADGER_TRIGRAM, RESULTS, hu_stimuli_with_filter, word_edit_distance


def main():
    hu, threshold = hu_stimuli_with_filter()
    sprouse = hu[hu.phenomenon == "LI-Sprouse"]
    in_hu = set(zip(sprouse["Good ID"], sprouse["Bad ID"]))

    d = pd.read_csv(LI_ADGER_2023)
    tri = pd.read_csv(LI_ADGER_TRIGRAM).set_index("id")
    pairs = d.drop_duplicates(["sentence_good_id", "sentence_bad_id"])[
        ["dataset", "sentence_good_id", "sentence_bad_id", "sentence_good", "sentence_bad"]].copy()
    pairs["edit_distance"] = [word_edit_distance(a, b) for a, b in zip(pairs.sentence_good, pairs.sentence_bad)]
    pairs["in_hu"] = [(a, b) in in_hu for a, b in zip(pairs.sentence_good_id, pairs.sentence_bad_id)]
    pairs["kept_by_hu"] = pairs.in_hu & (pairs.edit_distance <= threshold)
    print(f"threshold {threshold}; our pairs {len(pairs)} {pairs.dataset.value_counts().to_dict()}; "
          f"LI pairs in their file {int(pairs.in_hu.sum())}, kept {int(pairs.kept_by_hu.sum())}")

    # 2023 neural scores plus the BNC trigram (SLOR), one row per pair x scorer
    d = d.merge(pairs[["sentence_good_id", "sentence_bad_id", "edit_distance", "kept_by_hu"]],
                on=["sentence_good_id", "sentence_bad_id"])
    d["correct"] = d.pseudoLogProb_good > d.pseudoLogProb_bad
    t = pairs.copy()
    t["model"] = "Trigram-BNC-SLOR"
    t["correct"] = t.sentence_good_id.map(tri.BNCslor).values > t.sentence_bad_id.map(tri.BNCslor).values
    t = t[t.sentence_good_id.isin(tri.index) & t.sentence_bad_id.isin(tri.index)]
    d = pd.concat([d, t[d.columns.intersection(t.columns)]])

    subsets = {"all LI-Adger": d.index == d.index, "LI (all)": d.dataset == "LI",
               "LI kept by Hu et al.": d.kept_by_hu, "LI dropped by Hu et al.": (d.dataset == "LI") & ~d.kept_by_hu,
               "Adger": d.dataset == "Adger", "Adger, edit distance <= 4": (d.dataset == "Adger") & (d.edit_distance <= threshold)}
    out = pd.DataFrame({k: d[m].groupby("model").correct.mean() for k, m in subsets.items()}).round(3)
    n = {k: d[m & (d.model == "BabyBERTa-1")].shape[0] for k, m in subsets.items()}
    print("pairs per subset:", n)
    print(out.to_string())
    out.to_csv(RESULTS / "hu_filter_2023_accuracy.csv")
    pairs.to_csv(RESULTS / "hu_filter_2023_pairs.csv", index=False)


if __name__ == "__main__":
    main()
