# 2026 analyses: recovered 2023 heatmaps and a rerun on the Hu, Wilcox et al. (2026) data

This folder adds two things to the code of *Evaluating Neural Language Models as Cognitive Models of Language
Acquisition* ([GenBench 2023](https://aclanthology.org/2023.genbench-1.4)):

1. **The LI-Adger correlation heatmaps, rebuilt.** The code that produced the paper's two heatmaps (per-sentence-type
   mean and standard deviation of judgments, correlated between humans and models) was lost before it was pushed.
   `li_adger_heatmaps_2023.py` rebuilds it from the paper's description and checks it against the published figure.
2. **A rerun of the 2023 analyses on the data of Hu, Wilcox, Song, Mahowald & Levy (2026, TACL,
   [doi:10.1162/TACL.a.611](https://doi.org/10.1162/TACL.a.611))**, which criticises the 2023 LI pairs as not tightly
   controlled. We use their LI data exactly as released, with their own pair filter, and add a Wikipedia 5-gram.
   We also run their three predictions with the 5-gram next to their GPT-2 and Llama-3-70B.

The same rerun, scored with the 2021 Acceptability Delta Criterion, lives in the
[acceptability_delta_criterion](https://github.com/hjvm/acceptability_delta_criterion) repository.

## Running it

```bash
pip install -r analyses_2026/requirements.txt
bash analyses_2026/fetch_data.sh          # add --all for hu_predictions_ngram.py (about 60 MB)
cd analyses_2026
python li_adger_heatmaps_2023.py          # 1. rebuilt 2023 heatmaps + check against the published figure
python hu_filter_2023_scores.py           # 2a. 2023 scores, unchanged, on the pairs their filter keeps
python rerun_2023_on_hu.py                # 2b. accuracy, delta correlation, variability and heatmaps on their data
python plot_hu_heatmaps.py kept
python hu_predictions_ngram.py            # 2c. their three predictions (needs fetch_data.sh --all; see below)
```

The sentence-level scores are committed in `scores/`, so steps 1 to 2b run in seconds without any model.
To rescore from scratch:

* `score_pll.py`: pseudo-log-likelihood with the nine 2023 models from the Hugging Face hub. **CPU only.** On
  Apple-silicon GPUs (MPS) RoBERTa gave wrong values for some sentences even unbatched, and BERT did when batched.
  On CPU the scores match the 2023 file on all 2,022 sentences the two datasets share (max abs diff .0012).
* `score_ngram.py`: the public Wikipedia 5-gram KenLM model
  ([BramVanroy/kenlm_wikipedia_en](https://huggingface.co/BramVanroy/kenlm_wikipedia_en), `wiki_en_token.arpa.bin`,
  about 14.6 GB), tokenised with spaCy `en_core_web_sm`.

## Files

| File | What it is |
|---|---|
| `fetch_data.sh` | Downloads Hu, Wilcox et al.'s released data and Sprouse's official result files into `data/` (git-ignored; neither has a redistribution licence) |
| `common.py` | Paths, their edit-distance filter, score loaders, the heatmap style |
| `scores/li-adger_2023_mlm_scores.csv.gz` | The 2023 model scores on LI-Adger (`output/LI-Adger/all_models_li-adger_mlm.csv` of the 2023 working copy) |
| `scores/li-adger_bnc_trigram_scores.csv` | BNC trigram log-probability and SLOR on LI-Adger (Sprouse et al. 2018), extracted from the 2023 `bert_mlm_data.csv` |
| `scores/hu_li_pll_scores.csv.gz` | Nine models x 4,514 sentences of the Hu, Wilcox et al. LI set (`score_pll.py`) |
| `scores/hu_li_5gram_wiki_scores.csv` | Wikipedia 5-gram on the same sentences (`score_ngram.py`) |
| `results/` | Every table below as CSV, and the figures in `results/figures/` |

## 1. The 2023 heatmaps, rebuilt

Procedure (`li_adger_heatmaps_2023.py`): z-score every scorer over all sentences; group sentences into sentence types
with the `condition` column of Sprouse's official files (LI: Sprouse, Schütze & Almeida 2013; Adger: Sprouse & Almeida 2012;
8 tokens per type); take each type's mean and SD; correlate those across types. Human scores are each item's mean
z-score in Sprouse's per-participant magnitude-estimation files. 4,024 sentences in 503 types.

Human row, rebuilt against the published figure (`results/li_adger_2023_check_against_published.csv`):

| | Means, rebuilt | Means, published | SDs, rebuilt | SDs, published |
|---|---|---|---|---|
| Baby-CHI | .114 | .12 | .109 | .10 |
| Baby-News | .206 | .22 | .090 | .093 |
| Baby-Wiki | .166 | .18 | .089 | .083 |
| Baby-CHI+News+Wiki | .298 | .31 | .093 | .081 |
| RoBERTa-10M-1 | .357 | .37 | .062 | .047 |
| RoBERTa-10M-2 | .375 | .39 | .088 | .070 |
| RoBERTa-base | .680 | .68 | .111 | .10 |
| RoBERTa-large | .687 | .69 | .152 | .15 |
| BERT-base-cased | .610 | .62 | .151 | .16 |
| BERT-large-cased | .636 | .64 | .165 | .15 |
| Trigram-BNC-LogProb | **.095** | **.15** | .014 | .033 |
| Trigram-BNC-SLOR | **.295** | **.47** | .167 | .17 |

* **Neural models reproduce** within .015 for the means and about .02 for the SDs.
* **The two trigram entries do not**, and we have not found why. The trigram's correlations with the models do
  reproduce for log-probability (vs Baby-CHI: .88 rebuilt and published), but SLOR's do not (vs Baby-CHI: −.02 rebuilt,
  .064 published), so the published figure probably used a different SLOR, or trigram scores for a different
  sentence set, than the `BNCslor` column in `bert_mlm_data.csv`. Tried without success: z-scoring within each dataset,
  no z-scoring, Spearman, the COCA trigram, and the human column from `bert_mlm_data.csv` instead of Sprouse's files.
* RoBERTa-10M-3 is in the published figure but not in the 2023 score file, so it is not rebuilt.
* **Human variability.** The paper reports a mean within-type SD of .288 for humans. That is on Sprouse's own
  z-score scale, without re-z-scoring the item means: here .288 over LI-Adger (.293 LI, .283 Adger). After
  re-z-scoring, as for the models, it is .371. Models: .44 to .56, trigram log-probability .31
  (`results/li_adger_2023_within_type_sd.csv`).

## 2. Rerun on the Hu, Wilcox et al. (2026) data

**Their data.** Their "LI" set is not LI-Adger: it is 1,095 pairs from Sprouse et al. (2013) plus 1,175 from
Mahowald et al. (2016), with z-scored Likert ratings. Their filter keeps pairs whose word edit distance (after
stripping a final period or question mark) is at most the dataset's 75th percentile, which is 4 words: 1,884 pairs,
936 Sprouse and 948 Mahowald. They describe the result as "reasonably meaning-matched".

**2a. The 2023 scores on the pairs they keep** (`hu_filter_2023_scores.py`). 1,093 of our 1,200 LI pairs are in their
file and 942 pass their filter. Accuracy under the BLiMP criterion:

| | All LI-Adger (2,393) | LI (1,200) | LI kept by their filter (942) |
|---|---|---|---|
| Trigram, SLOR | .682 | .645 | .638 |
| BabyBERTa-1 to 3 | .643–.656 | .608–.623 | .588–.599 |
| BERT-base | .873 | .842 | .852 |
| RoBERTa-base | .891 | .863 | .872 |

The 2023 claim survives: on the pairs they keep, the trigram is still on par with BabyBERTa (a few points above it).

**2b. The 2023 analyses on their full data** (`rerun_2023_on_hu.py`), filtered set of 1,884 pairs:

| Scorer | Accuracy | r(model Δ, human Δ) | Within-condition SD |
|---|---|---|---|
| BabyBERTa-1 to 3 | .603–.617 | .069–.080 | .50 |
| 5-gram, log-probability | .679 | .202 | .32 |
| 5-gram, SLOR | .727 | .351 | .52 |
| RoBERTa-10M-1, -2 | .720, .732 | .280, .291 | .55–.57 |
| BERT-base, -large | .848, .850 | .456, .474 | .51–.53 |
| RoBERTa-base, -large | .856, .868 | .471, .466 | .58 |
| Humans | | | .36 |
| *Their GPT-2, Llama-3-70B* | *.785, .845* | *.344, .399* | |

Variability and heatmaps use 402 conditions with at least 4 sentences (17 dropped). Sprouse conditions come from his
official `condition` column; Mahowald conditions are the released ID minus its item number. Human correlation of
condition means: BabyBERTa .13–.14, RoBERTa-10M .36–.37, BERT .49–.51, RoBERTa-base/large .62–.63, 5-gram .23 and .26.
BabyBERTa correlates with the 5-gram's log-probability at .82, as the 2023 paper found for the trigram.
Figure: `results/figures/hu_rerun_kept_heatmaps.png`.

In short, the three 2023 findings hold on their data: an n-gram is on par with (here above) the acquisition-scale
models; models vary more than humans within a condition; and models agree with each other more than with humans.
As in 2023, the large models correlate with humans better than the small ones.

**2c. Their three predictions with a 5-gram** (`hu_predictions_ngram.py`):

| Test | 5-gram | GPT-2 | Llama-3-70B |
|---|---|---|---|
| P1a: r of acceptable vs unacceptable log-prob, BLiMP | .980 | .982 | .982 |
| P2a: r of model Δ vs human Δ, filtered LI | .202 | .344 | .399 |
| P3: pooled AUC, SLOR | .544 | .627 | .618 |
| Minimal-pair accuracy, BLiMP | .601 | .825 | .818 |
| Minimal-pair accuracy, SyntaxGym | .497 | .725 | .777 |
| Minimal-pair accuracy, filtered LI | .680 | .785 | .845 |

A 5-gram reproduces Prediction 1 and Prediction 3, and Prediction 2 in weaker form, so those three confirmations
do not by themselves require grammatical knowledge. It does not match their models on minimal-pair accuracy.
The GPT-2 and Llama-3 rows reproduce their released results exactly. The committed 5-gram rows come from the
original Nov 18 2026 run; its 5-gram score cache was not kept, so rerunning them needs the KenLM model.

## Caveats

* The n-gram here is a Wikipedia 5-gram, not the BNC trigram of the 2021 and 2023 papers, which covers only the
  Sprouse sentences.
* Mahowald ratings are released as means with no rater counts, so the human within-condition SD for that half
  may be inflated by rating noise.
* Their human scores are Likert ratings; LI-Adger used magnitude estimation.
* The five BabyBERTa variants trained on AO-CHILDES, Newsela and Wikipedia were not available locally, so 2b has
  BabyBERTa-1 to 3 only.
