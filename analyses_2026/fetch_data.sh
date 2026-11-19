#!/usr/bin/env bash
# Downloads the third-party data the 2026 analyses use into analyses_2026/data/ (git-ignored).
# Neither source carries a redistribution licence, so the files are fetched rather than committed.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/hu_wilcox_2026 data/sprouse

# 1. Hu, Wilcox, Song, Mahowald & Levy (2026), TACL, https://doi.org/10.1162/TACL.a.611
#    Repository: https://github.com/jennhu/probability-grammaticality (branch master). Files used as released.
HU=https://raw.githubusercontent.com/jennhu/probability-grammaticality/master
get() { mkdir -p "data/hu_wilcox_2026/$(dirname "$1")"; curl -fsSL "$HU/$1" -o "data/hu_wilcox_2026/$1"; echo "  $1"; }
echo "Hu, Wilcox et al. (2026):"
get data/stimuli/li/stimuli.csv
get results/li/gpt2.npz
get results/li/Meta-Llama-3-70B.npz
# Only needed by hu_predictions_ngram.py (their three predictions on all six English datasets)
if [[ "${1:-}" == "--all" ]]; then
  for ds in blimp scamp/plausible scamp/implausible syntaxgym li cola; do
    get "results/$ds/gpt2.npz"; get "results/$ds/Meta-Llama-3-70B.npz"
  done
  for ds in blimp scamp_plausible scamp_implausible syntaxgym li; do
    get "data/cosine_distance/${ds}_all-mpnet-base-v2.pkl"
  done
fi

# 2. Jon Sprouse's official acceptability data (per-participant judgments, with the `condition` column
#    that defines the sentence types). Cited in the 2023 paper.
echo "Sprouse et al.:"
curl -fsSL https://www.jonsprouse.com/data/Lingua2013/SSA.data.zip -o data/sprouse/SSA.data.zip      # Sprouse, Schutze & Almeida (2013), LI
curl -fsSL https://www.jonsprouse.com/data/JoL2012/SA2012.data.zip -o data/sprouse/SA2012.data.zip   # Sprouse & Almeida (2012), Adger
(cd data/sprouse && unzip -q -o SSA.data.zip && unzip -q -o SA2012.data.zip && rm -rf __MACOSX)
echo "  SSA.data/, SA2012.data/"
