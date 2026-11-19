"""Pseudo-log-likelihood scoring of the Hu, Wilcox et al. LI sentences with the nine 2023 models.

Same procedure as BabyBERTa.ipynb (2023): mask each subword in turn and sum log P(original subword).
Scores reproduce the 2023 file on all 2,022 sentences the two datasets share (max abs diff .0012).

Runs on CPU only. On Apple-silicon GPUs (MPS) RoBERTa returned wrong values for some sentences, even
unbatched, and BERT did too when batched, so do not move this to MPS.

Usage: python score_pll.py [out.csv]     (resumable: models already complete in out.csv are skipped)
Default output: scores/hu_li_pll_scores.csv (the committed copy is gzipped).
"""
import os
import sys

import pandas as pd
import torch
from transformers import (AutoModelForMaskedLM, AutoTokenizer, BertForMaskedLM, BertTokenizer,
                          RobertaForMaskedLM, RobertaTokenizerFast)

from common import HU_STIMULI, SCORES

MODELS = ["phueb/BabyBERTa-1", "phueb/BabyBERTa-2", "phueb/BabyBERTa-3", "nyu-mll/roberta-base-10M-1",
          "nyu-mll/roberta-base-10M-2", "roberta-base", "roberta-large", "bert-base-cased", "bert-large-cased"]
DEVICE = torch.device("cpu")


def load(name):
    if "BabyBERTa" in name:
        return RobertaForMaskedLM.from_pretrained(name), RobertaTokenizerFast.from_pretrained(name, add_prefix_space=True)
    if "roberta-base-10M-1" in name:
        return AutoModelForMaskedLM.from_pretrained(name), AutoTokenizer.from_pretrained(name)
    if "roberta" in name:
        return RobertaForMaskedLM.from_pretrained(name), RobertaTokenizerFast.from_pretrained(name)
    return BertForMaskedLM.from_pretrained(name), BertTokenizer.from_pretrained(name)


@torch.no_grad()
def pll(sentence, model, tok, batch_size=64):
    ids = tok(sentence, return_tensors="pt")["input_ids"][0]
    n = len(ids) - 2                      # exclude the special start/end tokens
    if n <= 0:
        return float("nan")
    batch = ids.repeat(n, 1)
    pos = torch.arange(1, n + 1)
    batch[torch.arange(n), pos] = tok.mask_token_id
    out = []
    for k in range(0, n, batch_size):
        logp = model(input_ids=batch[k:k + batch_size].to(DEVICE)).logits.float().log_softmax(-1).cpu()
        p = pos[k:k + batch_size]
        out.append(logp[torch.arange(logp.shape[0]), p, ids[p]])
    return float(torch.cat(out).double().sum())


def main(out=SCORES / "hu_li_pll_scores.csv"):
    stim = pd.read_csv(HU_STIMULI)
    sentences = pd.unique(pd.concat([stim.sentence_grammatical, stim.sentence_ungrammatical]).astype(str))
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=["model", "sentence", "pseudoLogProb"])
    for name in MODELS:
        short = name.split("/")[-1]
        if (done.model == short).sum() >= len(sentences):
            continue
        model, tok = load(name)
        model.to(DEVICE).eval()
        rows = [dict(model=short, sentence=s, pseudoLogProb=pll(s, model, tok)) for s in sentences]
        done = pd.concat([done[done.model != short], pd.DataFrame(rows)])
        done.to_csv(out, index=False)
        print("done", short, len(rows), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
