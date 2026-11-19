"""Scores the Hu, Wilcox et al. LI sentences with a public 5-gram KenLM model trained on English Wikipedia.

Model: huggingface.co/BramVanroy/kenlm_wikipedia_en, file wiki_en_token.arpa.bin (about 14.6 GB), with
spaCy en_core_web_sm tokenisation as the model card recommends. This is NOT the BNC trigram of the 2021 and
2023 papers (Sprouse et al. 2018), which covers only the Sprouse sentences.

Outputs natural-log probability and SLOR = (log P(s) - sum of unigram log P(w)) / tokens.

Usage: python score_ngram.py /path/to/wiki_en_token.arpa.bin
Requires: pip install https://github.com/kpu/kenlm/archive/master.zip spacy; python -m spacy download en_core_web_sm
"""
import math
import sys

import kenlm
import pandas as pd
import spacy

from common import HU_NGRAM, HU_STIMULI

LN10 = math.log(10)


def main(model_path):
    m = kenlm.Model(model_path)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"])
    stim = pd.read_csv(HU_STIMULI)
    sentences = pd.unique(pd.concat([stim.sentence_grammatical, stim.sentence_ungrammatical]).astype(str))
    rows = []
    for s in sentences:
        toks = [t.text for t in nlp(s.strip())]
        lp = m.score(" ".join(toks), bos=True, eos=True) * LN10
        uni = sum(m.score(t, bos=False, eos=False) for t in toks) * LN10
        rows.append(dict(sentence=s, ntok=len(toks), oov=sum(t not in m for t in toks), logprob=lp, unigram=uni,
                         slor=(lp - uni) / len(toks)))
    pd.DataFrame(rows).to_csv(HU_NGRAM, index=False)
    print(len(rows), "sentences; OOV tokens", sum(r["oov"] for r in rows))


if __name__ == "__main__":
    main(sys.argv[1])
