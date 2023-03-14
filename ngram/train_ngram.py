import sys
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import word_tokenize, sent_tokenize 
from nltk.lm import KneserNeyInterpolated, Lidstone
from nltk.lm import MLE
import dill as pickle

n = 5

def read_text(index, fname):
    text = []
    print("Reading text")
    with open(fname,"r") as fin:
        for line in fin:
#            cleanline = [tok.split("|")[min(index,len(tok)-1)] for tok in line.lower().strip().split(" ")]
            cleanline = [tok.split("|")[index] for tok in line.lower().strip().split(" ") if "|" in tok]
            text.append(cleanline)
    print(len(text))
    return text

def make_ngram_model(index, outfname):
    text = read_text(index, "../aochildes.tagged.txt")
    model = KneserNeyInterpolated(n, discount = 0.01)
    train, vocab = padded_everygram_pipeline(n, text)
    model.fit(train, vocab)
    with open('../models/5gram_%s.pkl' % outfname, 'wb') as fout:
        pickle.dump(model, fout)
    return
#    test, vocab = padded_everygram_pipeline(n, text)
#    for i, sents in enumerate(test):
#        entropy = model.entropy(sent)
#        print(entropy)

if __name__=="__main__":
    make_ngram_model(0, "word")
    make_ngram_model(1, "pos")
