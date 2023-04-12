import sys

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import word_tokenize, sent_tokenize 
from nltk.lm import KneserNeyInterpolated, Lidstone
from nltk.lm import MLE
import dill as pickle
from sklearn.pipeline import Pipeline
import sklearn_crfsuite
from sklearn_crfsuite import metrics

from train_ngram import read_text


def read_taggedtext(fname):
    words = read_text(0, fname)
    tags = read_text(1, fname)
    return [[(word, tag) for word, tag in zip(sentwords, senttags)] for sentwords, senttags in zip(words, tags)]

def get_features(sent, i):
    return {
        'word': sent[i],
        'is_first': i == 0,
        'is_last': i == len(sent) - 1,
        'is_capitalized': sent[i][0].upper() == sent[i][0],
        'is_all_caps': sent[i].upper() == sent[i],
        'is_all_lower': sent[i].lower() == sent[i],
        'prefix-1': sent[i][0],
        'prefix-2': sent[i][:2],
        'prefix-3': sent[i][:3],
        'suffix-1': sent[i][-1],
        'suffix-2': sent[i][-2:],
        'suffix-3': sent[i][-3:],
        'prev_word': '' if i == 0 else sent[i - 1],
        'next_word': '' if i == len(sent) - 1 else sent[i + 1],
        'has_hyphen': '-' in sent[i]
    }

def split(tagged_text):
    threshold = int(0.8*len(tagged_text))
    train = tagged_text[:threshold]
    test = tagged_text[threshold:]
    return train, test

def get_Xy(tagged_text):
    X = [[get_features([w for w, t in sent], i) for i in range(len(sent))] for sent in tagged_text]
    y = [[postag for token, postag in sent] for sent in tagged_text]
    return X,y
    for tagged_sent in tagged_text:
        sent = [w for w, t in tagged_sent]
        for i, pair in enumerate(tagged_sent):
            word, tag = pair
#            print(i, len(sent), word, tag, sent)
            X.append(get_features(sent, i))
            y.append(tag)
    return X, y

tagged_text = read_taggedtext("../aochildes.tagged.txt")

train, test = split(tagged_text)

trainX, trainy = get_Xy(train)
testX, testy = get_Xy(test)
allX, ally = get_Xy(tagged_text)

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(allX, ally)
predy = crf.predict(allX)
print(metrics.flat_accuracy_score(ally, predy))
#crf.fit(trainX, trainy)
#predy = crf.predict(testX)
#print(metrics.flat_accuracy_score(testy, predy))

with open('../models/tagger_crf.pkl', 'wb') as fout:
    pickle.dump(crf, fout)
