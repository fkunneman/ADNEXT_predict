
import sys
import pickle

import ucto
import numpy

model = sys.argv[1]
vocab = sys.argv[2]
weights = sys.argv[3]

with open(model, 'rb') as model_open:
    clf = pickle.load(model_open)
tokenizer = ucto.Tokenizer('/vol/customopt/lamachine/etc/ucto/tokconfig-nl-twitter')
vocabulary = {}
keys = []
with open(vocab, 'r', encoding = 'utf-8') as vocabularyfile:
    keys = [x.strip() for x in vocabularyfile.readlines()]
vocabulary_length = len(keys)
vocabulary = {x:i for i, x in enumerate(keys)}

def vectorize(text):
    vector = []
    tokenizer.process(text)
    tokens = [x.text for x in tokenizer]
    for i, token in enumerate(tokens):
        ngrams = tokens + [' '.join(x) for x in zip(tokens, tokens[1:]) ] + [ ' '.join(x) for x in zip(tokens, tokens[1:], tokens[2:])]
    in_vocabulary = [(x, ngrams.count(x)) for x in list(set(ngrams) & set(keys))]
    vector = [0.0] * vocabulary_length
    for ngram in in_vocabulary:
        vector[vocabulary[ngram[0]]] = ngram[1]
    if weights == 'frequency':
        wvector = vector
    elif weights == 'binary':
        wvector = [1 if x > 0 else 0 for x in vector]
    else:
        with open(weights, 'r', encoding = 'utf-8') as fw:
            ws = numpy.array([float(x.strip()) for x in fw.readlines()])    
            wvector = numpy.array(vector) * ws
    return wvector

while True:
    sentence = input('Voer een zin in en krijg een sarcasmescore...\n--> ')
    v = vectorize(sentence)
    prob = round(clf.predict_proba(v)[0][0], 3)
    if prob > 0.8:
        cl = 'Sarcastisch'
    else:
        cl = 'Niet sarcastisch'
    outstr = '\nOordeel: ' + cl + '\nSarcasmescore: ' + str(prob) + '\n'
    print(outstr)

