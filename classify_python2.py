
import sys
import os
import codecs
import joblib
from scipy.sparse import *
from scipy import *

import vectorizer
import sklearn_classifier

weight = sys.argv[1]
prune = int(sys.argv[2])
sparse_dirs = sys.argv[3:]

for sd in sparse_dirs:
    with open(sd + 'sparse.txt') as sf:
        data_fields = [line.strip() for line in sf.readlines()]
    data = array([int(x) for x in data_fields[0].split()])
    rows = array([int(x) for x in data_fields[1].split()])
    cols = array([int(x) for x in data_fields[2].split()])
    shape = [int(x) for x in data_fields[3].split()]

    with open(sd + 'labels.txt') as lf:
        labels = [line.strip() for line in lf.readlines()]
        
    with codecs.open(sd + 'vocabulary.txt', 'r', 'utf-8') as vf:
        vocabulary = [line.strip() for line in vf.readlines()]

    instances = csr_matrix((data, (rows, cols)), shape = (shape[0], shape[1]))
    vr = vectorizer.Vectorizer(instances, instances, labels, weight, prune)
    train_vectors, test_vectors, top_features, top_features_values =  vr.vectorize()
    vocabulary_topfeatures = [vocabulary[i] for i in top_features]
    train = {
        'instances' : train_vectors,
        'labels'    : labels
    }
    test = {
        'instances' : test_vectors,
        'labels'    : labels
    }
    print("Performing classification")
    skc = sklearn_classifier.SKlearn_classifier(train, test, {'nb' : {}, 'svm' : {}})
    # skc = sklearn_classifier.SKlearn_classifier(train, test, {'nb' : {}})
    predictions = skc.fit_transform()

    nb = sd + 'nb/'
    if not os.path.isdir(nb):
        os.mkdir(nb)
    model = nb + 'model.joblib.pkl'
    _ = joblib.dump(predictions['nb'][1], model, compress = 9)
    with codecs.open(nb + 'vocabulary.txt', 'w', 'utf-8') as vo:
        vo.write('\n'.join(vocabulary_topfeatures))
    with codecs.open(nb + 'weights.txt', 'w', 'utf-8') as wo:
        wo.write('\n'.join([str(x) for x in top_features_values]))

    svm = sd + 'svm/'
    if not os.path.isdir(svm):
        os.mkdir(svm)
    model = svm + 'model.joblib.pkl'
    _ = joblib.dump(predictions['svm'][1], model, compress=9)
    with codecs.open(svm + 'vocabulary.txt', 'w', 'utf-8') as vo:
        vo.write('\n'.join(vocabulary_topfeatures))
    with codecs.open(svm + 'weights.txt', 'w', 'utf-8') as wo:
        wo.write('\n'.join([str(x) for x in top_features_values]))



