
import sys
import os
import codecs
import joblib
from scipy import sparse

import vectorizer
import sklearn_classifier

weight = sys.argv[1]
prune = sys.argv[2]
sparse_dirs = sys.argv[3:]

for sd in sparse_dirs:
    with open(sd + 'sparse.txt') as sf:
        data_fields = sf.readlines()
    data = data_fields[0].split()
    rows = data_fields[1].split()
    cols = data_fields[2].split()
    shape = data_fields[3].split()

    with open(sd + 'labels.txt') as lf:
        labels = lf.readlines()

    instances = sparse.csr_matrix((data, (rows, cols)), shape = (shape[0], shape[1]))
    vr = vectorizer.Vectorizer(instances, instances, labels, weight, prune)
    train_vectors, test_vectors, top_features, top_features_values =  vr.vectorize()
    vocabulary_topfeatures = [vocabulary[i] for i in top_features]
    train = {
        'instances' : train_vectors,
        'labels'    : trainlabels
    }
    test = {
        'instances' : test_vectors,
        'labels'    : testlabels
    }
    print("Performing classification")
    skc = sklearn_classifier.SKlearn_classifier(train, test, {'nb' : {}, 'svm' : {}})
    predictions = skc.fit_transform()

    nb = sd + 'nb/'
    if not os.path.isdir(nb):
        os.mkdir(nb)
    model = nb + 'model.joblib.pkl'
    _ = joblib.dump(, model, compress=9)
    with codecs.open(nb + 'vocabulary.txt', 'w', 'utf-8') as vo:
        vo.write('\n'.join(predictions['features']))
    with codecs.open(nb + 'weights.txt', 'w', 'utf-8') as wo:
        wo.write('\n'.join(predictions['feature_weights']))

    svm = sd + 'svm/'
    if not os.path.isdir(svm):
        os.mkdir(svm)
    model = svm + 'model.joblib.pkl'
    _ = joblib.dump(, model, compress=9)
    with codecs.open(svm + 'vocabulary.txt', 'w', 'utf-8') as vo:
        vo.write('\n'.join(predictions['features']))
    with codecs.open(svm + 'weights.txt', 'w', 'utf-8') as wo:
        wo.write('\n'.join(predictions['feature_weights']))



