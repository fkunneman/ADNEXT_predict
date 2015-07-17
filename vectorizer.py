#!/usr/bin/env 

from collections import Counter
from scipy import sparse

class Vectorizer:
    """
    Vectorizer
    =====
    Class to transform featurized train and test instances into weighted and 
    pruned vectors as input for SKlearn classification

    Parameters
    -----
    train : list
        list of featurized train instances, as list with feature frequencies
    test : list 
        list of featurized test instances, as list with feature frequencies
    weight : str
        names of weighting to perform
        options : 'frequency', 'binary', 'tfidf'
        default : 'frequency'
    prune : int
        top N of pruning, all features not in the top N features with the highest weight in 
        training are pruned
        default : 5000
    """
    def __init__(self, train, test, weight = 'frequency', prune = 5000):
        self.train = train
        self.test = test
        self.weight = weight
        self.prune = prune

    def count_features(self):
        feature_counts = Counter()
        for instance in self.train:
            feature_counts.update([i for i, v in enumerate(instance) if v > 0]) #document count
        return feature_counts

    def prune_instances(self, indices):
        self.train = [[instance[index] for index in indices] for instance in self.train]
        self.test = [[instance[index] for index in indices] for instance in self.test] 

    def prune_features(self):
        feature_counts = self.count_features()
        selected_features = [x[0] for x in feature_counts.most_common()[:self.prune]]
        self.prune_instances(selected_features)

    def weight_features(self):
        #if weight == 'tfidf':
        #    idf = self.return_idf()
        if self.weight == 'binary':
            self.train = [[1 if x > 0 else 0 for x in instance] for instance in self.train]
            self.test = [[1 if x > 0 else 0 for x in instance] for instance in self.test]

    def vectorize(self):
        self.weight_features()
        self.prune_features()
        return sparse.csr_matrix(self.train), self.test

    # def return_idf(self):
    #     num_docs = len(self.train)
    #     df = defaultdict(int)
    #     idf = {}
    #     for instance in train:
    #         features = instance["sparse"]
    #         for feature in features.keys():
    #             df[feature] += 1

    #     for feature in df.keys():
    #         idf[feature] = math.log((num_docs/df[feature]),10)

    #     return idf
