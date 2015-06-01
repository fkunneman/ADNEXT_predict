#!/usr/bin/env 

from collections import Counter
import numpy as np
import operator
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

import utils

class Featurizer:
    """
    Parameters
    -----

    raw : list
        The raw data comes in an array where each entry represents a text
        instance in the data file.

    frogs : list
        The frog data ...

    features : dict
        Subset any of the entries in the following dictionary:

        features = {
            'simple_stats': {}
            'token_ngrams': {'n_list': bla, 'max_feats': bla}
        }

    Notes
    -----
    For an explanation regarding the frog features, please refer either to
    utils.frog.extract_tags or http://ilk.uvt.nl/frog/.
    """
    def __init__(self, raws, frogs, features, blackfeats = []):

        self.frog = frogs
        self.raw = raws
        self.modules = {
            'simple_stats':         SimpleStats,
            'simple_token_ngrams':  SimpleTokenNgrams
            'token_ngrams':         TokenNgrams,
            'char_ngrams':          CharNgrams,
            'pos_ngrams':           PosNgrams,
        }

        self.helpers = [v(**features[k]) for k, v in
                        self.modules.items() if k in features.keys()]

        self.filter = black_feats

        # construct feature_families by combining the given features with
        # their indices, omits the use of an OrderedDict

    def fit_transform(self):
        features = {}
        vocabularies = {}
        for helper in self.helpers:
            feats, vocabulary = helper.fit_transform(self.raw, self.frog)
            features[helper.name] = feats
            vocabularies[helper.name] = vocabulary
        submatrices = [features[ft] for ft in sorted(features.keys())]
        X = np.hstack(submatrices)
        vocab = [vocabularies[v] for v in sorted(vocabularies.keys())]
        return X, vocab

class BlueprintFeature:

    def __init__(self, **kwargs):
        self.name = 'blueprint_feature'
        self.some_option = kwargs['some_option']
        self.some_option = kwargs['some_option2']
        # etc.
        pass

    def fit(self, raw, frog):
        # get feature types
        pass

    def some_function(self, input_vector):
        # do some stuff to input_vector
        pass

    def transform(self, raw, frog):
        instances = []
        for input_vector in raw:
            your_feature_vector = self.some_function(input_vector)
            instances.append(your_feature_vector)
        return instances

    def fit_transform(self, raw_data, frog_data):
        self.fit(raw_data, frog_data)
        return self.transform(raw_data, frog_data)


class SimpleStats:

    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self):
        pass

    def fit_transform(self):
        self.fit()
        self.transform()


class TokenNgrams:
    """
    Calculate token ngram frequencies.
    """
    def __init__(self, **kwargs):
        self.name = 'token_ngrams'
        self.max_feats = kwargs['max_feats']
        self.n_list = kwargs['n_list']

    # retrieve indexes of features
    def fit(self, raw_data, frog_data):
        feats = {}
        for inst in frog_data:
            for n in self.n_list:
                tokens = [t[0] for t in inst]
                feats.update(utils.freq_dict(["_".join(item) for item in \
                    utils.find_ngrams(tokens, n)]))
        self.feats = [i for i, j in sorted(feats.items() \
            if not bool(set(i.split("_")) & set(self.blackfeats)), 
            reverse = True, key = operator.itemgetter(1))][:self.max_feats]

    def transform(self, raw_data, frog_data):
        instances = []
        for inst in frog_data:
            tok_dict = {}
            for n in self.n_list:
                tokens = [t[0] for t in inst]
                tok_dict.update(utils.freq_dict(["_".join(item) for item in \
                    utils.find_ngrams(tokens, n)]))
            instances.append([tok_dict.get(f, 0) for f in self.feats])
        return np.array(instances)

    def fit_transform(self, raw_data, frog_data):
        self.fit(raw_data, frog_data)
        return self.transform(raw_data, frog_data)

class SimpleTokenNgrams:
    """
    Calculate token ngram frequencies after simple tokenization.
    """
    def __init__(self, **kwargs):
        self.name = 'token_ngrams'
        self.max_feats = kwargs['max_feats']
        self.n_list = kwargs['n_list']

    # retrieve indices of features
    def fit(self, raw_data, frog_data):
        feats = {}
        for inst in raw_data:
            for n in self.n_list:
                tokens = inst.split(" ")
                feats.update(utils.freq_dict(["_".join(item) for item in \
                    utils.find_ngrams(tokens, n)]))
        self.feats = [i for i,j in sorted(feats.items(), reverse=True, \
            key=operator.itemgetter(1))][:self.max_feats]
        print(self.feats)

    def transform(self, raw_data, frog_data):
        instances = []
        for inst in frog_data:
            tok_dict = {}
            for n in self.n_list:
                tokens = inst.split(" ")
                tok_dict.update(utils.freq_dict(["_".join(item) for item in \
                    utils.find_ngrams(tokens, n)]))
            instances.append([tok_dict.get(f,0) for f in self.feats])
        return np.array(instances)

    def fit_transform(self, raw_data, frog_data):
        self.fit(raw_data, frog_data)
        return self.transform(raw_data, frog_data)


class CharNgrams:
    """
    Computes frequencies of char ngrams
    """
    def __init__(self):
        self.feats = None
        self.name = 'char_ngrams'

    def fit(self, raw_data, frog_data, n_list, max_feats=None):
        self.n_list = n_list
        feats = {}
        for inst in raw_data:
            inst = list(inst)
            for n in self.n_list:
                feats.update(freq_dict(["char-"+"".join(item) for item in find_ngrams(inst, n)]))
        self.feats = [i for i,j in sorted(feats.items(), reverse=True, key=operator.itemgetter(1))][:max_feats]

    def transform(self, raw_data, frog_data):
        if self.feats == None:
            raise ValueError('There are no features to transform the data with. You probably did not "fit" before "transforming".')
        instances = []
        for inst in raw_data:
            inst = list(inst)
            char_dict = {}
            for n in self.n_list:
                char_dict.update(freq_dict(["char-"+"".join(item) for item in find_ngrams(inst, n)]))
            instances.append([char_dict.get(f,0) for f in self.feats])
        return np.array(instances)

    def fit_transform(self, raw_data, frog_data, n_list, max_feats=None):
        self.fit(raw_data, frog_data, n_list, max_feats=max_feats)
        return self.transform(raw_data, frog_data)


class PosNgrams:
    """
    """
    def __init__(self):
        self.feats = None
        self.name = 'pos_ngrams'

    def fit(self, raw_data, frog_data, n_list, max_feats=None):
        self.n_list = n_list
        feats = {}
        for inst in frog_data:
            for n in self.n_list:
                feats.update(freq_dict(["pos-"+"_".join(item) for item in find_ngrams(zip(inst)[2], n)]))
        self.feats = [i for i,j in sorted(feats.items(), reverse=True, key=operator.itemgetter(1))][:max_feats]

    def transform(self, raw_data, frog_data):
        if self.feats == None:
            raise ValueError('There are no features to transform the data with. You probably did not "fit" before "transforming".')
        instances = []
        for inst in frog_data:
            pos_dict = {}
            for n in self.n_list:
                pos_dict.update(freq_dict(["pos-"+"_".join(item) for item in find_ngrams(zip(inst)[2], n)]))
            instances.append([pos_dict.get(f,0) for f in self.feats])
        return np.array(instances)

    def fit_transform(self, raw_data, frog_data, n_list, max_feats=None):
        self.fit(raw_data, frog_data, n_list, max_feats=max_feats)
        return self.transform(raw_data, frog_data)
