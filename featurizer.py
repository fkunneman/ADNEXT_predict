#!/usr/bin/env 

from collections import Counter
import numpy as np
import operator
from scipy import sparse

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

import utils

class Featurizer:
    """
    Featurizer
    =====
    Class to extract features from raw and frogged text files

    Parameters
    -----
    raw : list
        The raw data comes in an array where each entry represents a text
        instance in the data file.
    frogs : list
        The frog data comes in a list of lists, where each row represents a 
        text instance and the columns represent word - lemma - pos - 
        sentence number respectively
    features : dict
        Subset any of the entries in the following dictionary:

        features = {
            'simple_stats' : {},
            'simple_token_ngrams' : {'n_list' : [1, 2, 3]}
            'token_ngrams' : {'n_list' : [1, 2, 3]},
            'char_ngrams' : {'n_list' : [1, 2, 3]},
            'lemma_ngrams' : {'n_list' : [1, 2, 3]},
            'pos_ngrams' : {'n_list' : [1, 2, 3]}
        }

    Attributes
    -----
    self.frog : list
        The frog parameter
    self.raw : list
        The raw parameter
    self.modules : dict
        Template dict with all the helper classes
    self.helpers : list
        List to call all the helper classes
    self.features : dict
        Container of the featurized instances for the different feature types
    self.vocabularies : dict
        Container of the name of each feature index for the different feature types
    """
    def __init__(self, raws, frogs, features):
        self.frog = frogs
        self.raw = raws
        self.modules = {
            'simple_stats':         SimpleStats,
            'simple_token_ngrams':  SimpleTokenNgrams,
            'token_ngrams':         TokenNgrams,
            'char_ngrams':          CharNgrams,
            'pos_ngrams':           PosNgrams,
        }
        self.helpers = [v(**features[k]) for k, v in self.modules.items() if k in features.keys()]
        self.features = {}
        self.vocabularies = {}

    def fit_transform(self):
        """
        Featurizer
        =====
        Function to extract every helper feature type

        Transforms
        -----
        self.features : dict
            The featurized instances of every helper are written to this dict
        self.vocabularies : dict
            The name of each feature index for every feature type is written to this dict

        """
        for helper in self.helpers:
            feats, vocabulary = helper.fit_transform(self.raw, self.frog)
            self.features[helper.name] = feats
            self.vocabularies[helper.name] = vocabulary

    def return_instances(self, helpernames):
        """
        Information extractor
        =====
        Function to extract featurized instances in any combination of feature types

        Parameters
        ------
        helpernames : list
            List of the feature types to combine
            Names of feature types correspond with the keys of self.modules

        Returns
        -----
        instances : scipy csr matrix
            Featurized instances
        Vocabulary : list
            List with the feature name per index
        """
        submatrices = [features[name] for name in helpernames]
        #instances = sparse.csr_matrix(np.hstack(submatrices))
        instances = np.hstack(submatrices)
        vocabulary = np.hstack([self.vocabularies[name] for name in helpernames])
        return instances, vocabulary

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


class TokenNgrams: # note: to be Colibrized in the future? 
    """
    Calculate token ngram frequencies.
    """
    def __init__(self, **kwargs):
        self.name = 'token_ngrams'
        self.n_list = kwargs['n_list']
        self.blackfeats = kwargs['blackfeats']

    # retrieve indexes of features
    def fit(self, frog_data):
        feats = {}
        for inst in frog_data:
            for n in self.n_list:
                tokens = [t[0] for t in inst]
                feats.update(utils.freq_dict(["_".join(item) for item in \
                    utils.find_ngrams(tokens, n)]))
        self.feats = [i for i, j in sorted(feats.items(), reverse = True, 
            key = operator.itemgetter(1)) if not bool(set(i.split("_")) & 
            set(self.blackfeats))][:self.max_feats]

    def transform(self, frog_data):
        instances = []
        for inst in frog_data:
            tok_dict = {}
            for n in self.n_list:
                tokens = [t[0] for t in inst]
                tok_dict.update(utils.freq_dict(["_".join(item) for item in \
                    utils.find_ngrams(tokens, n)]))
            instances.append([tok_dict.get(f, 0) for f in self.feats])
        return np.array(instances), self.feats

    def fit_transform(self, raw_data, frog_data):
        self.fit(frog_data)
        return self.transform(frog_data)

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
                feats.update(freq_dict(["char-" +  "".join(item) for item in find_ngrams(inst, n)]))
        self.feats = [i for i,j in sorted(feats.items(), reverse=True, 
            key=operator.itemgetter(1))][:max_feats]

    def transform(self, raw_data, frog_data):
        instances = []
        for inst in raw_data:
            inst = list(inst)
            char_dict = {}
            for n in self.n_list:
                char_dict.update(freq_dict(["char-" + "".join(item) for item in find_ngrams(inst, n)]))
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
                feats.update(freq_dict(["pos-" + "_".join(item) for item in find_ngrams(zip(inst)[2], n)]))
        self.feats = [i for i,j in sorted(feats.items(), reverse=True, 
            key=operator.itemgetter(1))][:max_feats]

    def transform(self, raw_data, frog_data):
        instances = []
        for inst in frog_data:
            pos_dict = {}
            for n in self.n_list:
                pos_dict.update(freq_dict(["pos-" + "_".join(item) for item in find_ngrams(zip(inst)[2], n)]))
            instances.append([pos_dict.get(f,0) for f in self.feats])
        return np.array(instances)

    def fit_transform(self, raw_data, frog_data, n_list, max_feats=None):
        self.fit(raw_data, frog_data, n_list, max_feats=max_feats)
        return self.transform(raw_data, frog_data)
