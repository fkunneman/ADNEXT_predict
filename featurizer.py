#!/usr/bin/env 

import numpy as np
import operator
import os

import colibricore

import utils

class Featurizer:
    """
    Featurizer
    =====
    Class to extract features from raw and tagged text files

    Parameters
    -----
    raw : list
        The raw data comes in an array where each entry represents a text
        instance in the data file.
    tagged : list
        The tagged data comes in a list of lists, where each row represents a 
        text instance and the columns represent word - lemma - pos - 
        sentence number respectively
    features : dict
        Subset any of the entries in the following dictionary:

        features = {
            'simple_stats' : {},
            'token_ngrams' : {'n_list' : [1, 2, 3]},
            'char_ngrams' : {'n_list' : [1, 2, 3]},
            'lemma_ngrams' : {'n_list' : [1, 2, 3]},
            'pos_ngrams' : {'n_list' : [1, 2, 3]}
        }

    Attributes
    -----
    self.tagged : list
        The tagged parameter
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
    def __init__(self, raws, tagged, features):
        self.tagged = tagged
        self.raw = raws
        self.modules = {
            'simple_stats':         SimpleStats,
            'token_ngrams':         TokenNgrams,
            'lemma_ngrams':         LemmaNgrams,
            'pos_ngrams':           PosNgrams,
            'char_ngrams':          CharNgrams,
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
            feats, vocabulary = helper.fit_transform(self.raw, self.tagged)
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
        submatrices = [self.features[name] for name in helpernames]
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

class CocoNgrams:

    def __init__(self, tmpdir):
        self.tmpdir = tmpdir
        self.vocabulary = []
        self.classencoder = False
        self.classdecoder = False
        self.model = False
        
    def fit(self, lines, minimum, max_ngrams):
        ngram_file = self.tmpdir + 'ngrams.txt'
        with open(ngram_file, 'w', encoding = 'utf-8') as txt:
            for line in lines:
                txt.write(line)
        classfile = self.tmpdir + 'ngrams.colibri.cls'
        # Build class encoder
        self.classencoder = colibricore.ClassEncoder()
        self.classencoder.build(ngram_file)
        self.classencoder.save(classfile)

        # Encode corpus data
        corpusfile = self.tmpdir + 'ngrams.colibri.dat'
        self.classencoder.encodefile(ngram_file, corpusfile)

        # Load class decoder
        self.classdecoder = colibricore.classdecoder(classfile) 

        # Train model
        options = colibricore.PatternModelOptions(mintokens = minimum, maxlength = max_ngrams, doreverseindex=True)
        self.model = colibricore.UnindexedPatternModel()
        self.model.train(corpusfile, options)

        # Extract vocabulary
        for pattern, count in sorted(self.model.items(), key = lambda x : x[1], reverse = True):
            self.vocabulary.append('_'.join(pattern.tostring(self.classdecoder)), count)
        print(self.vocabulary)

    def transform(self, ngrams):
        for line in self.model.
        pass

        # for line in lines:
        #     ngrams = []
        #     pattern = self.classencoder.build(line)
        #     for n in ngrams:
                
    #         for ngram in surface:
    #             if ngram:
    #                 pattern = classencoder.buildpattern(ngram)
    #                 if pattern.unknown():
    #                     print("WARNING: Anchor has unknown part " +  ngram + ", skipping... (" + pattern.tostring(classdecoder) + ")",file=sys.stderr)
    #                 else:
    #                     if len(pattern) <= 5:
    #                         anchormodel.add(pattern) #(will count +1  if already exists)

class TokenNgrams(CocoNgrams): 
    """
    Token ngram extractor
    =====
    Class to extract token ngrams from all documents

    Parameters
    -----
    kwargs : dict
        n_list : list
            The values of N (1 - ...)
        blackfeats : list
            Features to exclude

    Attributes
    -----
    self.name : str
        The name of the module
    self.n_list : list
        The n_list parameter
    self.blackfeats : list
        The blackfeats parameter
    self.feats : list
        List of feature names, to keep track of the values of feature indices
    """
    def __init__(self, **kwargs):
        self.name = 'token_ngrams'
        self.n_list = [int(x) for x in kwargs['n_list']]
        self.tmpdir = kwargs['tmp']
        if 'blackfeats' in kwargs.keys():
            self.blackfeats = kwargs['blackfeats']
        else:
            self.blackfeats = []
        self.feats = []        
        CocoNgrams.__init__(self, self.tmpdir)

    # retrieve indices of features
    def fit(self, tagged_data):
        """
        Model fitter
        =====
        Function to make an overview of all the existing features

        Parameters
        -----
        tagged_data : list
            List of lists, where each row represents a text instance and the columns 
            represent word - lemma - pos - sentence number respectively

        Attributes
        -----
        feats : dict
            dictionary of features and their count
        """        
        tokenized = [' '.join([t[0] for t in instance]) + '\n' for instance in tagged_data]
        CocoNgrams.fit(self, tokenized, 3, max(self.n_list))
        # feats = {}
        # for inst in tagged_data:
        #     for n in self.n_list:
        #         tokens = [t[0] for t in inst]
        #         feats.update(utils.freq_dict(["_".join(item) for item in \
        #             utils.find_ngrams(tokens, n)]))
        # self.feats = [i for i, j in sorted(feats.items(), reverse = True, 
        #     key = operator.itemgetter(1)) if not bool(set(i.split("_")) & 
        #     set(self.blackfeats))]

    def transform(self, tagged_data):
        """
        Model transformer
        =====
        Function to featurize instances based on the fitted features 

        Parameters
        -----
        tagged_data : list
            List of lists, where each row represents a text instance and the columns 
            represent word - lemma - pos - sentence number respectively

        Attributes
        -----
        instances : list
            The documents represented as feature vectors
        """       
        instances = []
        for inst in tagged_data:
            tok_dict = {}
            for n in self.n_list:
                tokens = [t[0] for t in inst]
                tok_dict.update(utils.freq_dict(["_".join(item) for item in \
                    utils.find_ngrams(tokens, n)]))
            instances.append([tok_dict.get(f, 0) for f in self.feats])
        return np.array(instances)

    def fit_transform(self, raw_data, tagged_data):
        """
        Fit transform
        =====
        Function to perform the fit and transform sequence

        Parameters
        -----
        raw_data : list
            Each entry represents a text instance in the data file
        tagged_data : list
            List of lists, where each row represents a text instance and the columns 
            represent word - lemma - pos - sentence number respectively

        Returns
        -----
        self.transform(tagged_data) : list
            The featurized instances
        self.feats : list
            The vocabulary
        """  
        self.fit(tagged_data)
        return self.transform(tagged_data), self.feats

class LemmaNgrams:
    """
    Lemma ngram extractor
    =====
    Class to extract Lemma ngrams from all documents

    Parameters
    -----
    kwargs : dict
        n_list : list
            The values of N (1 - ...)
        blackfeats : list
            Features to exclude

    Attributes
    -----
    self.name : str
        The name of the module
    self.n_list : list
        The n_list parameter
    self.blackfeats : list
        The blackfeats parameter
    self.feats : list
        List of feature names, to keep track of the values of feature indices
    """
    def __init__(self, **kwargs):
        self.name = 'lemma_ngrams'
        self.n_list = kwargs['n_list']
        if 'blackfeats' in kwargs.keys():
            self.blackfeats = kwargs['blackfeats']
        else:
            self.blackfeats = []
        self.feats = []

    def fit(self, tagged_data):
        """
        Model fitter
        =====
        Function to make an overview of all the existing features

        Parameters
        -----
        tagged_data : list
            List of lists, where each row represents a text instance and the columns 
            represent word - lemma - pos - sentence number respectively

        Attributes
        -----
        feats : dict
            dictionary of features and their count
        """
        feats = {}
        for inst in tagged_data:
            for n in self.n_list:
                tokens = [t[1] for t in inst]
                feats.update(utils.freq_dict(["_".join(item) for item in \
                    utils.find_ngrams(tokens, n)]))
        self.feats = [i for i, j in sorted(feats.items(), reverse = True, 
            key = operator.itemgetter(1)) if not bool(set(i.split("_")) & 
            set(self.blackfeats))]

    def transform(self, tagged_data):
        """
        Model transformer
        =====
        Function to featurize instances based on the fitted features 

        Parameters
        -----
        tagged_data : list
            List of lists, where each row represents a text instance and the columns 
            represent word - lemma - pos - sentence number respectively

        Attributes
        -----
        instances : list
            The documents represented as feature vectors
        """       
        instances = []
        for inst in tagged_data:
            tok_dict = {}
            for n in self.n_list:
                tokens = [t[1] for t in inst]
                tok_dict.update(utils.freq_dict(["_".join(item) for item in \
                    utils.find_ngrams(tokens, n)]))
            instances.append([tok_dict.get(f, 0) for f in self.feats])
        return np.array(instances)

    def fit_transform(self, raw_data, tagged_data):
        """
        Fit transform
        =====
        Function to perform the fit and transform sequence

        Parameters
        -----
        raw_data : list
            Each entry represents a text instance in the data file
        tagged_data : list
            List of lists, where each row represents a text instance and the columns 
            represent word - lemma - pos - sentence number respectively

        Returns
        -----
        self.transform(tagged_data) : list
            The featurized instances
        self.feats : list
            The vocabulary
        """  
        self.fit(tagged_data)
        return self.transform(tagged_data), self.feats

class PosNgrams:
    """
    Part-of-Speech tag ngram extractor
    =====
    Class to extract PoS ngrams from all documents

    Parameters
    -----
    kwargs : dict
        n_list : list
            The values of N (1 - ...)
        blackfeats : list
            Features to exclude

    Attributes
    -----
    self.name : str
        The name of the module
    self.n_list : list
        The n_list parameter
    self.blackfeats : list
        The blackfeats parameter
    self.feats : list
        List of feature names, to keep track of the values of feature indices
    """
    def __init__(self, **kwargs):
        self.name = 'pos_ngrams'
        self.n_list = [int(x) for x in kwargs['n_list']]
        if 'blackfeats' in kwargs.keys():
            self.blackfeats = kwargs['blackfeats']
        else:
            self.blackfeats = []
        self.feats = []

    def fit(self, tagged_data):
        """
        Model fitter
        =====
        Function to make an overview of all the existing features

        Parameters
        -----
        tagged_data : list
            List of lists, where each row represents a text instance and the columns 
            represent word - lemma - pos - sentence number respectively

        Attributes
        -----
        feats : dict
            dictionary of features and their count
        """
        feats = {}
        for inst in tagged_data:
            for n in self.n_list:
                tokens = [t[2] for t in inst]
                feats.update(utils.freq_dict(["_".join(item) for item in \
                    utils.find_ngrams(tokens, n)]))
        self.feats = [i for i, j in sorted(feats.items(), reverse = True, 
            key = operator.itemgetter(1)) if not bool(set(i.split("_")) & 
            set(self.blackfeats))]

    def transform(self, tagged_data):
        """
        Model transformer
        =====
        Function to featurize instances based on the fitted features 

        Parameters
        -----
        tagged_data : list
            List of lists, where each row represents a text instance and the columns 
            represent word - lemma - pos - sentence number respectively

        Attributes
        -----
        instances : list
            The documents represented as feature vectors
        """       
        instances = []
        for inst in tagged_data:
            tok_dict = {}
            for n in self.n_list:
                tokens = [t[2] for t in inst]
                tok_dict.update(utils.freq_dict(["_".join(item) for item in \
                    utils.find_ngrams(tokens, n)]))
            instances.append([tok_dict.get(f, 0) for f in self.feats])
        return np.array(instances)

    def fit_transform(self, raw_data, tagged_data):
        """
        Fit transform
        =====
        Function to perform the fit and transform sequence

        Parameters
        -----
        raw_data : list
            Each entry represents a text instance in the data file
        tagged_data : list
            List of lists, where each row represents a text instance and the columns 
            represent word - lemma - pos - sentence number respectively

        Returns
        -----
        self.transform(tagged_data) : list
            The featurized instances
        self.feats : list
            The vocabulary
        """  
        self.fit(tagged_data)
        return self.transform(tagged_data), self.feats

class CharNgrams:
    """
    Character ngram extractor
    =====
    Class to extract character ngrams from all documents

    Parameters
    -----
    kwargs : dict
        n_list : list
            The values of N (1 - ...)
        blackfeats : list
            Features to exclude

    Attributes
    -----
    self.name : str
        The name of the module
    self.n_list : list
        The n_list parameter
    self.blackfeats : list
        The blackfeats parameter
    self.feats : list
        List of feature names, to keep track of the values of feature indices
    """
    def __init__(self, **kwargs):
        self.name = 'char_ngrams'
        self.n_list = [int(x) for x in kwargs['n_list']]
        if 'blackfeats' in kwargs.keys():
            self.blackfeats = kwargs['blackfeats']
        else:
            self.blackfeats = []
        self.feats = []

    def fit(self, raw_data):
        """
        Model fitter
        =====
        Function to make an overview of all the existing features

        Parameters
        -----
        raw_data : list
            Each entry represents a text instance in the data file

        Attributes
        -----
        feats : dict
            dictionary of features and their count
        """       
        feats = {}
        for inst in raw_data:
            inst = list(inst)
            for n in self.n_list:
                feats.update(utils.freq_dict(["".join(item) for item in utils.find_ngrams(inst, n)]))
        self.feats = [i for i,j in sorted(feats.items(), reverse=True, 
            key=operator.itemgetter(1)) if not bool(set(i.split("_")) & 
            set(self.blackfeats))]

    def transform(self, raw_data):
        """
        Model transformer
        =====
        Function to featurize instances based on the fitted features 

        Parameters
        -----
        raw_data : list
            Each entry represents a text instance in the data file

        Attributes
        -----
        instances : list
            The documents represented as feature vectors
        """       
        instances = []
        for inst in raw_data:
            inst = list(inst)
            char_dict = {}
            for n in self.n_list:
                char_dict.update(utils.freq_dict(["".join(item) for item in utils.find_ngrams(inst, n)]))
            instances.append([char_dict.get(f,0) for f in self.feats])
        return np.array(instances)

    def fit_transform(self, raw_data, tagged_data):
        """
        Fit transform
        =====
        Function to perform the fit and transform sequence

        Parameters
        -----
        raw_data : list
            Each entry represents a text instance in the data file
        tagged_data : list
            List of lists, where each row represents a text instance and the columns 
            represent word - lemma - pos - sentence number respectively

        Returns
        -----
        self.transform(tagged_data) : list
            The featurized instances
        self.feats : list
            The vocabulary
        """  
        self.fit(raw_data)
        return self.transform(raw_data), self.feats
