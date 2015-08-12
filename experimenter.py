###############################################################
#  PyNLPl - Evaluation Library
#       by Maarten van Gompel (proycon)
#       http://ilk.uvt.nl/~mvgompel
#       Induction for Linguistic Knowledge Research Group
#       Universiteit van Tilburg
#
#       Licensed under GPLv3
#
# This is a Python library with classes and functions for evaluation
# and experiments .
#
###############################################################

import time
import os
import itertools
from collections import defaultdict
import pickle

import featurizer
import vectorizer
import lcs_classifier
import sklearn_classifier
import reporter
import utils

class Experiment:
    """
    Experiment wrapper
    ======
    Interface to classification experiments. Can run a grid of combinations

    Parameters
    ------
    train : list
        train instances from the datahandler class
    test : list
        test instances from the datahandler class
        can be set to false to run tenfold classifications
    features : dict
        dict of featuretypes to extract, as input to the featurizer class
        all single featuretypes and combinations of featuretypes will be 
            included in the grid
        options : 
            'char_ngrams' --> 'n_list', 'blackfeats'
            'token_ngrams' --> 'n_list', 'blackfeats'
            'lemma_ngrams' --> 'n_list', 'blackfeats'
            'pos_ngrams' --> 'n_list', 'blackfeats'
    weights : list
        list of featureweights to include, as input to the vectorizer class
        options : 'frequency', 'binary', 'tfidf', 'infogain', 'pmi'
    prune : int
        top N features to select after weighting, as input to the vectorizer class
        all features not in the top N features with the highest weight are pruned
        only one pruning value is used throughout the grid, to reduce its size 
    classifiers : list
        list of classifiers to apply
        options : 'svm', 'knn', 'nb', 'ripper', 'tree', 'ensemble_clf', 
            'ensemble_prediction', 'joint_prediction'    
    directory : str
        the directory to write all experiment output to

    Example
    -----
    grid = experimenter.Experiment(train, False, features, weights, prune, clfs, directory)
    grid.set_features()
    grid.run_grid()
    """

    def __init__(self, train, test, features, weights, prune, classifiers, directory):
        self.train_csv = train
        self.test_csv = test # can be False
        self.features = features
        self.weights = weights
        self.prune = prune
        self.featurizer = False
        self.classifiers = classifiers
        self.directory = directory
        self.reporter = reporter.Reporter(directory, list(set(train['labels'])))
    
    def set_features(self):
        """
        Featurizer interface
        =====
        Function to transform documents into lists of feature values

        Transforms
        -----
        self.featurizer : featurizer.Featurizer object
            container of the feature values for different feature types, with an accompanying feature vocabulary
        """
        text = self.train_csv['text'][:] 
        frogs = self.train_csv['frogs'][:]
        if self.test_csv:
            text += self.test_csv['text']
            frogs += self.test_csv['frogs']
        self.featurizer = featurizer.Featurizer(text, frogs, self.features)
        self.featurizer.fit_transform()

    def run_predictions(self, train, trainlabels, test, testlabels, weight, prune):
        """
        Classification nterface
        =====
        Function to run classifiers on a given set of train and test instances

        Parameters
        -----
        train : list
            list of featurized train instances
        trainlabels : list
            list of train labels
            each index of a label corresponds to the index of the train instance
        test : list
            list of featurized test instances
        testlabels : list
            list of test labels
            each index of a label corresponds to the index of the test instance
        weight : str
            feature weight to apply, as input to the vectorizer class
            options : 'frequency', 'binary', 'tfidf', 'infogain', 'pmi'
        prune : int
            top N features to select after weighting, as input to the vectorizer class
            all features not in the top N features with the highest weight are pruned

        Returns
        -----
        predictions : dict
            keys : classifiers (str)
            values : classifier output (tup) 
                0 - prediction output : zip of lists
                    list of target labels
                    list of classifications
                    list of prediction certainties
                1 - classifier model : pkl
                2 - parameter settings : dict
        """
        print('running vectorizer', weight, prune)
        vr = vectorizer.Vectorizer(train, test, trainlabels, weight, prune)
        train_vectors, test_vectors =  vr.vectorize()
        train = {
            'instances' : train_vectors,
            'labels'    : trainlabels
        }
        test = {
            'instances' : test_vectors,
            'labels'    : testlabels
        }
        print("Performing classification")
        skc = sklearn_classifier.SKlearn_classifier(train, test, self.classifiers)
        predictions = skc.fit_transform()
        return predictions

    def run_experiment(self, featuretypes, weight, prune, directory):
        """
        Experiment interface
        =====
        Function to run an experiment based on a combination of feature, weight and pruning settings
        Will run train-test or tenfold dependent on the value of self.test_csv

        Parameters
        -----
        Featuretypes : list
            list of the feature types to combine, as input to the self.featurizer object
            options : 'char_ngrams', 'token_ngrams', 'lemma_ngrams', 'pos_ngrams'
        weight : str
            feature weight to apply, as input to the vectorizer class
            options : 'frequency', 'binary', 'tfidf', 'infogain', 'pmi'
        prune : int
            top N features to select after weighting, as input to the vectorizer class
            all features not in the top N features with the highest weight are pruned
        directory : str
            the directory to write the experiment output to        
        """
        # Select features
        instances, vocabulary = self.featurizer.return_instances(featuretypes)
        # Save vocabulary
        with open(directory + 'vocabulary.txt', 'w', encoding = 'utf-8') as v_out:
            v_out.write('\n'.join(vocabulary))
        # if test, run experiment
        if self.test_csv:
            len_training = len(self.train_csv['text'])
            train = instances[:len_training]
            trainlabels = self.train_csv['label']
            test = instances[len_training:]
            testlabels = self.test_csv['label']
            predictions = self.run_predictions(train, trainlabels, test, testlabels, weight, prune)
            self.save_predictions(self.test_csv['text'], predictions, directory)
        else: #run tenfold
            instances_full = list(zip(instances, self.train_csv['label'], self.train_csv['text']))
            folds = utils.return_folds(instances_full)
            classifier_foldperformance = defaultdict(list)
            for i, fold in enumerate(folds):
                print('fold', i)
                train = [x[0] for x in fold[0]]
                trainlabels = [x[1] for x in fold[0]]
                test = [x[0] for x in fold[1]]
                testlabels = [x[1] for x in fold[1]]
                testdocuments = [x[2] for x in fold[1]]
                predictions = self.run_predictions(train, trainlabels, test, testlabels, weight, prune)
                for classifier in self.classifiers:
                    classifier_foldperformance[classifier].append([testdocuments, predictions[classifier]])
            for classifier in self.classifiers:
                classifier_directory = directory + classifier + '/'
                if not os.path.isdir(classifier_directory):
                    os.mkdir(classifier_directory)
                self.reporter.add_folds_test(classifier_foldperformance[classifier], classifier_directory)

    def run_grid(self):
        """
        Grid interface
        =====
        Function to generate and run a grid of experiments
        """ 
        # Make grid
        featuretypes = []
        for length in range(1, len(self.features.keys()) + 1):
            for subset in itertools.combinations(self.features.keys(), length):
                featuretypes.append(list(subset))
        all_settings = [featuretypes, self.weights, self.prune]
        combinations = list(itertools.product(*all_settings))
        # For each grid cell
        for combination in combinations:
            print('Combi', combination)
            featurestring = '+'.join(combination[0])
            directory = self.directory + featurestring + '_' + '_'.join([str(x) for x in combination[1:]]) + '/'
            print("Directory", directory)
            if not os.path.isdir(directory):
                os.mkdir(directory)
            self.run_experiment(combination[0], combination[1], combination[2], directory)
