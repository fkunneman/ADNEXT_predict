#!/usr/bin/env 

import time
import os
import itertools

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
    Interface to classifiers. Can run a grid of combinations

    Parameters
    ------
    classifiers : list
        names of classifiers to include
        options : 'lcs', 'svm', 'knn', 'nb', 'ripper', 'tree', 'ensemble_clf', 
            'ensemble_prediction', 'joint_prediction'    
    directory : str
        the directory to write all experiment files to

    Attributes
    -----
        features : holder of 'features' parameter
        featurefilter : holder of 'featurefilter' parameter
        classifiers : holder of 'classifiers' parameter
        directory : holder of 'directory' parameter
        grid : holder of 'grid' parameter
        featurized : list
            holder featurized instances to be fed to classifiers

    Examples
    -----
    grid = experimenter.ExperimentGrid(features, classifiers, directory, 'low')
    grid.set_features(dataset)
    grid.experiment()

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
    
    def run_predictions(self, train, test, trainlabels, testlabels, weight, prune):
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
        # Select features
        instances, vocabulary = self.featurizer.return_instances(featuretypes)
        print("Instance", instances[:5])
        print("Vocab", vocabulary[:50], vocabulary[-50:])
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
            print(predictions)
        else: #run 10-fold
            instances_labels = zip(instances, self.train_csv['label'])
            folds = utils.return_folds(instances_labels)
            fold_predictions = []
            for i, fold in enumerate(folds):
                print(fold, i)
                train = [x[0] for x in fold[0]]
                trainlabels = [x[1] for x in fold[0]]
                test = [x[0] for x in fold[1]]
                testlabels = [x[1] for x in fold[1]]
                predictions = self.run_predictions(train, test, trainlabels, testlabels, weight, prune)
                fold_predictions.append(predictions)

    def run_grid(self):
        """
        Classifier interface
        =====
        Function to perform classifications
        """ 
        # Make grid
        featuretypes = []
        for length in range(1, len(self.features.keys()) + 1):
            for subset in itertools.combinations(self.features.keys(), length):
                featuretypes.append(list(subset))
        all_settings = [featuretypes, self.weights, self.prune]
        combinations = list(itertools.product(*all_settings))
        # For each cell
        for combination in combinations:
            print('Combi', combination)
            featurestring = '+'.join(combination[0])
            directory = self.directory + featurestring + '_' + '_'.join([str(x) for x in combination[1:]]) + '/'
            print("Directory", directory)
            if not os.path.isdir(directory):
                os.mkdir(directory)
            if self.test_csv:
                self.run_experiment(combination[0], combination[1], combination[2], directory)
            else:
                self.run_experiment_10fold(combination[0], combination[1], combination[2], directory)

    def set_features(self):
        """
        Featurizer interface
        =====
        Function to transform documents into features

        Parameters
        -----
        features : dict
            dictionary of features to include
            (see the featurizer class for an overview of the features and how 
                to extract them)
        """
        text = self.train_csv['text'][:] 
        frogs = self.train_csv['frogs'][:]
        if self.test_csv:
            text += self.test_csv['text']
            frogs += self.test_csv['frogs']
        self.featurizer = featurizer.Featurizer(text, frogs, self.features)
        self.featurizer.fit_transform()
