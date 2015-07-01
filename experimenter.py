#!/usr/bin/env 

import time
import os

import featurizer
import lcs_classifier
import sklearn_classifier
import reporter
import utils

class ExperimentGrid:
    """
    Experiment wrapper
    ======
    Interface to classifiers. Can run a grid of combinations

    Parameters
    ------
    features : dict
        dictionary of features to include
        keys : 'token_ngrams' (list of int), 'max_feats' : int
    featurefilter : list
        strings of feature tokens to exclude
    classifiers : list
        names of classifiers to include
        options : 'lcs', 'svm', 'knn', 'nb', 'ripper', 'tree'    
    directory : str
        the directory to write all experiment files to
    grid : str
        parameter to define the degree of different experiment combinations
        options : 'low', 'normal', 'high'

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

    def __init__(self, features, classifiers, directory, grid):
        self.features = features
        self.classifiers = classifiers
        self.directory = directory
        self.grid = grid
        self.featurized = []

    def set_features(self, train, test = False, grid = 'low'):
        """
        Featurizer interface
        =====
        Function transform documents into features

        Parameters
        -----
        train : dict
            csv dictionary from datahandler
        test : dict, default = False
            csv dictionary from datahandler

        Attributes
        -----
        settings : the number of different feature combinations
        """
        if self.grid == 'low': #only one setting
            settings = [self.features]
        #else: #combinations of settings
        for setting in settings:
            text = train['text']
            frog = train['frogs'] 
            if test:
                text += test['text']
                frog += test['frogs']      
            fr = featurizer.Featurizer(text, frog, setting)
            vectors, vocabulary = fr.fit_transform()
            train_instances = list(zip(vectors[:len(train['label'])], train['label']))
            if test:
                test_instances = list(zip(vectors[len(train['label']):], test['label']))
            else:
                test_instances = False
            # the different feature settings are appended to a class-level list, in order to 
            # make different combinations of features and classifiers in the experiment function
            self.featurized.append([train_instances, test_instances, vocabulary, setting.keys()])

    def experiment(self):
        """
        Classifier interface
        =====
        Function to perform classifications
        """
        if self.grid == "low" or self.grid == "normal": #all single classifiers
            classifications = [[clf] for clf in self.classifiers]
        #elif self.grid == "high": #all single classifiers + all different combinations of ensemble

        experimentlog = self.directory + 'log.txt'
        overview = self.directory + 'overview.txt'
        expindex = 1
        for classification in classifications:
            for setting in self.featurized:
                train, test, vocabulary, featuretypes = setting
                features = '-'.join(featuretypes)
                expdir = self.directory + 'exp' + str(expindex) + '/'
                clf = sklearn_classifier.SKlearn_classifier(train, test, expdir)
                os.mkdir(expdir)
                expindex += 1
                #report on experiment
                expname = expdir + "\t" + features + '_' + '+'.join(classification)
                print('classifying', expname)
                with open(experimentlog, 'a') as el:
                    el.write(str(time.asctime()) + '\t' + expname + '\n')
                #perform classification
                if len(classification) == 1:
                    classifier = classification
                if classifier == 'lcs':
                    clf = lcs_classifier.LCS_classifier(train, test, expdir, vocabulary)
                    clf.classify()
                else:
                    if classifier == 'nb':
                        clf.train_nb()
                    clf.predict()

                rep = reporter.Reporter()
                rep.add_instances(clf.predictions)
                performance = rep.calculate_performance()
                with open(expdir + "results.txt", "w") as resultsfile:
                    resultsfile.write(
                        "\n".join(["\t".join([str(x) for x in label]) for label in performance]))
                with open(overview, "a") as ov:
                    ov.write("\t".join([expname] + performance[-1]))
