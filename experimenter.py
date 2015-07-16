#!/usr/bin/env 

import time
import os
import itertools

import featurizer
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

    def __init__(self, train, test, features, classifiers, directory):
        self.train_csv = train
        self.test_csv = test # can be False
        self.features = features
        self.featurizer = False
        self.classifiers = classifiers
        self.directory = directory
    
    def run_experiment(self, featuretypes, weight, prune, directory):
        # Select features
        instances, vocabulary = self.featurizer.return_instances(featuretypes)
        print("Instance", instances[:5])
        print("Vocab", vocabulary[:25])
        quit()
        # Save vocabulary
        with open(directory + 'vocabulary.txt', 'w', encoding = 'utf-8') as v_out:
            v_out.write('\n'.join(vocabulary))
        # if test, run experiment
        if self.test_csv:
            train = {
                'featurized' : instances[:len(self.train_csv['text'])],
                'labels'    : self.train_csv['label']
            }
            test = {
                'featurized' : instances[len(self.train_csv['text']):],
                'labels'    : self.test_csv['label']
            } 



        else: # 10-fold
            pass

        # Weight and prune

        # Classify

    def run_grid(self):
        """
        Classifier interface
        =====
        Function to perform classifications
        """ 
        # Make grid
        featuretypes = self.features.keys()
        weights = ['binary']
        pruning = [5000]
        combinations = list(itertools.product(featuretypes, 
            list(itertools.product(weights, pruning))))
        # For each cell
        for combination in combinations:
            print("Combi", combination)
            directory = self.directory + "_".join(combination) + "/"
            print("Directory", directory)
            os.mkdir(directory)
            self.run_experiment(combination[0], combination[1], combination[2],
                directory)


        # if self.grid == "low" or self.grid == "normal": #all single classifiers
        #     classifiers = [[clf] for clf in self.classifiers]
        # #elif self.grid == "high": #all single classifiers + all different combinations of ensemble
        # experimentlog = self.directory + 'log.txt'
        # overview = self.directory + 'overview.txt'
        # expindex = 1
        # print("iter", list(itertools.product(self.featurized, classifiers)))
        # quit()
        # for setting in self.featurized:
        #     train, test, vocabulary, featuretypes = setting
        #     clf = sklearn_classifier.SKlearn_classifier(train, test)
        #     for classifier in classifiers:            
        #         expdir = self.directory + 'exp' + str(expindex) + '/'
        #         os.mkdir(expdir)
        #         expindex += 1
        #         #report on experiment
        #         expname = expdir + "\t" + features + '_' + '+'.join(classification)
        #         print('classifying', expname)
        #         with open(experimentlog, 'a') as el:
        #             el.write(str(time.asctime()) + '\t' + expname + '\n')
        #         #perform classification
        #         if len(classifier) == 1:
        #             classifier = classifier[0]
        #         if classifier == 'lcs':
        #             clf = lcs_classifier.LCS_classifier(train, test, expdir, vocabulary)
        #             clf.classify()
        #         else:
        #             if classifier == 'nb':
        #                 clf.train_nb()
        #             clf.predict()

        #         rep = reporter.Reporter()
        #         rep.add_instances(clf.predictions)
        #         performance = rep.calculate_performance()
        #         with open(expdir + "results.txt", "w") as resultsfile:
        #             resultsfile.write(
        #                 "\n".join(["\t".join([str(x) for x in label]) for label in performance]))
        #         with open(overview, "a") as ov:
        #             ov.write("\t".join([expname] + performance[-1]))


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
        text = self.train_csv['text'] 
        frogs = self.train_csv['frogs']
        if self.test_csv:
            text += self.test_csv['text']
            frogs += self.test_csv['frogs']
        self.featurizer = featurizer.Featurizer(text, frogs, self.features)
