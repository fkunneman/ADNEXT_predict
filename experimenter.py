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
        print("Vocab", vocabulary[:50], vocabulary[-50:])
        # Save vocabulary
        with open(directory + 'vocabulary.txt', 'w', encoding = 'utf-8') as v_out:
            v_out.write('\n'.join(vocabulary))
        # if test, run experiment
        if self.test_csv:
            vr = vectorizer.Vectorizer(instances[:len(self.train_csv['text'])],
                instances[len(self.train_csv['text']):], weight, prune)
            train_vectors, test_vectors =  vr.vectorize()
            train = {
                'instances' : train_vectors,
                'labels'    : self.train_csv['label']
            }
            test = {
                'instances' : test_vectors,
                'labels'    : self.test_csv['label']
            }
            print("Performing classification")
            skc = sklearn_classifier.SKlearn_classifier(train, test, self.classifiers)
            results = skc.fit_transform()
            print(results)

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
        featuretypes = []
        for length in range(1, len(self.features.keys()) + 1):
            for subset in itertools.combinations(self.features.keys(), length):
                featuretypes.append(list(subset))
        weights = ['binary']
        pruning = [5000]
        all_settings = [featuretypes, weights, pruning]
        combinations = list(itertools.product(*all_settings))
        # For each cell
        for combination in combinations:
            print('Combi', combination)
            featurestring = '+'.join(combination[0])
            directory = self.directory + featurestring + '_' + '_'.join([str(x) for x in combination[1:]]) + '/'
            print("Directory", directory)
            if not os.path.isdir(directory):
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
        text = self.train_csv['text'][:] 
        frogs = self.train_csv['frogs'][:]
        if self.test_csv:
            text += self.test_csv['text']
            frogs += self.test_csv['frogs']
        self.featurizer = featurizer.Featurizer(text, frogs, self.features)
        self.featurizer.fit_transform()
