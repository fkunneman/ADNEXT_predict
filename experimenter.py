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
        if 'bwinnow' in self.classifiers:
            self.lcs = True
        else:
            self.lcs = False
        self.directory = directory
        self.reporter = reporter.Reporter(directory, list(set(train['label'])))
    
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
        tags = self.train_csv['tagged'][:]
        if self.test_csv:
            text += self.test_csv['text']
            tags += self.test_csv['tagged']
        self.featurizer = featurizer.Featurizer(text, tags, self.directory, self.features)
        self.featurizer.fit_transform()

    def run_predictions(self, train, trainlabels, test, testlabels, classifiers, weight, prune, vocabulary):
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
        train_vectors, test_vectors, top_features, top_features_values =  vr.vectorize()
        # save vocabulary
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
        skc = sklearn_classifier.SKlearn_classifier(train, test, classifiers)
        predictions = skc.fit_transform()
        predictions['features'] = vocabulary_topfeatures
        predictions['feature_weights'] = top_features_values
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
        # If lcs classification is applied, make the necessary preparations
        if self.lcs:
            print('Preparing files LCS')
            lcs_directory = directory + 'bwinnow/'
            if not os.path.isdir(lcs_directory):
                os.mkdir(lcs_directory)
            filesdir = lcs_directory + 'files/'
            if not os.path.isdir(filesdir):
                os.mkdir(filesdir)
            parts = []
            # make chunks of 25000 from the data
            labels = self.train_csv['label']
            if self.test_csv:
                labels.extend(self.test_csv['label'])
            if instances.shape[0] > 25000:
                chunks = []
                for i in range(0, instances.shape[0], 25000):
                    chunks.append(range(i, i+25000)) 
            else:
                chunks = [range(len(labels))]
            for i, chunk in enumerate(chunks):
                # make subdirectory
                subpart = 'sd' + str(i) + '/'
                subdir = filesdir + subpart
                if not os.path.isdir(subdir):
                    os.mkdir(subdir)
                for j, index in enumerate(chunk):
                    zeros = 5 - len(str(j))
                    filename = subpart + ('0' * zeros) + str(j) + '.txt'
                    label = labels[index]
                    features = [vocabulary[x] for x in instances[index].indices]
                    with open(filesdir + filename, 'w', encoding = 'utf-8') as outfile: 
                        outfile.write('\n'.join(features))
                    parts.append(filename + ' ' + label)
            with open(lcs_directory + 'parts.txt', 'w', encoding = 'utf-8') as partsfile:
                partsfile.write('\n'.join(parts))
            # write standard lcs config file
            if weight == 'binary': 
                lts = 'TERMFREQ'
                ts = 'BOOL'
            elif weight == 'frequency':
                lts = 'TERMFREQ'
                ts = 'FREQ'
            else: # infogain or tfidf
                lts = weight.upper()
                ts = 'BOOL'
            utils.write_lcs_config(lcs_directory, ts, lts, str(prune))
        len_training = len(self.train_csv['text'])
        # if test, run experiment
        if self.test_csv:
            for classifier in self.classifiers:
                clf_dict = {classifier : self.classifiers[classifier]}
                classifier_directory = directory + classifier + '/'
                if not os.path.isdir(classifier_directory):
                    os.mkdir(classifier_directory)
                if classifier == 'bwinnow':
                    train = range(len_training)
                    test = range(len_training, len(parts))
                    clf_dict[classifier]['main'] = lcs_directory
                    clf_dict[classifier]['save'] = classifier_directory
                    skc = sklearn_classifier.SKlearn_classifier(train, test, clf_dict)
                    predictions = skc.fit_transform()
                    predictions['features'] = []
                    predictions['feature_weights'] = []
                else:
                    train = instances[:len_training]
                    test = instances[len_training:]
                    trainlabels = self.train_csv['label']
                    testlabels = self.test_csv['label']
                    predictions = self.run_predictions(train, trainlabels, test, testlabels, clf_dict, weight, prune, vocabulary)
                self.reporter.add_test([self.test_csv['text'], predictions[classifier]], predictions['features'], 
                    predictions['feature_weights'], classifier_directory)
        else: #run tenfold
            folds = utils.return_folds(len_training)
            #instances_full = list(zip(instances, self.train_csv['label'], self.train_csv['text']))
            classifier_foldperformance = defaultdict(list)
            for classifier in self.classifiers:
                classifier_directory = directory + classifier + '/'
                if not os.path.isdir(classifier_directory):
                    os.mkdir(classifier_directory)  
                clf_dict = {classifier : self.classifiers[classifier]}
                for i, fold in enumerate(folds):
                    f = i + 1
                    fold_directory = classifier_directory + 'fold_' + str(f) + '/'
                    if not os.path.isdir(fold_directory):
                        os.mkdir(fold_directory)
                    print('fold', f)
                    if classifier == 'bwinnow':
                        train = fold[0]
                        test = fold[1]
                        clf_dict[classifier]['main'] = lcs_directory
                        clf_dict[classifier]['save'] = fold_directory
                        skc = sklearn_classifier.SKlearn_classifier(train, test, clf_dict)
                        predictions = skc.fit_transform()
                        predictions['features'] = []
                        predictions['feature_weights'] = []
                    else:
                        train = instances[fold[0]]
                        trainlabels = [self.train_csv['label'][x] for x in fold[0]]
                        test = instances[fold[1]]
                        testlabels = [self.train_csv['label'][x] for x in fold[1]]
                        testdocuments = [self.train_csv['text'][x] for x in fold[1]]
                        predictions = self.run_predictions(train, trainlabels, test, testlabels, clf_dict, weight, prune, vocabulary)
                    self.reporter.add_test([testdocuments, predictions[classifier]], predictions['features'], predictions['feature_weights'], fold_directory, f)
                self.reporter.assess_performance_folds(classifier_directory)
                # to acquire a classifier model trained on all data:
                classify_all = classifier_directory + 'all/'
                if not os.path.isdir(classify_all):
                    os.mkdir(classify_all)
                print('training on all instances')
                if classifier == 'bwinnow':
                    train = range(len(instances))
                    test = range(len(instances) - 10, len(instances))
                    clf_dict[classifier]['main'] = lcs_directory
                    clf_dict[classifier]['save'] = classify_all
                    skc = sklearn_classifier.SKlearn_classifier(train, test, clf_dict)
                    predictions = skc.fit_transform()
                    predictions['features'] = []
                    predictions['feature_weights'] = []
                else:
                    train = instances
                    trainlabels = self.train_csv['label']
                    test = instances[-10:]
                    testlabels = self.train_csv['label'][-10:]
                    testdocuments = self.train_csv['text'][-10:]
                    predictions = self.run_predictions(train, trainlabels, test, testlabels, clf_dict, weight, prune, vocabulary)
                self.reporter.add_test([testdocuments, predictions[classifier]], predictions['features'], predictions['feature_weights'], classify_all, fold = 10)
        self.reporter.report_comparison()
                
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
