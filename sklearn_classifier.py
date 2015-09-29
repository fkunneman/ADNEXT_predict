#!/usr/bin/env

import os
from sklearn import preprocessing
from sklearn import svm, naive_bayes, tree
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OutputCodeClassifier
from scipy import sparse

import utils

class SKlearn_classifier:
    """
    SKlearn classifier
    =====
    Interface to SKlearn classifications

    Parameters
    -----
    train : dict
        labels = list of train labels
        instances = list of featurized train instances        
    test : dict
        labels = list of test labels
        instances = list of featurized test instances
    clfs : list
        names of classifiers to include
        options : 'lcs', 'svm', 'knn', 'nb', 'ripper', 'tree', 'ensemble_clf', 
            'ensemble_prediction', 'joint_prediction'

    Attributes
    -----
    self.train : dict
        The train parameter
    self.test : dict
        The test parameter
    le : LabelEncoder
        Object to transform string labels to labels in SKlearn format (and back)
    modules : dict
        keys : clfs
        values : classifier classes
    self.helpers : list
        List to call all the selected classifiers
    """
    def __init__(self, train, test, clfs):
        self.train = train
        self.test = test
        le = preprocessing.LabelEncoder()
        try:
            le.fit(train['labels'] + test['labels'])
        except:
            print('No labelencoding')
        modules = {
            'nb' :              NB_classifier,
            'svm' :             SVM_classifier,
            'tree' :            Tree_classifier,
            'bwinnow' :         LCS_classifier,
            'ensemble_clf' :    EnsembleClf_classifier
        }
        self.helpers = [modules[key](le, **clfs[key]) for key in clfs.keys()]

    def fit_transform(self):
        """
        Interface function
        =====
        Function to train and test all classifiers

        Uses
        -----
        self.train : dict 
            The train data
        self.test : dict
            The test data
        self.helpers : list
            The classifiers

        Returns
        -----
        output : dict
            keys : classifiers
            values : classifier output
        """
        output = {}
        for helper in self.helpers:
            print(helper, 'classification')
            output[helper.name] = helper.fit_transform(self.train, self.test)
        return output

class NB_classifier:
    """
    Naive Bayes Classifier
    =====
    Class to perform Naive Bayes classification

    Parameters
    -----
    le : LabelEncoder
        Object to transform string labels to labels in SKlearn format (and back)

    Attributes
    -----
    self.le : LabelEncoder
        The le parameter
    self.clf : MultinomialNB
        Classifier model
    self.settings : dict
        Classifier parameter settings

    """
    def __init__(self, le, **kwargs):
        self.name = 'nb'
        self.le = le
        self.clf = False
        self.settings = False

    def fit(self, train):
        """
        Naive Bayes Learner
        =====
        Function to train a Naive Bayes classifier

        Uses
        -----
        self.train : dict
            labels = list of train labels
            instances = list of featurized train instances

        Generates
        -----
        self.clf : MultinomialNB
            Trained Naive Bayes classifier
        """
        self.clf = naive_bayes.MultinomialNB()
        self.clf.fit(train['instances'], self.le.transform(train['labels']))

    def transform(self, test):
        """
        Classifier
        =====
        Function to apply the saved classifier on the test set

        Parameters
        -----
        test : dict
            labels = list of test labels
            instances = list of featurized test instances

        Uses
        -----
        self.clf : MultinomialNB

        Returns
        -----
        output : zip of lists
            list of target labels
            list of classifications
            list of prediction certainties
        """
        predictions = []
        predictions_prob = []
        for i, instance in enumerate(test['instances']):
            prediction = self.clf.predict(instance)[0]
            predictions.append(prediction)
            predictions_prob.append(self.clf.predict_proba(instance)[0][prediction])
        output = list(zip(test['labels'], self.le.inverse_transform(predictions), predictions_prob))
        return output

    def fit_transform(self, train, test):
        """
        Interface function
        =====
        Function to train and test the classifier

        Parameters
        -----
        train : dict
            labels = list of train labels
            instances = list of featurized train instances
        test : dict
            labels = list of test labels
            instances = list of featurized test instances

        Returns
        -----
        output : tuple
            test output : zip of lists
            self.clf : MultinomialNB
            self.settings : dict
                parameter settings
        """
        print('Fitting')
        self.fit(train)
        print('Transforming')
        output = (self.transform(test), self.clf, self.settings)
        return output

class Tree_classifier:
    """
    Decision Tree Classifier
    =====
    Class to perform Decision tree classification

    Parameters
    -----
    le : LabelEncoder
        Object to transform string labels to labels in SKlearn format 
        (and back)

    Attributes
    -----
    self.le : LabelEncoder
        The le parameter
    self.clf : DecisionTreeClassifier
        Classifier model
    self.settings : dict
        Classifier parameter settings
    """
    def __init__(self, le, **kwargs):
        self.name = 'tree'
        self.le = le
        self.clf = False
        self.settings = False

    def fit(self, train):
        """
        Naive Bayes Learner
        =====
        Function to train a Naive Bayes classifier

        Uses
        -----
        self.train : dict
            labels = list of train labels
            instances = list of featurized train instances

        Generates
        -----
        self.clf : DecisionTreeClassifier
            Trained Decision Tree classifier
        """
        self.clf = tree.DecisionTreeClassifier()
        self.clf.fit(train['instances'], self.le.transform(train['labels']))

    def transform(self, test):
        """
        Classifier
        =====
        Function to apply the saved classifier on the test set

        Parameters
        -----
        test : dict
            labels = list of test labels
            instances = list of featurized test instances

        Uses
        -----
        self.clf : MultinomialNB

        Returns
        -----
        output : zip of lists
            list of target labels
            list of classifications
            list of prediction certainties
        """
        predictions = []
        predictions_prob = []
        for i, instance in enumerate(test['instances']):
            prediction = self.clf.predict(instance)[0]
            predictions.append(prediction)
            predictions.append(self.clf.predict(instance))
            predictions_prob.append(self.clf.predict_proba(instance)[0][prediction])
        output = list(zip(test['labels'], self.le.inverse_transform(predictions), predictions_prob))
        return output

    def fit_transform(self, train, test):
        """
        Interface function
        =====
        Function to train and test the classifier

        Parameters
        -----
        train : dict
            labels = list of train labels
            instances = list of featurized train instances
        test : dict
            labels = list of test labels
            instances = list of featurized test instances

        Returns
        -----
        output : tuple
            test output : zip of lists
            self.clf : DecisionTreeClassifier
            self.settings : dict
                parameter settings
        """
        print('Fitting')
        self.fit(train)
        print('Transforming')
        output = (self.transform(test), self.clf, self.settings)
        return output

class SVM_classifier:
    """
    Support Vector Machines Classifier
    =====
    Class to perform Support Vector Machines classification

    Parameters
    -----
    le : LabelEncoder
        Object to transform string labels to labels in SKlearn format 
        (and back)

    Attributes
    -----
    self.le : LabelEncoder
        The le parameter
    self.clf : SVC
        Classifier model
    self.settings : dict
        Classifier parameter settings
    """
    def __init__(self, le, **kwargs):
        self.name = 'svm'
        self.le = le
        # Different settings are required if there are more than two classes
        if len(self.le.classes_) > 2:
            self.multi = True
        else:
            self.multi = False
        self.clf = False
        self.settings = {}
        self.c = kwargs['C'] if 'C' in kwargs.keys() else False
        self.kernel = kwargs['kernel'] if 'kernel' in kwargs.keys() else False
        self.gamma = kwargs['gamma'] if 'gamma' in kwargs.keys() else False
        self.degree = kwargs['degree'] if 'degree' in kwargs.keys() else False
        self.cweight = kwargs['weight'] if 'weight' in kwargs.keys() else None
        self.approach = kwargs['params'] if 'params' in kwargs.keys() else 'default'

    def fit(self, train):
        """
        Support Vector Machines classifier
        =====
        Function to train a Support Vector Machines classifier

        Uses
        -----
        self.train : dict
            labels = list of train labels
            instances = list of featurized train instances

        Generates
        -----
        self.clf : SVC
            Trained Support Vector Machines classifier
        """
        # scale data
        # self.min_max_scaler = preprocessing.MinMaxScaler()
        # train_minmax = self.min_max_scaler.fit_transform(train['instances'].toarray())
        # train_minmax = sparse.csr_matrix(train_minmax)
        if self.multi:
            params = ['estimator__C', 'estimator__kernel', 'estimator__gamma', 'estimator__degree']
        else:
            params = ['C', 'kernel', 'gamma', 'degree']
        if self.approach == 'paramsearch':
            # try different parameter settings for an svm outputcode classifier
            grid_values = [
                [0.001, 0.005, 0.01, 0.5, 1, 5, 10, 50, 100, 500, 1000],
                ['linear', 'rbf', 'poly'], 
                [0.0005, 0.002, 0.008, 0.032, 0.128, 0.512, 1.024, 2.048],
                [1, 2, 3, 4]
            ]
            param_grid = {}
            param_grid[params[0]] = self.c if self.c else grid_values[0]
            param_grid[params[1]] = self.kernel if self.kernel else grid_values[1]
            param_grid[params[2]] = self.gamma if self.gamma else grid_values[2]
            param_grid[params[3]] = self.degree if self.degree else grid_values[3]
            model = svm.SVC(probability=True, class_weight = cweight)
            if self.multi:
                model = OutputCodeClassifier(model)
            paramsearch = RandomizedSearchCV(model, param_grid, cv = 5, verbose = 3, n_iter = 10, n_jobs = 10, pre_dispatch = 4) 
            paramsearch.fit(train['instances'], self.le.transform(train['labels']))
            self.settings = paramsearch.best_params_
        elif self.approach == 'default':
            self.settings[params[0]] = self.c if self.c else 1
            self.settings[params[1]] = self.kernel if self.kernel else 'linear'
            self.settings[params[2]] = self.gamma if self.gamma else 1 / train['instances'].shape[1] # number of features
            self.settings[params[3]] = self.degree if self.degree else 3
        # train an SVC classifier with the settings that led to the best performance
        self.clf = svm.SVC(
           probability = True, 
           C = self.settings[params[0]],
           kernel = self.settings[params[1]],
           gamma = self.settings[params[2]],
           degree = self.settings[params[3]],
           cache_size = 1000,
           class_weight = self.cweight,
           verbose = 2
           )      
        if self.multi:
            self.clf = OutputCodeClassifier(self.clf)
        self.clf.fit(train['instances'], self.le.transform(train['labels']))

    def transform(self, test):
        """
        Classifier
        =====
        Function to apply the saved classifier on the test set

        Parameters
        -----
        test : dict
            labels = list of test labels
            instances = list of featurized test instances

        Uses
        -----
        self.clf : SVC

        Returns
        -----
        output : zip of lists
            list of target labels
            list of classifications
            list of prediction certainties
        """
        predictions = []
        predictions_prob = []
        for i, instance in enumerate(test['instances']):
            prediction = self.clf.predict(instance)[0]
            predictions.append(prediction)
            if self.multi:
                predictions_prob.append(0)
            else:
                predictions_prob.append(self.clf.predict_proba(instance)[0][prediction])
        output = list(zip(test['labels'], self.le.inverse_transform(predictions), predictions_prob))
        return output

    def fit_transform(self, train, test):
        """
        Interface function
        =====
        Function to train and test the classifier

        Parameters
        -----
        train : dict
            labels = list of train labels
            instances = list of featurized train instances
        test : dict
            labels = list of test labels
            instances = list of featurized test instances

        Returns
        -----
        output : tuple
            test output : zip of lists
            self.clf : SVC
            self.settings : dict
                parameter settings with keys:
                    estimator__C
                    estimator__kernel
                    estimator__gamma
                    estimator__degree
        """
        print('Fitting')
        self.fit(train)
        print('Transforming')
        output = (self.transform(test), self.clf, self.settings)
        return output

class EnsembleClf_classifier:
    """
    Ensemble classification Classifier
    =====
    Class for ensemble classification, leveraging the judgements of multiple classifiers

    """
    def __init__(self, le, **kwargs):
        self.name = 'ensemble_clf'
        self.le = le
        self.modules = {
            'nb':               NB_classifier,
            'svm':              SVM_classifier,
            'tree':             Tree_classifier
        }
        self.helpers = kwargs['helpers']
        self.assessor = kwargs['assessor']
        self.approach = kwargs['approach']

    def add_classification_features(self, clfs, test):
        instances = test['instances'].copy()
        if self.approach == 'classifications_only':
            instances = sparse.csr_matrix([[]] * instances.shape[0])   
        for clf in clfs:
            output = clf.transform(test)
            classifications = self.le.transform([x[1] for x in output])
            classifications_csr = sparse.csr_matrix([[classifications[i]] for i in range(classifications.size)])
            if self.approach == 'classifications_only':
                if instances.shape[1] == 0:
                    instances = classifications_csr
                else:
                    instances = sparse.hstack((instances, classifications_csr))
            else:
                instances = sparse.hstack((instances, classifications_csr))
        return (instances)

    def fit(self, train):
        """

        """
        # train helper classifiers
        self.clfs = [value(self.le, **self.helpers[key]) for key, value in self.modules.items() if key in self.helpers.keys()]
        for clf in self.clfs:
            clf.fit(train)
        # extend training data with classification features
        # make folds
        folds = utils.return_folds(len(train['labels']))
        # add classifier features
        new_instances = []
        new_labels = []
        for fold in folds:
            fold_train = {
                'instances' : sparse.vstack([train['instances'][i] for i in fold[0]]), 
                'labels' : [train['labels'][i] for i in fold[0]]
            }
            fold_test = {
                'instances' : sparse.vstack([train['instances'][i] for i in fold[1]]), 
                'labels' : [train['labels'][i] for i in fold[1]]
            }
            clfs = [value(self.le, **self.helpers[key]) for key, value in self.modules.items() if key in self.helpers.keys()]
            for clf in clfs:
                clf.fit(fold_train)
            test_instances = self.add_classification_features(clfs, fold_test)
            new_instances.append(test_instances)
            new_labels.extend(fold_test['labels'])
        new_instances_folds = sparse.csr_matrix(sparse.vstack(new_instances))
        train_classifications = {'instances' : new_instances_folds, 'labels' : new_labels}
        # train ensemble classifier
        self.ensemble = self.modules[self.assessor[0]](self.le, **self.assessor[1])        
        self.ensemble.fit(train_classifications)

    def transform(self, test):
        """

        """
        # extend test data with classification features
        test_instances = self.add_classification_features(self.clfs, test)
        test_all = {'instances' : sparse.csr_matrix(test_instances), 'labels' : test['labels']}
        # make predictions
        output = self.ensemble.transform(test_all)
        return (output)

    def fit_transform(self, train, test):
        """

        """
        print('Fitting')
        self.fit(train)
        print('Transforming')
        tf = self.transform(test)
        output = (tf, self.ensemble.clf, self.ensemble.settings)
        return output

class LCS_classifier:
    """
    Classification by LCS balanced winnow
    ======
    Interface to LCS classifier

    Parameters
    ------
    train : list 
        featurized training instances
    test : list
        featurized test instances
    directory : str
        directory in which classificationfiles are written 
        experiment itself is performed in current directory
    vocabulary : dict
        dictionary with a mapping between indices and features

    Attributes
    -----
    train : holder of 'train' parameter
    test : holder of 'test' parameter
    expdir : holder of 'expdir' parameter
    vocabulary : holder of 'vocabulary' parameter
    convert_features : function call
        converts featurized features to format used in LCS classification

    Examples
    -----
    Interactive:

    >>> reader = Datareader(max_n=1000)
    >>> reader.set('blogs.csv')
    >>> docs = reader.rows
    >>> reader.set_rows(docs)

    """

    def __init__(self, le, **kwargs):
        """
        """
        self.name = 'bwinnow'
        self.clf = False
        self.settings = False
        self.maindir = kwargs['main']
        self.savedir = kwargs['save']
        self.expdir = os.getcwd() + '/'
        self.targets = {}

    def fit(self, train, test):
        with open(self.maindir + 'parts.txt') as partsfile:
            parts = partsfile.read().split('\n')
        traininstances = [parts[i] for i in train]
        with open(self.expdir + 'train', 'w', encoding = 'utf-8') as trainfile:
            trainfile.write('\n'.join(traininstances))
        testinstances = [parts[i] for i in test]
        for instance in testinstances:
            file_label = instance.split()
            self.targets[file_label[0]] = file_label[1]
        with open(self.expdir + 'test', 'w', encoding = 'utf-8') as testfile:
            testfile.write('\n'.join(testinstances))
        os.system('cp ' + self.maindir + 'lcs3.conf .')

    def transform(self):
        """
        LCS classifier
        =====
        Function to call upon the LCS classifier to train and test on the 
            partsfiles.
        Classifier needs to be properly set-up
        Performs classification in current directory

        Parameters
        -----
        trainparts : list
            all train instances as line with a file reference and label
        testparts : list
            all test instances as line with a file reference and label
        """
        os.system("lcs --verbose")
        predictions = self.extract_performance()
        os.system("mv * " + self.savedir)
        return predictions

    def extract_performance(self):
        performance = []
        with open('test.rnk') as rnk:
            for i, line in enumerate(rnk.readlines()):
                tokens = line.strip().split()
                filename = tokens[0].strip()
                classification, score = tokens[1].split()[0].split(":")
                classification = classification.replace("?","")
                target = self.targets[filename]
                performance.append([target, classification, float(score)])
        return performance

    def fit_transform(self, train, test):
        self.fit(train, test)
        output = (self.transform(), self.clf, self.settings)
        return output
