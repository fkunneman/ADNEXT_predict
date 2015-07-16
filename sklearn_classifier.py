#!/usr/bin/env

from sklearn import preprocessing
from sklearn import svm, naive_bayes, tree
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OutputCodeClassifier

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
        le.fit(train['labels'] + test['labels'])
        modules = {
            'Naive Bayes':          NB_classifier,
            'SVM':  SVM_classifier,
            'Tree': Tree_classifier
        }
        self.helpers = [value(le) for key, value in modules.items() if \
            key in clfs.keys()]

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
    def __init__(self, le):
        self.le = le
        self.clf = None
        self.settings = None

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
        self.clf.fit(train['instances'], self.le.transform([train['labels']]))

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
            predictions.append(self.clf.predict(instance))
            predictions_prob.append(self.clf.predict_proba(instance))
        output = zip(test['labels'], self.le.inverse_transform(predictions), 
            predictions_prob)
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
        self.fit(train)
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
    def __init__(self, le):
        self.le = le
        self.clf = None
        self.settings = None

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
        self.clf.fit(train['instances'], self.le.transform([train['labels']]))

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
            predictions.append(self.clf.predict(instance))
            predictions_prob.append(self.clf.predict_proba(instance))
        output = zip(test['labels'], self.le.inverse_transform(predictions), 
            predictions_prob)
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
        self.fit(train)
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
    def __init__(self, le):
        self.le = le
        self.clf = None
        self.settings = None

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
        # try different parameter settings for an svm outputcode classifier
        param_grid = {
            'estimator__C': [0.001, 0.005, 0.01, 0.5, 1, 5, 10, 50, 100, 500, 
                1000],
            'estimator__kernel': ['linear', 'rbf', 'poly'], 
            'estimator__gamma': [0.0005, 0.002, 0.008, 0.032, 0.128, 0.512, 
                1.024, 2.048],
            'estimator__degree': [1, 2, 3, 4]
            }
        model = OutputCodeClassifier(svm.SVC(probability = True))
        paramsearch = RandomizedSearchCV(model, param_grid, cv = 5, 
            verbose = 2, n_iter = 10, n_jobs = 12) 
        paramsearch.fit(train['instances'], self.le.transform([train['labels']]))
        self.settings = paramsearch.best_params_
        # train an SVC classifier with the settings that led to the best performance
        clf = svm.SVC(
            probability = True, 
            C = self.settings['estimator__C'],
            kernel = self.settings['estimator__kernel'],
            gamma = self.settings['estimator__gamma'],
            degree = self.settings['estimator__degree']
            )
        self.clf = OutputCodeClassifier(clf, n_jobs = 12)
        self.clf.fit(train['instances'], self.le.transform([train['labels']]))

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
            predictions.append(self.clf.predict(instance))
            predictions_prob.append(self.clf.predict_proba(instance))
        output = zip(test['labels'], self.le.inverse_transform(predictions), 
            predictions_prob)
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
        self.fit(train)
        output = (self.transform(test), self.clf, self.settings)
        return output