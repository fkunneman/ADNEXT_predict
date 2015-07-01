#!/usr/bin/env

from scipy import sparse
from sklearn import svm, naive_bayes, tree

class SKlearn_classifier:

    def __init__(self, train, test, experimentdir):
        print(test)
        print([instance[0] for instance in train][:2])
        self.train = sparse.csr_matrix([instance[0] for instance in train])
        self.trainlabels = [instance[1] for instance in train]
        self.test = [instance[0] for instance in test]
        self.testlabels = [instance[1] for instance in test]
        self.expdir = experimentdir

    def train_nb(self):
        """
        Naive Bayes Learner
        =====
        Function to train a Naive Bayes classifier

        Uses
        -----
        self.train : list of train instances
        self.trainlabels : list of train labels

        Generates
        -----
        Trained Naive Bayes classifier
        """
        self.clf = naive_bayes.MultinomialNB()
        self.clf.fit(self.train, self.trainlabels)

    def predict(self):
        """
        Classifier
        =====
        Function to apply the saved classifier on the test set

        Uses
        -----
        self.clf : trained classifier
        self.test : list of test instances
        self.testlabels : list of test labels

        Generates
        -----
        self.predictions : list
            list of target, prediction and probability per instance
        """
        self.predictions = []
        for i, instance in enumerate(self.test):
            prediction = self.clf.predict(instance)
            proba = self.clf.predict_proba(t)
            self.predictions.append(self.testlabels[i], prediction, proba)
