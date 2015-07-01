#!/usr/bin/env

from sklearn import svm, naive_bayes, tree

class SKlearn_classifier:

    def __init__(self, train, test, classifier):
        self.test = 'test'



#    def prepare(self):
        

    def train_nb(self):
        """
        Naive Bayes Learner
        =====
        Function to train a Naive Bayes classifier

        """
        self.clf = naive_bayes.MultinomialNB()
        self.clf.fit(self.training_csr, self.trainlabels)

