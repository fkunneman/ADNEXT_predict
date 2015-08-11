#!/usr/bin/env 

import os
import pickle
from sklearn.metrics import auc

from pynlpl import evaluation
import utils

class Reporter:

    def __init__(self, grid_directory):
        self.directory = grid_directory
        self.comparison = []
        self.comparison_file = self.directory + 'grid_performance.txt' 

    def add_folds(self, classifier_output, directory):
        folds = []
        for fold_index, output in enumerate(classifier_output):
            fold_directory = directory + 'fold_' + str(fold_index) + '/'
            if not os.path.isdir(fold_directory):
                os.mkdir(fold_directory)
            fold_evaluation = Eval(output, fold_directory)
            fold_evaluation.report()
            folds.append(fold_evaluation.performance)

    def add_test(self):
        pass

    def report(self):
        pass

class Eval:

    def __init__(self, clf_output, directory):
        self.ce = evaluation.ClassEvaluation()
        self.documents = clf_output[0]
        self.classifications = clf_output[1][0]
        self.model = clf_output[1][1]
        self.settings = clf_output[1][2]
        self.directory = directory
        self.performance = {}

    def save_classifier_output(self):
        for instance in self.classifications:
            self.ce.append(instance[0], instance[1])

    def assess_performance(self):
        labels = sorted(list(set(self.ce.goals)))
        for label in labels:
            label_results = \
                [
                self.ce.precision(cls = label), self.ce.recall(cls = label), self.ce.fscore(cls = label),
                self.ce.tp_rate(cls = label), self.ce.fp_rate(cls = label),
                auc([0, self.ce.fp_rate(cls = label), 1], [0, self.ce.tp_rate(cls = label), 1]),
                self.ce.tp[label] + self.ce.fn[label], self.ce.tp[label] + self.ce.fp[label], self.ce.tp[label]
                ]
            self.performance[label] = label_results
        micro_results = \
            [
            self.ce.precision(), self.ce.recall(), self.ce.fscore(),
            self.ce.tp_rate(), self.ce.fp_rate(),
            auc([0, self.ce.fp_rate(), 1], [0, self.ce.tp_rate(), 1]), 
            len(self.ce.observations), len(self.ce.observations), sum([self.ce.tp[label] for label in labels])
            ]
        self.performance["micro"] = micro_results

    def write_classifier_output(self):
        """
        File writer
        =====
        Function to document classifier output

        Parameters
        -----
        documents : list
            list with the text of each document
        predictions : dict
            keys : classifiers (str)
            values : classifier output (tup) 
                0 - prediction output : zip of lists
                    list of target labels
                    list of classifications
                    list of prediction certainties
                1 - classifier model : pkl
                2 - parameter settings : dict
        directory : str
            the directory to write classification files to

        """
        with open(self.directory + 'classifications_document.txt', 'w', encoding = 'utf-8') as out:
            out.write(' | '.join(['document', 'target', 'prediction', 'prob']) + '\n' + ('=' * 30) + '\n')
            for i, instance in enumerate(self.classifications):
                out.write(' | '.join([self.documents[i], instance[0], instance[1], instance[2]]) + '\n')
        with open(self.directory + 'classifications.txt', 'w', encoding = 'utf-8') as out:
            info = [['target', 'prediction', 'probability']]
            info.append([('-' * 20)] * 3)
            info.extend(self.classifications)
            info_str = utils.format_table(info, [20, 20, 20])
            out.write('\n'.join(info_str))
        with open(self.directory + 'classifiermodel.joblib.pkl', 'wb') as model_out:
            pickle.dump(self.model, model_out)
        if self.settings:
            with open(self.directory + 'settings.txt', 'w') as settings_out:
                settings_out.write(settings)

    def write_performance(self):
        labels = sorted(list(set(self.ce.goals)))
        labeldict = {}
        results = \
        [
        ["Cat", "Pr", "Re", "F1", "TPR", "FPR", "AUC", "Tot", "Clf", "Cor"],
        [('-' * 6)] * 10
        ]
        for i, label in enumerate(labels):
            labeldict[i] = label
            results.append([str(i)] + [str(round(val, 2)) for val in self.performance[label]])
        results.append(['Mcr'] + [str(round(val, 2)) for val in self.performance['micro']])
        with open(self.directory + 'performance.txt', 'w', encoding = 'utf-8') as out:
            out.write('legend:\n')
            for index in sorted(labeldict.keys()):
                out.write(str(index) + ' = ' + labeldict[index] + '\n')
            out.write('\n')
            results_str = utils.format_table(results, [6] * 10)
            out.write('\n'.join(results_str))       

    def report(self):
        self.save_classifier_output()
        self.assess_performance()
        self.write_classifier_output()
        self.write_performance()
