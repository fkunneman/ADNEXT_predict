#!/usr/bin/env 

import os
import pickle
import numpy
from sklearn.metrics import auc

from pynlpl import evaluation
import utils

class Reporter:

    def __init__(self, grid_directory, labels):
        self.directory = grid_directory
        self.comparison = []
        self.comparison_file = self.directory + 'grid_performance' 
        self.labels = labels

    def add_folds_test(self, classifier_output, directory):
        folds = [] 
        for fold_index, output in enumerate(classifier_output):
            fold_directory = directory + 'fold_' + str(fold_index) + '/'
            if not os.path.isdir(fold_directory):
                os.mkdir(fold_directory)
            fold_evaluation = Eval(output, self.labels, fold_directory)
            fold_evaluation.report()
            folds.append(fold_evaluation.performance)
        performance_std, performance = self.assess_performance_folds(folds)
        self.write_performance_folds(performance_std, directory)
        self.comparison.append((directory, performance))

    def add_test(self, classifier_output, directory):
        evaluation = Eval(classifier_output, self.labels, directory)
        evaluation.report()
        self.comparison.append((directory, evaluation.performance))

    def assess_performance_folds(self, folds):
        label_performance_std = {}
        label_performance = {}
        for label in self.labels:
            combined_lists = [[fold[label][i] for fold in folds] for i in range(9)]
            label_performance_std[label] = [[numpy.mean(l), numpy.std(l)] for l in combined_lists[:6]] + \
                [sum(l) for l in combined_lists[6:]]
            label_performance[label] = [numpy.mean(l) for l in combined_lists[:6]] + \
                [sum(l) for l in combined_lists[6:]]
        combined_lists_micro = [[fold['micro'][i] for fold in folds] for i in range(9)]
        label_performance_std['micro'] = [[numpy.mean(l), numpy.std(l)] for l in combined_lists_micro[:6]] + \
            [sum(l) for l in combined_lists_micro[6:]]
        label_performance['micro'] = [numpy.mean(l) for l in combined_lists_micro[:6]] + \
            [sum(l) for l in combined_lists_micro[6:]]
        return label_performance_std, label_performance

    def write_performance_folds(self, performance, directory):
        labeldict = {}
        results = \
        [
        ['Cat', 'Pr', 'Re', 'F1', 'TPR', 'FPR', 'AUC', 'Tot', 'Clf', 'Cor'],
        [('-' * 5)] + [('-' * 13)] * 6 + [('-' * 7)] * 3
        ]
        for i, label in enumerate(self.labels):
            labeldict[i] = label
            results.append([str(i)] + [" ".join([str(round(val[0], 2)), '(' + str(round(val[1], 2)) + ')']) \
                for val in performance[label][:6]] + [str(val) for val in performance[label][6:]])
        results.append(['Mcr'] + [" ".join([str(round(val[0], 2)), '(' + str(round(val[1], 2)) + ')']) \
            for val in performance['micro'][:6]] + [str(val) for val in performance['micro'][6:]])
        with open(directory + 'performance.txt', 'w', encoding = 'utf-8') as out:
            out.write('legend:\n')
            for index in sorted(labeldict.keys()):
                out.write(str(index) + ' = ' + labeldict[index] + '\n')
            out.write('\n')
            results_str = utils.format_table(results, [5] + [13] * 6 + [7] * 3)
            out.write('\n'.join(results_str))    

    def report_comparison(self):
        value_column = {'precision' : 1, 'recall' : 2, 'f1' : 3, 'fpr' : 5, 'auc' : 6}
        for label in self.labels + ['micro']:
            for value in value_column.keys():
                overview = [['setting', 'Pr', 'Re', 'F1', 'TPR', 'FPR', 'AUC', 'Tot', 'Clf', 'Cor']]
                overview.append([('-' * 60)] + [('-' * 5)] * 6 + [('-' * 6)] * 3)
                label_results = [['_'.join(performance[0].split('/')[-3:-1])] + [round(val, 2) for val in \
                    performance[1][label]] for performance in self.comparison]
                sorted_results = sorted(label_results, key = lambda k: k[value_column[value]], reverse = True)
                overview.extend(sorted_results)
                overview_str = utils.format_table(overview, [60] + [5] * 6 + [6] * 3)
                with open(self.comparison_file + '_' + label + '_' + value + '.txt', 'w', encoding = 'utf-8') as out:
                    out.write('\n'.join(overview_str))

    def report(self):
        pass

class Eval:

    def __init__(self, clf_output, labels, directory):
        self.ce = evaluation.ClassEvaluation()
        self.documents = clf_output[0]
        self.classifications = clf_output[1][0]
        self.model = clf_output[1][1]
        self.settings = clf_output[1][2]
        self.labels = labels
        self.directory = directory
        self.instances = []
        self.performance = {}

    def save_classifier_output(self):
        for instance in self.classifications:
            self.ce.append(instance[0], instance[1])

    def assess_performance(self):
        for label in self.labels:
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
            len(self.ce.observations), len(self.ce.observations), sum([self.ce.tp[label] for label in self.labels])
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
        classifications_str = []
        for instance in self.classifications:
            classifications_str.append([instance[0], instance[1], str(round(instance[2], 2))])
        with open(self.directory + 'classifications_document.txt', 'w', encoding = 'utf-8') as out:
            out.write(' | '.join(['document', 'target', 'prediction', 'prob']) + '\n' + ('=' * 30) + '\n')
            for i, instance in enumerate(classifications_str):
                out.write(' | '.join([self.documents[i], instance[0], instance[1], instance[2]]) + '\n')
        with open(self.directory + 'classifications.txt', 'w', encoding = 'utf-8') as out:
            info = [['target', 'prediction', 'probability']]
            info.append([('-' * 20)] * 3)
            info.extend(classifications_str)
            info_str = utils.format_table(info, [20, 20, 20])
            out.write('\n'.join(info_str))
        with open(self.directory + 'classifiermodel.joblib.pkl', 'wb') as model_out:
            pickle.dump(self.model, model_out)
        if self.settings:
            settings_table = []
            for parameter in self.settings.keys():
                settings_table.append([parameter, str(self.settings[parameter])])
            settings_str = utils.format_table(settings_table, [15, 10])
            with open(self.directory + 'settings.txt', 'w') as settings_out:                
                settings_out.write('\n'.join(settings_str))

    def write_performance(self):
        labeldict = {}
        results = \
        [
        ["Cat", "Pr", "Re", "F1", "TPR", "FPR", "AUC", "Tot", "Clf", "Cor"],
        [('-' * 6)] * 10
        ]
        for i, label in enumerate(self.labels):
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

    def write_confusion_matrix(self):
        confusion_matrix = self.ce.confusionmatrix()
        with open(self.directory + 'confusion_matrix.txt', 'w', encoding = 'utf-8') as out:
            out.write(confusion_matrix.__str__())
            
    def write_top_fps(self):
        for label in self.labels:
            ranked_fps = sorted([[self.documents[i], instance[0], instance[1], instance[2]] for i, instance in \
                enumerate(self.classifications) if instance[1] == label and instance[0] != instance[1]], 
                key = lambda k : k[3], reverse = True)
            with open(self.directory + label + '_ranked_fps.txt', 'w', encoding = 'utf-8') as out:
                out.write('\n'.join(['\t'.join([fp[0], str(round(fp[3], 2))]) for fp in ranked_fps]))

    def write_top_tps(self):
        for label in self.labels:
            ranked_tps = sorted([[self.documents[i], instance[0], instance[1], instance[2]] for i, instance in \
                enumerate(self.classifications) if instance[1] == label and instance[0] == instance[1]], 
                key = lambda k : k[3], reverse = True)
            with open(self.directory + label + '_ranked_tps.txt', 'w', encoding = 'utf-8') as out:
                out.write('\n'.join(['\t'.join([tp[0], str(round(tp[3], 2))]) for tp in ranked_tps]))

    def report(self):
        self.save_classifier_output()
        self.assess_performance()
        self.write_classifier_output()
        self.write_performance()
        self.write_confusion_matrix()
        self.write_top_fps()
        self.write_top_tps()
        
