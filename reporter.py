#!/usr/bin/env 

from pynlpl import evaluation
from sklearn.metrics import auc

class Reporter:

    def __init__(self):
        self.ce = evaluation.ClassEvaluation()
        self.model = []

    def add_instances(self, instances):
        for instance in instances:
            self.ce.append(instance[0], instance[1])

    def calculate_performance(self):
        results = 
        [[
        "Class", 
        "Precision", "Recall", "F1", 
        "TPR", "FPR", "AUC", 
        "Samples", "Classifications", "Correct"
        ]]
        for label in sorted(list(set(self.ce.goals))):
            label_results = 
                [
                self.ce.precision(cls = label),
                self.ce.recall(cls = label),
                self.ce.fscore(cls = label),
                self.ce.tp_rate(cls = label),
                self.ce.fp_rate(cls = label),
                auc([0, self.ce.fp_rate(cls = label), 1], [0, self.ce.tp_rate(cls = label), 1]),
                self.ce.tp[label] + self.ce.fn[label],
                self.ce.tp[label] + self.ce.fp[label],
                self.ce.tp[label]
                ]
            results.append([label] + [round(x, 2) for x in label_results])
        micro_results = 
            [
            self.ce.precision(), self.ce.recall(), self.ce.fscore(),
            self.ce.tp_rate(), self.ce.fp_rate(),
            auc([0, self.ce.fp_rate(), 1], [0, self.ce.tp_rate(), 1]), 
            len(self.ce.observations), len(self.ce.observations)
            ]
        results.append(["micro"] + [str(round(x, 2)) for x in micro_results] + [" "])
        return results
