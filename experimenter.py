#!/usr/bin/env 

import time

import featurizer
import lcs_classifier
#import sklearn_classifier
import reporter
import utils

class ExperimentGrid:

    def __init__(self, features, featurefilter, classifiers, directory):
        self.features = features
        self.featurefilter = featurefilter # list of tokens to be removed
        self.classifiers = classifiers
        self.directory = directory
        self.featurized = []

    def set_features(self, train, test = False, grid = 'low'):
        if grid == 'low': #only one setting
            settings = [self.features]
        for setting in settings:
            text = self.train['text']
            frog = self.train['frog'] 
            if self.test:
                text += self.test['text']
                frog += self.test['frog']      
            featurizer = featurizer.Featurizer(text, frog, setting)
            vectors, vocabulary = featurizer.fit_transform()
            train = list(zip(self.train['labels'], vectors[:len(self.train['text'])]))
            if self.test:
                test = list(zip(self.test['labels'], vectors[len(self.train['text']):]))
            else:
                test = False
            self.featurized.append([train, test, vocabulary, setting.keys()])


    def experiment(self):
        experimentlog = self.directory + "log.txt"
        overview = self.directory + "overview.txt"
        expindex = 1
        for setting in self.featurized:
            train, test, vocabulary, featuretypes = setting
            features = "-".join(featuretypes)
            for classifier in self.classifiers:
                expdir = self.directory + "exp" + str(expindex) + ".txt"
                os.mkdir(expdir)
                expname = features + "_" + classifier
                print("classifying", expname)
                with open(experimentlog, "a") as el:
                    el.write(str(time.asctime()) + "\t" + expname)
                if classifier == "lcs":
                    clf = lcs_classifier.LCS_classifier(train, test, expdir, vocabulary)
                    rep = reporter.Reporter()
                    rep.add_instances(clf.classifications)
                    performance = reporter.calculate_performance()
                    with open(expdir + "results.txt", "w") as resultsfile:
                        results.file.write(
                            "\n".join
                            (
                                ["\t".join([str(x) for x in label]) for label in performance])
                            )
                    with open(overview, "a") as ov:
                        ov.write("\t".join([expname] + performance[-1]))
