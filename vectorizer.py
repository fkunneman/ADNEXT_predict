#!/usr/bin/env 

from collections import Counter
from scipy import sparse
import math

class Vectorizer:
    """
    Vectorizer
    =====
    Class to transform featurized train and test instances into weighted and 
    pruned vectors as input for SKlearn classification

    Parameters
    -----
    train_instances : list
        list of featurized train instances, as list with feature frequencies
    test_instances : list 
        list of featurized test instances, as list with feature frequencies
    train_labels : list
        list of labels (str) of the train instances,
        each index of a label corresponds to the index of the train instance
    test_labels : list
        list of labels (str) of the test instances,
        each index of a label corresponds to the index of the test instance
    weight : str
        names of weighting to perform
        options : 'frequency', 'binary', 'tfidf'
        default : 'frequency'
    prune : int
        top N of pruning, all features not in the top N features with the highest weight in 
        training are pruned
        default : 5000
    """
    def __init__(self, train_instances, test_instances, train_labels, weight = 'frequency', prune = 5000):
        self.metrics = {
            'frequency':    Frequency,
            'binary':       Binary,
            'tfidf':        TfIdf,
            'infogain':     InfoGain,
            'pmi':          PMI
        }
        self.metric = self.metrics[weight](train_instances, train_labels, test_instances)
        self.train = train_instances
        self.test = test_instances
        self.feature_weight = {}
        self.prune_threshold = prune

    def weight_features(self):
        """
        Feature weighter
        =====
        Function to weight features 

        Parameters
        -----
        indices : list
            List of feature indices that are to be pruned

        Transforms
        -----
        self.train : list
            Each training instance is stripped of the values in the indices parameter
        self.test : list
            Each testing instance is stripped of the values in the indices parameter
        """
        self.train, self.test, self.feature_weight = self.metric.fit_transform()

    def prune_features(self):
        """
        Instance pruner
        =====
        Function to prune every train and test instance of a set list of features

        Parameters
        -----
        indices : list
            List of feature indices that are to be pruned

        Transforms
        -----
        self.train : list
            Each training instance is stripped of the values in the indices parameter
        self.test : list
            Each testing instance is stripped of the values in the indices parameter
        """
        # select top features
        top_features = sorted(self.feature_weight, key = self.feature_weight.get, reverse = True)[:self.prune_threshold]
        # transform instances
        self.train = [[instance[index] for index in top_features] for instance in self.train]
        self.test = [[instance[index] for index in top_features] for instance in self.test] 

    def vectorize(self):
        """
        Vectorizer
        =====
        Function to weight features 

        Transforms
        -----
        self.train : list
        self.test : list
        """        
        self.weight_features()
        self.prune_features()    
        return sparse.csr_matrix(self.train), sparse.csr_matrix(self.test)

class Counts:
    """
    Counter
    =====
    Function to perform general count operations on featurized instances
    Used as parent class in several classes

    Parameters
    ------
    Instances : list
        list of featurized instances, as list with feature frequencies
    Labels : list
        list of labels (str) of the instances,
        each index of a label corresponds to the index of the instance
    """
    def __init__(self, instances, labels):
        self.instances = instances
        self.labels = labels

    def count_document_frequency(self, label = False):
        """
        Feature counter
        =====
        Function to return document counts of all features 

        Parameters
        -----
        label : str
            Choose to count the frequency that each feature co-occurs with the given label
            If False, the total document count is returned

        Returns
        -----
        document_frequency : Counter
            Counts of the number of documents or labels with which a feature occurs
            key : The feature index (int)
            value : The document / label count of the feature index (int)
        """
        document_frequency = Counter()
        if label:
            for i, instance in enumerate(self.instances):
                document_frequency.update([j for j, v in enumerate(instance) if v > 0 and self.labels[i] == label])
        else:
            for instance in self.instances:
                document_frequency.update([i for i, v in enumerate(instance) if v > 0]) #document count 
        return document_frequency

    def count_label_frequency(self):
        """
        Label counter
        =====
        Function to return counts of all document labels 

        Returns
        -----
        label_frequency : dict
            Counts of each label
            key : The label (str)
            value : The count of the label (int)
        """
        label_frequency = {}
        for label in set(self.labels):
            label_frequency[label] = self.labels.count(label)
        return label_frequency

    def count_idf(self):
        """
        Inverse Document Frequency counter
        =====
        Function to calculate the inverse document frequency of every feature

        Attributes
        -----
        num_docs : int
            The number of training instances

        Transforms
        -----
        self.idf : dict
            The idf of every feature based on the training documents
            key : The feature index
            value : The idf of the feature index
        """
        idf = dict.fromkeys(range(len(self.instances[0])), 0) # initialize for all features
        num_docs = len(self.instances)
        feature_counts = self.count_document_frequency()
        for feature in feature_counts.keys():
            idf[feature] = math.log((num_docs / feature_counts[feature]), 10)
        return idf

class Frequency(Counts):

    def __init__(self, train_instances, labels, test_instances):
        Counts.__init__(self, train_instances, labels)
        self.train_instances = train_instances
        self.test_instances = test_instances
        self.feature_frequency = {}

    def fit(self):
        self.feature_frequency = Counts.count_document_frequency(self)

    def transform(self):
        return self.train_instances, self.test_instances, self.feature_frequency

    def fit_transform(self):
        self.fit()
        return self.transform()

class Binary(Counts):

    def __init__(self, train_instances, labels, test_instances):
        Counts.__init__(self, train_instances, labels)
        self.train_instances = train_instances
        self.test_instances = test_instances
        self.feature_frequency = {}

    def fit(self):
        self.feature_frequency = Counts.count_document_frequency(self)        

    def transform(self):
        self.train_instances = [[1 if i > 0 else 0 for i in instance] for instance in self.train_instances]
        self.test_instances = [[1 if i > 0 else 0 for i in instance] for instance in self.test_instances]
        return self.train_instances, self.test_instances, self.feature_frequency

    def fit_transform(self):
        self.fit()
        return self.transform()

class TfIdf(Counts):

    def __init__(self, train_instances, labels, test_instances):
        Counts.__init__(self, train_instances, labels)
        self.train_instances = train_instances
        self.test_instances = test_instances
        self.idf = {}

    def fit(self):
        self.idf = Counts.count_idf(self)        

    def transform(self):
        self.train_instances = [[v * self.idf[i] if i > 0 else 0 for i, v in enumerate(instance)] \
            for instance in self.train_instances]
        self.test_instances = [[v * self.idf[i] if i > 0 else 0 for i, v in enumerate(instance)] \
            for instance in self.test_instances]
        return self.train_instances, self.test_instances, self.idf

    def fit_transform(self):
        self.fit()
        return self.transform()

class InfoGain(Counts):
    """
    Information Gain Weighter
    =====
    Class to calculate the information gain for each feature, and weight instances accordingly

    Parameters
    -----
    train_instances : list
        list of featurized train instances, as list with feature frequencies
    labels : list
        list of labels (str) of the train instances,
        each index of a label corresponds to the index of the train instance
    test_instances : list 
        list of featurized test instances, as list with feature frequencies

    Parent class
    -----
    Counts : class to perform frequency counts
    """

    def __init__(self, train_instances, labels, test_instances):
        Counts.__init__(self, train_instances, labels)
        self.train_instances = train_instances
        self.labels = labels
        self.test_instances = test_instances
        self.feature_infogain = dict.fromkeys(range(len(self.train_instances[0])), 0) # initialize for all features
    
    def calculate_label_feature_frequency(self, labels):
        """
        Frequency calculator
        =====
        Function to calculate the frequency of each feature in combination with specific labels

        Parameters
        -----
        labels : list
            list of labels (str) of the train instances

        Returns
        -----
        label_feature_frequency : dict of dicts
            key1 : label, str
            key2 : feature index, int
            value : number of times the two co-occur on the document level, list
        """
        label_feature_frequency = {}
        for label in labels:
            label_feature_frequency[label] = Counts.count_document_frequency(self, label)
        return label_feature_frequency

    def calculate_entropy(self, probs):
        """
        Entropy calculator
        =====
        Function to calculate the entropy based on a list of probabilities

        Parameters
        -----
        probs : list
            list of probabilities

        Returns
        -----
        entropy : float
        """
        entropy = -sum([prob * math.log(prob, 2) for prob in probs if prob != 0])
        return entropy

    def calculate_initial_entropy(self, len_instances, label_frequency):
        """
        Initial entropy calculator
        =====
        Function to calculate the initial entropy of the different labels of a set of instances

        Parameters
        -----
        len_instances : int
            The number of instances
        label_frequency : dict
            key : label, str
            value : frequency, int

        Returns
        -----
        initial_entropy : float
        """
        label_probability = [(label_frequency[label] / len_instances) for label in label_frequency.keys()]
        initial_entropy = self.calculate_entropy(label_probability)
        return initial_entropy

    def calculate_positive_feature_entropy(self, feature, len_instances, feature_frequency, label_feature_frequency):
        """
        Positive feature entropy calculator
        =====
        Function to calculate the entropy for all instances with the target feature

        Parameters
        -----
        feature : int
            the index of the feature
        len_instances : int
            The number of instances
        feature_frequency : dict
            key : feature index, int
            value: feature frequency, int
        label_feature_frequency : dict of dicts
            key1 : label, str
            key2 : feature index, int
            value : number of times the two co-occur on the document level, list

        Returns
        -----
        positive_entropy : float
        """
        frequency = feature_frequency[feature]
        feature_probability = frequency / len_instances
        feature_label_probs = [(label_feature_frequency[label][feature] / frequency) for label in label_feature_frequency.keys()]
        positive_entropy = self.calculate_entropy(feature_label_probs) * feature_probability
        return positive_entropy

    def calculate_negative_feature_entropy(self, feature, len_instances, feature_frequency, label_frequency, label_feature_frequency):
        """
        Negative feature entropy calculator
        =====
        Function to calculate the entropy for all instances without the target feature

        Parameters
        -----
        feature : int
            the index of the feature
        len_instances : int
            The number of instances
        feature_frequency : dict
            key : feature index, int
            value: feature frequency, int
        label_frequency : dict
            key : label, str
            value : frequency, int
        label_feature_frequency : dict of dicts
            key1 : label, str
            key2 : feature index, int
            value : number of times the two co-occur on the document level, list

        Returns
        -----
        negative_entropy : float
        """
        inverse_frequency = len_instances - feature_frequency[feature]
        negative_probability = inverse_frequency / len_instances
        negative_label_probabilities = [((label_frequency[label] - label_feature_frequency[label][feature]) / inverse_frequency) \
            for label in label_frequency.keys()]
        negative_entropy = self.calculate_entropy(negative_label_probabilities) * negative_probability
        return negative_entropy

    def fit(self):
        """
        Infogain calculator
        =====
        Function to calculate the information gain of each feature

        Transforms
        -----
        self.feature_infogain : dict
            key : feature index, int
            value : information gain, float
        """
        # some initial calculations
        len_instances = len(self.instances)
        feature_frequency = Counts.count_document_frequency(self)
        label_frequency = Counts.count_label_frequency(self)
        label_feature_frequency = self.calculate_label_feature_frequency(label_frequency.keys())
        initial_entropy = self.calculate_initial_entropy(len_instances, label_frequency)
        # assign infogain values to each feature
        for feature in feature_frequency.keys():
            entropy1 = self.calculate_positive_feature_entropy(feature, len_instances, feature_frequency, label_feature_frequency)
            entropy0 = self.calculate_negative_feature_entropy(feature, len_instances, feature_frequency, label_frequency, 
                label_feature_frequency)
            after_entropy = entropy1 + entropy0 
            self.feature_infogain[feature] = initial_entropy - after_entropy

    def transform(self):
        """
        Instance transformer
        -----
        Function to weight each training and test instance by information gain

        Transforms
        -----
        self.train_instances : list
            before : list of featurized train instances, as list with feature frequencies
            after : list of featurized train instances, 
                weighted as feature information gain if feature is present, 0 otherwise
        self.test_instances : list
            before : list of featurized test instances, as list with feature frequencies
            after : list of featurized test instances, 
                weighted as feature information gain if feature is present, 0 otherwise

        Returns
        -----
        self.train_instances : list
            list of featurized train instances, 
                weighted as feature information gain if feature is present, 0 otherwise
        self.test_instances : list
            list of featurized test instances, 
                weighted as feature information gain if feature is present, 0 otherwise
        self.feature_infogain : dict
            key : feature index, int
            value : information gain, float
        """

        self.train_instances = [[self.feature_infogain[i] if i > 0 else 0 for i, v in enumerate(instance)] \
            for instance in self.train_instances]
        self.test_instances = [[self.feature_infogain[i] if i > 0 else 0 for i, v in enumerate(instance)] \
            for instance in self.test_instances]
        return self.train_instances, self.test_instances, self.feature_infogain

    def fit_transform(self):
        """
        Fit transform
        =====
        Function to perform the fit and transform sequence

        Returns
        -----
        self.train_instances : list
            list of featurized train instances, 
                weighted as feature information gain if feature is present, 0 otherwise
        self.test_instances : list
            list of featurized test instances, 
                weighted as feature information gain if feature is present, 0 otherwise
        self.feature_infogain : dict
            key : feature index, int
            value : information gain, float
        """
        self.fit()
        return self.transform()

class PMI(Counts):

    def __init__(self, train_instances, labels, test_instances):
        pass
