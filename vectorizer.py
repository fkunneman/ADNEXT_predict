
#!/usr/bin/env 

from collections import Counter
from scipy import sparse
import numpy
import math

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
    def __init__(self, instances):
        self.instances = instances

    def count_document_frequency(self, labels = False):
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
        if labels:
            target_instances = self.instances[list(numpy.where(numpy.array(labels) == label)[0])]
        else:
            target_instances = self.instances
        feature_indices = range(self.instances.shape[1])
        feature_counts = target_instances.sum(axis = 0).tolist()[0]
        document_frequency = dict(zip(feature_indices, feature_counts))
        return document_frequency

    def count_label_frequency(self, labels):
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
        for label in set(labels):
            label_frequency[label] = labels.count(label)
        return label_frequency

    def count_label_feature_frequency(self, labels):
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
            label_feature_frequency[label] = self.count_document_frequency(labels)
        return label_feature_frequency

    def count_idf(self):
        """
        Inverse Document Frequency counter
        =====
        Function to calculate the inverse document frequency of every feature

        Returns
        -----
        idf : dict
            The idf of every feature based on the training documents
            key : The feature index
            value : The idf of the feature index
        """
        idf = dict.fromkeys(range(self.instances.shape[1]), 0) # initialize for all features
        num_docs = self.instances.shape[0]
        feature_counts = self.count_document_frequency()
        for feature in feature_counts.keys():
            idf[feature] = math.log((num_docs / feature_counts[feature]), 10) if feature_counts[feature] > 0 else 0
        return idf

def return_document_frequency(instances, labels):

    cnt = Counts(instances)
    document_frequency = cnt.count_document_frequency()
    return document_frequency

def return_idf(instances, labels):

    cnt = Counts(instances)
    idf = cnt.count_idf()
    return idf 

def return_infogain(instances, labels):
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
    infogain = dict.fromkeys(range(instances.shape[1]), 0)
    cnt = Counts(instances)
    len_instances = instances.shape[0]
    feature_frequency = cnt.count_document_frequency()
    label_frequency = cnt.count_label_frequency(labels)
    label_feature_frequency = cnt.count_label_feature_frequency(labels)
    label_probability = [(label_frequency[label] / len_instances) for label in label_frequency.keys()]
    initial_entropy = -sum([prob * math.log(prob, 2) for prob in label_probability if prob != 0])
    # assign infogain values to each feature
    for feature in feature_frequency.keys():
        # calculate positive entropy
        frequency = feature_frequency[feature]
        if frequency > 0:
            feature_probability = frequency / len_instances
            positive_label_probabilities = []
            for label in labels:
                if label_feature_frequency[label][feature] > 0:
                    positive_label_probabilities.append(label_feature_frequency[label][feature] / frequency)
                else:
                    positive_label_probabilities.append(0)
            positive_entropy = -sum([prob * math.log(prob, 2) for prob in positive_label_probabilities if prob != 0])
        else:
            positive_entropy = 0
        # calculate negative entropy
        inverse_frequency = len_instances - feature_frequency[feature]
        negative_probability = inverse_frequency / len_instances
        negative_label_probabilities = [((label_frequency[label] - label_feature_frequency[label][feature]) / inverse_frequency) for label in labels]
        negative_entropy = -sum([prob * math.log(prob, 2) for prob in negative_label_probabilities if prob != 0])
        # based on positive and negative entropy, calculate final entropy
        final_entropy = positive_entropy - negative_entropy
        infogain[feature] = initial_entropy - after_entropy
    return infogain

def return_binary_vectors(instances):

    binary_vectors = numpy.array([1 for cell in instances.data]) 
    return binary_vectors

def return_tfidf_vectors(instances, idfs):

    feature_idf_ordered = sparse.csr_matrix([idfs[feature] for feature in sorted(idfs.keys())])
    tfidf_vectors = instances.multiply(feature_idf_ordered)
    return tfidf_vectors

def return_infogain_vectors(instances, infogain):

    infogain_ordered = sparse.csr_matrix([infogain[feature] for feature in sorted(infogain.keys())])
    instances_binary = return_binary_vectors(instances)
    infogain_vectors = instances_binary.multiply(infogain_ordered)
    return infogain_vectors

def prune_features(feature_weights, prune):
    
    top_features = sorted(feature_weights, key = feature_weights.get, reverse = True)[:prune]
    return top_features

def compress_vectors(instances, top_features):
    compressed_vectors = instances[:, top_features]
    return compressed_vectors
