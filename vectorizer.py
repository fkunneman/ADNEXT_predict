
#!/usr/bin/env 

from collections import Counter
from scipy import sparse
import numpy
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
        list of featurized train instances, as sparse.csr_matrix with feature frequencies
    test_instances : list 
        list of featurized test instances, as sparse.csr_matrix with feature frequencies
    train_labels : list
        list of labels (str) of the train instances,
        each index of a label corresponds to the index of the train instance
    weight : str
        names of weighting to perform
        options : 'frequency', 'binary', 'tfidf', 'infogain', 'pmi'
        default : 'frequency'
    prune : int
        top N features to select, all features not in the top N features with the highest weight are pruned
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
        self.top_features = []
        self.prune_threshold = prune

    def weight_features(self):
        """
        Feature weighter
        =====
        Function to calculate feature weights and transform document vectors accordingly

        Transforms
        -----
        self.train : list
            Each train instance is weighted by the selected metric
        self.test : list
            Each test instance is weighted by the selected metric
        self.feature_weight : dict
            key : feature index (int)
            value : feature weight or count (int / float)
        """
        self.train, self.test, self.feature_weight = self.metric.fit_transform()

    def prune_features(self):
        """
        Feature pruner
        =====
        Function to prune every train and test instance of the top N features with the highest weight

        Transforms
        -----
        self.train : list
            Each train instance is stripped of the feature indices not in the top N weighted features
        self.test : list
            Each test instance is stripped of the feature indices not in the top N weighted features
        """
        # select top features
        self.top_features = sorted(self.feature_weight, key = self.feature_weight.get, reverse = True)[:self.prune_threshold]
        # transform instances
        self.train = self.train[:, self.top_features]
        self.test = self.test[:, self.top_features]

    def vectorize(self):
        """
        Vectorizer
        =====
        Function to weight instances

        Returns
        -----
        self.train : list
            scipy csr_matrix of weighted train vectors
        self.test : list
            scipy csr_matrix of weighted test vectors
        """        
        self.weight_features()
        print('pruning features')
        self.prune_features()    
        return sparse.csr_matrix(self.train), sparse.csr_matrix(self.test), self.top_features, [str(self.feature_weight[i]) for i in self.top_features]

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
        if label:
            target_instances = self.instances[list(numpy.where(numpy.array(self.labels) == label)[0])]
        else:
            target_instances = self.instances
        feature_indices = range(self.instances.shape[1])
        feature_counts = target_instances.sum(axis = 0).tolist()[0]
        document_frequency = dict(zip(feature_indices, feature_counts))
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

class Frequency(Counts):
    """
    Frequency Weighter
    =====
    Class to count the document frequency of all features
    Instances already represent feature frequency, and are left unaltered

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
        self.test_instances = test_instances
        self.feature_frequency = {}

    def fit(self):
        """
        Frequency fitter
        =====

        Transforms
        -----
        self.feature_frequency : dict
            dictionary of the frequency per feature
            key : feature index (int)
            value : feature document frequency (int)
        """
        self.feature_frequency = Counts.count_document_frequency(self)

    def transform(self):
        """
        Instance transformer
        =====

        Returns
        -----
        train_instances : list
            list of featurized train instances, as list with feature frequencies
        test_instances : list 
            list of featurized test instances, as list with feature frequencies
        self.feature_frequency : dict
            dictionary of the frequency per feature
            key : feature index (int)
            value : feature document frequency (int)
        """
        return self.train_instances, self.test_instances, self.feature_frequency

    def fit_transform(self):
        """
        Fit transform
        =====
        Function to perform the fit and transform sequence

        Returns
        -----
        train_instances : list
            list of featurized train instances, as list with feature frequencies
        test_instances : list 
            list of featurized test instances, as list with feature frequencies
        self.feature_frequency : dict
            dictionary of the frequency per feature
            key : feature index (int)
            value : feature document frequency (int)
        """
        self.fit()
        return self.transform()

class Binary(Counts):
    """
    Binary Weighter
    =====
    Class to count the document frequency of all features and convert instances to binary vectors

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
        self.test_instances = test_instances
        self.feature_frequency = {}

    def fit(self):
        """
        Frequency fitter
        =====

        Transforms
        -----
        self.feature_frequency : dict
            dictionary of the frequency per feature
            key : feature index (int)
            value : feature document frequency (int)
        """
        self.feature_frequency = Counts.count_document_frequency(self)        

    def transform(self):
        """
        Instance transformer
        =====

        Transforms
        -----
        self.train_instances : list
            Feature frequency is transformed into a binary value
        self.test_instances : list
            Feature frequency is transformed into a binary value

        Returns
        -----
        train_instances : list
            list of featurized train instances, as list with binary values
        test_instances : list 
            list of featurized test instances, as list with binary values
        self.feature_frequency : dict
            dictionary of the frequency per feature
            key : feature index (int)
            value : feature document frequency (int)
        """
        binary_values_train = numpy.array([1 for cell in self.train_instances.data])
        binary_values_test = numpy.array([1 for cell in self.test_instances.data])
        self.train_instances = sparse.csr_matrix((binary_values_train, self.train_instances.indices, self.train_instances.indptr), shape = self.train_instances.shape)
        self.test_instances = sparse.csr_matrix((binary_values_test, self.test_instances.indices, self.test_instances.indptr), shape = self.test_instances.shape)
        return self.train_instances, self.test_instances, self.feature_frequency

    def fit_transform(self):
        """
        Fit transform
        =====
        Function to perform the fit and transform sequence

        Returns
        -----
        train_instances : list
            list of featurized train instances, as list with binary values
        test_instances : list 
            list of featurized test instances, as list with binary values
        self.feature_frequency : dict
            dictionary of the frequency per feature
            key : feature index (int)
            value : feature document frequency (int)
        """
        self.fit()
        return self.transform()

class TfIdf(Counts):
    """
    Tfidf Weighter
    =====
    Class to calculate the inverse document frequency of all features and weight instances by tfidf

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
        self.test_instances = test_instances
        self.idf = {}

    def fit(self):
        """
        Tfidf fitter
        =====

        Transforms
        -----
        self.idf : dict
            dictionary with the idf per feature
            key : feature index (int)
            value : feature idf (float)
        """
        self.idf = Counts.count_idf(self)        

    def transform(self):
        """
        Instance transformer
        =====

        Transforms
        -----
        self.train_instances : list
            Feature frequency is replaced by tfidf
        self.test_instances : list
            Feature frequency is replaced by tfidf

        Returns
        -----
        train_instances : list
            list of featurized train instances, weighted with tfidf values
        test_instances : list 
            list of featurized test instances, weighted with tfidf values
        self.feature_frequency : dict
            dictionary with the idf per feature
            key : feature index (int)
            value : feature idf (float)
        """
        feature_idf_ordered = sparse.csr_matrix([self.idf[feature] for feature in sorted(self.idf.keys())])
        self.train_instances = self.train_instances.multiply(feature_idf_ordered)
        self.test_instances = self.test_instances.multiply(feature_idf_ordered)
        return self.train_instances, self.test_instances, self.idf

    def fit_transform(self):
        """
        Fit transform
        =====
        Function to perform the fit and transform sequence

        Returns
        -----
        train_instances : list
            list of featurized train instances, weighted with tfidf values
        test_instances : list 
            list of featurized test instances, weighted with tfidf values
        self.feature_frequency : dict
            dictionary with the idf per feature
            key : feature index (int)
            value : feature idf (float)
        """
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
        self.feature_infogain = dict.fromkeys(range(self.instances.shape[1]), 0) # initialize for all features
    
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
        if frequency > 0:
            feature_probability = frequency / len_instances
            feature_label_probs = []
            for label in label_feature_frequency.keys():
                if label_feature_frequency[label][feature] > 0:
                    feature_label_probs.append(label_feature_frequency[label][feature] / frequency)
                else:
                    feature_label_probs.append(0)
            positive_entropy = self.calculate_entropy(feature_label_probs) * feature_probability
        else:
            positive_entropy = 0
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
        len_instances = self.instances.shape[0]
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
        binary_vectorizer = Binary(self.train_instances, self.labels, self.test_instances)
        train_instances_binary, test_instances_binary, feature_frequency = binary_vectorizer.fit_transform()
        feature_infogain_ordered = numpy.array([self.feature_infogain[feature] for feature in sorted(self.feature_infogain.keys())])        
        self.train_instances = feature_infogain_ordered * train_instances_binary.toarray()
        self.test_instances = feature_infogain_ordered * test_instances_binary.toarray()
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
