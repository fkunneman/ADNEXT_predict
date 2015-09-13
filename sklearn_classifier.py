#!/usr/bin/env

from sklearn import preprocessing
from sklearn import svm, naive_bayes, tree
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OutputCodeClassifier
from scipy import sparse

import utils

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
            'nb':               NB_classifier,
            'svm':              SVM_classifier,
            'tree':             Tree_classifier,
            'ensemble_clf':     EnsembleClf_classifier
        }
        self.helpers = [modules[key](le, **clfs[key]) for key in clfs.keys()]

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
            print(helper, 'classification')
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
    def __init__(self, le, **kwargs):
        self.name = 'nb'
        self.le = le
        self.clf = False
        self.settings = False

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
        self.clf.fit(train['instances'], self.le.transform(train['labels']))

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
            prediction = self.clf.predict(instance)[0]
            predictions.append(prediction)
            predictions_prob.append(self.clf.predict_proba(instance)[0][prediction])
        output = list(zip(test['labels'], self.le.inverse_transform(predictions), predictions_prob))
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
        print('Fitting')
        self.fit(train)
        print('Transforming')
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
    def __init__(self, le, **kwargs):
        self.name = 'tree'
        self.le = le
        self.clf = False
        self.settings = False

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
        self.clf.fit(train['instances'], self.le.transform(train['labels']))

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
            prediction = self.clf.predict(instance)[0]
            predictions.append(prediction)
            predictions.append(self.clf.predict(instance))
            predictions_prob.append(self.clf.predict_proba(instance)[0][prediction])
        output = list(zip(test['labels'], self.le.inverse_transform(predictions), predictions_prob))
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
        print('Fitting')
        self.fit(train)
        print('Transforming')
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
    def __init__(self, le, **kwargs):
        self.name = 'svm'
        self.le = le
        # Different settings are required if there are more than two classes
        if len(self.le.classes_) > 2:
            self.multi = True
        else:
            self.multi = False
        self.clf = False
        self.settings = {}
        self.c = kwargs['C'] if 'C' in kwargs.keys() else False
        self.kernel = kwargs['kernel'] if 'kernel' in kwargs.keys() else False
        self.gamma = kwargs['gamma'] if 'gamma' in kwargs.keys() else False
        self.degree = kwargs['degree'] if 'degree' in kwargs.keys() else False
        self.approach = kwargs['params'] if 'params' in kwargs.keys() else 'default'

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
        # scale data
        # self.min_max_scaler = preprocessing.MinMaxScaler()
        # train_minmax = self.min_max_scaler.fit_transform(train['instances'].toarray())
        # train_minmax = sparse.csr_matrix(train_minmax)
        if self.multi:
            params = ['estimator__C', 'estimator__kernel', 'estimator__gamma', 'estimator__degree']
        else:
            params = ['C', 'kernel', 'gamma', 'degree']
        if self.approach == 'paramsearch':
            # try different parameter settings for an svm outputcode classifier
            grid_values = [
                [0.001, 0.005, 0.01, 0.5, 1, 5, 10, 50, 100, 500, 1000],
                ['linear', 'rbf', 'poly'], 
                [0.0005, 0.002, 0.008, 0.032, 0.128, 0.512, 1.024, 2.048],
                [1, 2, 3, 4]
            ]
            param_grid = {}
            param_grid[params[0]] = self.c if self.c else grid_values[0]
            param_grid[params[1]] = self.kernel if self.kernel else grid_values[1]
            param_grid[params[2]] = self.gamma if self.gamma else grid_values[2]
            param_grid[params[3]] = self.degree if self.degree else grid_values[3]
            model = svm.SVC(probability=True)
            if self.multi:
                model = OutputCodeClassifier(model)
            paramsearch = RandomizedSearchCV(model, param_grid, cv = 5, verbose = 3, n_iter = 10, n_jobs = 10, pre_dispatch = 4) 
            paramsearch.fit(train['instances'], self.le.transform(train['labels']))
            self.settings = paramsearch.best_params_
        elif self.approach == 'default':
            self.settings[params[0]] = self.c if self.c else 1
            self.settings[params[1]] = self.kernel if self.kernel else 'linear'
            self.settings[params[2]] = self.gamma if self.gamma else 1 / train['instances'].shape[1] # number of features
            self.settings[params[3]] = self.degree if self.degree else 3
        # train an SVC classifier with the settings that led to the best performance
        self.clf = svm.SVC(
           probability = True, 
           C = self.settings[params[0]],
           kernel = self.settings[params[1]],
           gamma = self.settings[params[2]],
           degree = self.settings[params[3]],
           cache_size = 1000,
           class_weight = 'auto',
           verbose = 2
           )      
        if self.multi:
            self.clf = OutputCodeClassifier(self.clf)
        self.clf.fit(train['instances'], self.le.transform(train['labels']))

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
            prediction = self.clf.predict(instance)[0]
            predictions.append(prediction)
            if self.multi:
                predictions_prob.append(0)
            else:
                predictions_prob.append(self.clf.predict_proba(instance)[0][prediction])
        output = list(zip(test['labels'], self.le.inverse_transform(predictions), predictions_prob))
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
        print('Fitting')
        self.fit(train)
        print('Transforming')
        output = (self.transform(test), self.clf, self.settings)
        return output

class EnsembleClf_classifier:
    """
    Ensemble classification Classifier
    =====
    Class for ensemble classification, leveraging the judgements of multiple classifiers

    """
    def __init__(self, le, **kwargs):
        self.name = 'ensemble_clf'
        self.le = le
        self.modules = {
            'nb':               NB_classifier,
            'svm':              SVM_classifier,
            'tree':             Tree_classifier
        }
        self.helpers = kwargs['helpers']
        self.assessor = kwargs['assessor']
        self.approach = kwargs['approach']

    def add_classification_features(self, clfs, test):
        instances = test['instances'].copy()
        if self.approach == 'classifications_only':
            instances = sparse.csr_matrix([[]] * instances.shape[0])   
        for clf in clfs:
            output = clf.transform(test)
            classifications = self.le.transform([x[1] for x in output])
            classifications_csr = sparse.csr_matrix([[classifications[i]] for i in range(classifications.size)])
            if self.approach == 'classifications_only':
                if instances.shape[1] == 0:
                    instances = classifications_csr
                else:
                    instances = sparse.hstack((instances, classifications_csr))
            else:
                instances = sparse.hstack((instances, classifications_csr))
        return (instances)

    def fit(self, train):
        """

        """
        # train helper classifiers
        self.clfs = [value(self.le, **self.helpers[key]) for key, value in self.modules.items() if key in self.helpers.keys()]
        for clf in self.clfs:
            clf.fit(train)
        # extend training data with classification features
        # make folds
        folds = utils.return_folds(len(train['labels']))
        # add classifier features
        new_instances = []
        new_labels = []
        for fold in folds:
            fold_train = {
                'instances' : sparse.vstack([train['instances'][i] for i in fold[0]]), 
                'labels' : [train['labels'][i] for i in fold[0]]
            }
            fold_test = {
                'instances' : sparse.vstack([train['instances'][i] for i in fold[1]]), 
                'labels' : [train['labels'][i] for i in fold[1]]
            }
            clfs = [value(self.le, **self.helpers[key]) for key, value in self.modules.items() if key in self.helpers.keys()]
            for clf in clfs:
                clf.fit(fold_train)
            test_instances = self.add_classification_features(clfs, fold_test)
            new_instances.append(test_instances)
            new_labels.extend(fold_test['labels'])
        new_instances_folds = sparse.csr_matrix(sparse.vstack(new_instances))
        train_classifications = {'instances' : new_instances_folds, 'labels' : new_labels}
        # train ensemble classifier
        self.ensemble = self.modules[self.assessor[0]](self.le, **self.assessor[1])        
        self.ensemble.fit(train_classifications)

    def transform(self, test):
        """

        """
        # extend test data with classification features
        test_instances = self.add_classification_features(self.clfs, test)
        test_all = {'instances' : sparse.csr_matrix(test_instances), 'labels' : test['labels']}
        # make predictions
        output = self.ensemble.transform(test_all)
        return (output)

    def fit_transform(self, train, test):
        """

        """
        print('Fitting')
        self.fit(train)
        print('Transforming')
        tf = self.transform(test)
        output = (tf, self.ensemble.clf, self.ensemble.settings)
        return output

class LCS_classifier:
    """
    Classification by LCS balanced winnow
    ======
    Interface to LCS classifier

    Parameters
    ------
    train : list 
        featurized training instances
    test : list
        featurized test instances
    directory : str
        directory in which classificationfiles are written 
        experiment itself is performed in current directory
    vocabulary : dict
        dictionary with a mapping between indices and features

    Attributes
    -----
    train : holder of 'train' parameter
    test : holder of 'test' parameter
    expdir : holder of 'expdir' parameter
    vocabulary : holder of 'vocabulary' parameter
    convert_features : function call
        converts featurized features to format used in LCS classification

    Examples
    -----
    Interactive:

    >>> reader = Datareader(max_n=1000)
    >>> reader.set('blogs.csv')
    >>> docs = reader.rows
    >>> reader.set_rows(docs)

    """

    def __init__(self, train, test, directory, vocabulary):
        """
        """
        self.train = train
        self.test = test
        self.expdir = directory
        self.vocabulary = vocabulary
        self.convert_features()
        self.targets = {}
        self.classifications = []



    def experiment(self):
        if self.test:
            trainparts = self.prepare(self.train)
            testparts = self.prepare(self.test)
            test_tuples = [instance.split() for instance in testparts]
            self.targets = dict((filename, target) for filename, target in test_tuples)
            self.classify(trainparts, testparts, self.expdir)
        else: 
            print("preparing files")
            parts = self.prepare(self.train)
            parts_tuples = [instance.split() for instance in parts]
            self.targets = dict((filename, target) for filename, target in parts_tuples)
            # perform tenfold on train
            folds = utils.return_folds(parts,10)
            for i, fold in enumerate(folds):
                expdir = self.expdir + "fold_" + str(i) + "/"
                os.mkdir(expdir)
                train, test = fold
                #self.targets.update()
                print(len(train), len(test))
                self.classify(train, test, expdir)

    def prepare(self, data):
        """
        LCS file writer
        =====
        function to write instances to files
        
        Parameters
        -----
        partsfile : str
            the file to write filenames and labels to

        Returns
        -----
        parts : list
            list of references to files, along with the label
        """
        parts = []
        # make directory to write files to
        self.filesdir = self.expdir + "files/"
        #os.mkdir(self.filesdir)
        # make chunks of 25000 from the data
        if len(data) > 25000:
            chunks = [list(t) for t in zip(*[iter(data)]*int(round(len(data) / 25000), 0))]
        else:
            chunks = [data]
        for i, chunk in enumerate(chunks):
            # make subdirectory
            subpart = "sd" + str(i) + "/"
            subdir = self.filesdir + subpart
            #os.mkdir(subdir)
            for j, instance in enumerate(chunk):
                zeros = 5 - len(str(j))
                filename = subpart + ('0' * zeros) + str(j) + ".txt"
                label = instance[0]
                features = instance[1]
                #with open(self.filesdir + filename, 'w', encoding = 'utf-8') as outfile: 
                #    outfile.write("\n".join(features))
                parts.append(filename + " " + label)
        return parts

    def classify(self, trainparts, testparts, expdir):
        """
        LCS classifier
        =====
        Function to call upon the LCS classifier to train and test on the 
            partsfiles.
        Classifier needs to be properly set-up
        Performs classification in current directory

        Parameters
        -----
        trainparts : list
            all train instances as line with a file reference and label
        testparts : list
            all test instances as line with a file reference and label
        """
        with open("train", "w", encoding = "utf-8") as train:
            train.write("\n".join(trainparts))
        with open("test", "w", encoding = "utf-8") as test:
            test.write("\n".join(testparts))
        self.write_config()
        os.system("lcs --verbose")
        self.extract_performance()
        os.system("mv * " + expdir)

    def extract_performance(self):
        with open('test.rnk') as rnk:
            for line in rnk.readlines():
                tokens = line.strip().split()
                filename = tokens[0].strip()
                classification, score = tokens[1].split()[0].split(":")
                classification = classification.replace("?","")
                self.classifications.append([self.targets[filename], classification, score])

    def convert_features(self):
        """
        Feature converter
        =====
        Function to convert vectorized features to a set of tokens 
            needed for LCS format

        Calls
        -----
        self.return_featurelist
            function the returns featurenames of a single instance

        Alters
        -----
        self.train
        self.test
        """
        new_train = []
        for instance in self.train:
            new_train.append(self.return_featurelist(instance))
        self.train = new_train
        if self.test:
            new_test = []
            for instance in self.test:
                new_test.append(self.return_featurelist(instance))
            self.test = new_test

    def return_featurelist(self, instance):
        """
        Function to retrieve the vocabulary feature from feature indices

        Parameters
        -----
        instance : numpy array
            the feature vector

        Uses
        -----
        self.vocabulary : dict
            matches feature indices to their proper name

        Returns
        -----
        features : list
            list of the proper name of each feature, occuring as often as 
            it's mentioned in the text
        """
        feature_freqs = [(i,f) for i,f in enumerate(instance[1]) if f > 0]
        features = []
        for feature in feature_freqs:
            features += [self.vocabulary[feature[0]]] * feature[1]
        return [instance[0], features]

    def write_config(self):
        fileschunks = self.filesdir.split("/")
        files = "/".join(fileschunks[:-1]) + "/./" + fileschunks[-1]
        current = os.getcwd()
        current_chunks = current.split("/")
        data = "/".join(current_chunks) + "/./data"
        index = "/".join(current_chunks) + "/./index"
        config = "\n".join\
            ([
            "docprof.normalise=NONE",
            "general.analyser=nl.cs.ru.phasar.lcs3.analyzers.FreqAnalyzer",
            "general.autothreshold=true",
            "general.data=" + data,
            "general.files=" + files,
            "general.index=" + index,
            "general.numcpus=16",
            "general.termstrength=BOOL", # hier een parameter
            "gts.mindf=1",
            "gts.mintf=6",
            "lts.algorithm=INFOGAIN", # parameter
            "lts.maxterms=100000",
            "profile.memory=false",
            "research.fullconfusion=false",
            "research.writemit=true",
            "research.writemitalliters=false",
            "general.algorithm=WINNOW",
            "general.docext=",
            "general.fbeta=1.0",
            "general.fullranking=true",
            "general.maxranks=1",
            "general.minranks=1",
            "general.preprocessor=",
            "general.rankalliters=false",
            "general.saveclassprofiles=true",
            "general.threshold=1.0",
            "general.writetestrank=true",
            "gts.maxdf=1000000",
            "gts.maxtf=1000000",
            "lts.aggregated=true",
            "naivebayes.smoothing=1.0",
            "positivenaivebayes.classprobability=0.2",
            "regwinnow.complexity=0.1",
            "regwinnow.initialweight=0.1",
            "regwinnow.iterations=10",
            "regwinnow.learningrate=0.01",
            "regwinnow.ownthreshold=true",
            "research.conservememory=true",
            "research.mitsortorder=MASS",
            "rocchio.beta=1.0",
            "rocchio.gamma=1.0",
            "svmlight.params=",
            "winnow.alpha=1.05",
            "winnow.beta=0.95",
            "winnow.beta.twominusalpha=false",
            "winnow.decreasing.alpha=false",
            "winnow.decreasing.alpha.strategy=LOGARITMIC",
            "winnow.maxiters=3",
            "winnow.negativeweights=true",
            "winnow.seed=-1",
            "winnow.termselect=false",
            "winnow.termselect.epsilon=1.0E-4",
            "winnow.termselect.iterations=1,2,",
            "winnow.thetamin=0.5",
            "winnow.thetaplus=2.5"
            ])
        with open("lcs3.conf", "w", encoding = "utf-8") as config_out:
            config_out.write(config)
