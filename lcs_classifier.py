#!/usr/bin/env 

import os

import utils

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
            chunks = [list(t) for t in zip(*[iter(data)]*int(round(len(data)/25000),0))]
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
        Function to retrieve the feature from feature indices

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
            config_out.write("\n".join(config))
