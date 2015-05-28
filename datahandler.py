
import csv
import sys
import re

class Datahandler:
    """
    Datahandler
    ======
    Container of datasets to be passed on to a featurizer. Can convert .csv
    files into a dataset

    Parameters
    ------
    max_n : int, optional, default False
        Maximum number of data instances *per dataset* user wants to work with.

    Attributes
    -----
    datasets : dict
        Dictionary where key is the name of a dataset, and the value its rows.
        Will not be filled if data is streamed.

    Examples
    -----
    Interactive:

    >>> reader = Datareader(max_n=1000)
    >>> reader.set('blogs.csv')
    >>> blogs = reader.get()
    """

    def __init__(self, filename, max_n=False):
        self.max_n = max_n
        self.headers = "label tweet_id user_id date time username text frogs".split()
        self.dataset = self.set(filename)
        if self.dataset['frogs'][0] != '-':
            self.decode_frog()

    def set(self, filename):
        """
        Csv reader
        =====
        Function to read in a csv file and store it as a dict of lists

        Parameters
        -----
        filename : str
            The name of the csv file

        Returns
        -----
        dataset : dict of lists
            each column with an identifier

        """
        csv.field_size_limit(sys.maxsize)
        self.rows = []
        with open(filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for line in csv_reader:
                self.rows.append(line)
        
        dataset = self.rows_2_dataset()
        return dataset

    def decode_frog(self):
        """
        Frog decoder
        =====
        Function to decode a frog string into a list of lists per document
        """
        new_frogs = []
        for doc in self.dataset['frogs']:
            new_frogs.append([token.split("\t") for token in doc.split("\n")])
        self.dataset['frogs'] = new_frogs

    def dataset_2_rows(self):
        """
        Dataset converter
        =====
        Converts a dataset into rows
        Needed to write a dataset to a file in csv format 

        Sets
        -----
        self.rows : list of lists (rows and columns respectively)
        """
        #format is now {'texts'=[], 'user_id'=[], ...}. Needs to be converted in an instance per line
        self.rows = list(zip(*[self.dataset[field] for field in self.headers]))

    def rows_2_dataset(self):
        """
        Row converter
        =====
        Converts rows into a dataset

        returns
        -----
        dataset : dict of lists (each column with an identifier)
        """
        dataset = {k: [] for k in self.headers}
        for row in self.rows:
            for category, val in zip(self.headers, row):
                dataset[category].append(val)
        return dataset

    def return_sequences(self, tag):
        """
        Tag selecter
        =====
        Function to extract the sequence of a specific tag per 
        document
        Presumes a column with frogged data

        Parameters
        -----
        tag : str
            the tag of which to return the sequences
            options: 'token', 'lemma', 'postag', 'sentence'

        Returns
        -----
        sequences : list of lists
            the sequence of a tag per document
        """
        tagdict = {'token' : 0, 'lemma' : 1, 'postag' : 2, 'sentence' : 3}
        tagindex = tagdict[tag]
        sequences = []
        for instance in self.dataset['frogs']:
            sequences.append([token[tagindex] for token in instance])
        return sequences

    def filter_instances(self, blacklist):
        """
        Instance filter
        ===
        Function to filter instances from the dataset if they contain a string
        from the blacklist

        Parameters
        -----
        blacklist : list of strings 
            Any instance that contains a word from the blacklist is filtered
        """

        tokenized_docs = self.return_sequences('token')
        filtered_docs = [] #list of indices
        for i, doc in enumerate(tokenized_docs):
            black = False
            for token in doc:
                for string in blacklist:
                    if re.match(string, token, re.IGNORECASE):
                        black = True
            if not black:
                filtered_docs.append(i)

        self.rows = [self.rows[i] for i in filtered_docs]
        self.rows_2_dataset()        
