
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

    def __init__(self, max_n = False):
        self.max_n = max_n
        self.headers = "label tweet_id user_id date time username text frogs".split()
        self.dataset = {}
        self.rows = {}

    def set_rows(self, rows):
        """
        Data reader
        =====
        Function to read in rows directly

        Parameters
        -----
        rows : list of lists (rows and columns)
        """
        self.rows = rows
        self.rows_2_dataset()

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
        
        self.rows_2_dataset()

    def write_csv(self, outfile):
        """
        CSV writer
        =====
        Function to write rows to a file in csv format

        Parameters
        -----
        outfile : the name of the file to write the rows to

        """
        self.dataset_2_rows()
        with open(outfile, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for row in self.rows:
                writer.writerow(row)

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
        self.encode_frog()
        self.rows = list(zip(*[self.dataset[field] for field in self.headers]))
        self.decode_frog()

    def rows_2_dataset(self):
        """
        Row converter
        =====
        Converts rows into a dataset
        """
        self.dataset = {k: [] for k in self.headers}
        for row in self.rows:
            for category, val in zip(self.headers, row):
                self.dataset[category].append(val)
        self.decode_frog()

    def decode_frog(self):
        """
        Frog decoder
        =====
        Function to decode a frog string into a list of lists per document
        """
        if self.dataset['frogs'][0] != '-':
            new_frogs = []
            for doc in self.dataset['frogs']:
                new_frogs.append([token.split("\t") for token in doc.split("\n")])
            self.dataset['frogs'] = new_frogs

    def encode_frog(self):
        """
        Frog encoder
        =====
        Function to encode a frog list into a string
        """
        frogstrings = []
        for doc in self.dataset['frogs']:
            frogstrings.append("\n".join(["\t".join(token) for token in doc]))
        self.dataset['frogs'] = frogstrings

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
        =====
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

        self.dataset_2_rows()
        self.rows = [self.rows[i] for i in filtered_docs]
        self.rows_2_dataset()

    def normalize(self, regex, dummy):
        """
        Normalizer
        =====
        Function to normalize tokens and lemmas that match a regex to a dummy

        Parameters
        -----
        regex : re.compile object
            the regular expression to match
        dummy : string
            the dummy to replace a matching token with

        """
        new_frogs = []
        for doc in self.dataset['frogs']:
            new_doc = []
            for token in doc:
                if regex.match(token[0]):
                    token[0] = dummy
                    token[1] = dummy
                new_doc.append(token)
            new_frogs.append(new_doc)
        self.dataset['frogs'] = new_frogs

    def normalize_urls(self):
        """
        URL normalizer
        =====
        Function to normalize URLs to a dummy
        """
        find_url = re.compile(r"^(http://|www|[^\.]+)\.([^\.]+\.)*[^\.]{2,}")
        dummy = "_URL_"
        self.normalize(find_url, dummy)

    def normalize_usernames(self):
        """
        Username normalizer
        =====
        Function to normalize usernames to a dummy
            presumes 'twitter-format' (@username)
        """
        find_username = re.compile("^@\w+")
        dummy = "_USER_"
        self.normalize(find_username, dummy)

    def filter_punctuation(self):
        """
        Punctuation remover
        =====
        Function to remove punctuation from frogged data

        """
        new_frogs = []
        for doc in self.dataset['frogs']:
            new_doc = []
            for token in doc:
                if not token[2] == "LET()":
                    new_doc.append(token)
            new_frogs.append(new_doc)
        self.dataset['frogs'] = new_frogs

    def set_label(self, label):
        """
        Label editor
        =====
        Function to set a universal label for each instance

        Parameters
        -----
        label : string

        """
        self.dataset['label'] = [label] * len(self.dataset['label'])
