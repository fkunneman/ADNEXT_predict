
import csv
import sys

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
        self.headers = "label tweet_id user_id date time username text frog".split()
        self.dataset = self.set(filename)

    def set(self, filename):
        """
        Csv reader
        =====
        Function to read in a csv file and store it as a dict of lists

        Parameters
        -----
        filename : str
            The name of the csv file

        """
        csv.field_size_limit(sys.maxsize)
        rows = []
        with open(filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for line in csv_reader:
                rows.append(line)
        
        for row in rows:
            for category, val in zip(self.headers, row):
                self.dataset[category].append(val)

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
        for instance in dataset['frogs']:
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

        print("removing tweets containing",blacklist)
        print("freq tweets before",len(self.instances))
        templist = []
        for t in self.instances:
            black = False
            for w in t.wordsequence:
                for b in blacklist:
                    if re.match(b,w,re.IGNORECASE):
                        black = True
            if not black:
                templist.append(t)

        self.instances = templist
        print("freq tweets after",len(self.instances))
