
import csv

class Datareader:
    """
    Datareader
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

    def __init__(self, max_n=False):
        self.max_n = max_n
        self.headers = "label tweet_id user_id date time username text frog".split()
        self.dataset = {k: [] for k in fields}

    def set(self, filename):
        """
        Csv reader
        =====
        function to read in a csv file and store it as a dict of lists

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
            for category, val in zip(fields, row):
                self.dataset[category].append(val)

    def get(self):
        """
        Returns dataset
        =====
        Returns the stored dataset

        Returns
        -----
        self.dataset : dict of lists

        """
        return self.dataset
