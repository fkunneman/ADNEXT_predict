
import random as rnd
import csv

class Datareader:
    """
    Datareader
    ======
    Container of datasets to be passed on to a featurizer. Can convert .csv
    files into a dataset with a chosen key. Envisioned for training:
    - can return combined datasets with any combinations of keys
    - works with a specified number of lines

    Parameters
    ------
    max_n : int, optional, default False
        Maximum number of data instances *per dataset* user wants to work with.

    shuffle : bool, optional, default True
        If the order of the dataset should be randomized.

    rnd_seed : int, optional, default 99
        A seed number used for reproducing the random order.

    Attributes
    -----
    datasets : dict
        Dictionary where key is the name of a dataset, and the value its rows.
        Will not be filled if data is streamed.

    headers : list
        List of headers / labels if they were found present in the datasets. If
        not, standard AMiCA list is provided. Might introduce bugs.

    Examples
    -----
    Interactive:

    >>> reader = Datareader(max_n=1000)
    >>> reader.add_dataset('blogs.csv', dataset_key = 'blogs')
    >>> reader.add_dataset('csi.csv', dataset_key = 'csi')
    >>> csi = reader.get_data('csi')
    >>> blogs_csi = reader.combine_datasets(['blogs','csi'])
    >>> instance = reader.process_raw_text(string)

    Passive:

    >>> reader = Datareader(max_n=1000)
    >>> data = ['~/Documents/data1.csv', '~/Documents/data2.csv']
    >>> dataset = reader.load(data, dict_format=True)
    """
    def __init__(self, max_n=False, shuffle=True, rnd_seed=99):
        self.max_n = max_n
        self.shuffle = shuffle
        self.rnd_seed = rnd_seed
        self.headers = "user_id age gender loc_country loc_region \
                       loc_city education pers_big5 pers_mbti texts".split()
        self.datasets = {}

        rnd.seed(self.rnd_seed)

    ## Passive use ------------------------------------------------------------

    def load(self, file_list, dict_format=False):
        """
        Raw data loader
        =====
        If you'd rather load all the data directly by a list of files, this is
        the go-to function. Produces exactly the same result with dict_format=
        True as the interactive commands (see class docstring).

        Parameters
        -----
        file_list : list of strings
            List with document directories to be loaded.

        dict_format : bool, optional, default False
            Set to True if the datasets should be divided in a dictionary where
            their key is the filename and the value the data matrix.

        Returns
        -----
        data : list or dict
            Either a flat list of lists with all data mixed, or a dict where 
            the datasets are split (see dict_format).
        """
        if dict_format:
            # note: fix unix only split (sometime :) )
            data = {filename.split('/')[-1:][0]: 
                    self.load_data_linewise(filename) 
                    for filename in file_list}
        else:
            data = [row for filename in file_list for row
                    in self.load_data_linewise(filename)]
        if self.shuffle and not dict_format:
            rnd.shuffle(data)
        return data

    ## General functions ------------------------------------------------------

    def load_data_linewise(self, filename):
        """
        Csv reader
        =====
        Reads a csv file by pathname, extracts headers and returns matrix.

        Parameters
        -----
        filename : str
            Directory of a .csv file to be stored into a list.

        Returns
        -----
        rows : list
            List of lists where each row is an instance and column a label 
            entry or text data.

        """
        data : list or dict
            Either a flat list of lists with all data mixed, or a dict where 
            the datasets are split (see dict_format).
        rows, has_header = [], False
        with open(filename, 'r') as sniff_file:
            if csv.Sniffer().has_header(sniff_file.read(200)):
                has_header = True
        with open(filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for i, line in enumerate(csv_reader):
                if has_header and i == 0:
                    self.headers = line
                elif self.max_n and i >= self.max_n:
                    break
                else:
                    rows.append(line)
        return rows