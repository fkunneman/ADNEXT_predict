#!/usr/bin/env 

import colibricore

class Coco:
    """
    Colibri Core interface
    ======
    Class that handles fast word calculations

    Parameters
    ------
    tmp : str
        temporary directory for storing Colibri Core related files
    docs : list
        list of documents (str)

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

    """


    def __init__(self, tmp, txtfile):
        self.classfile = tmp + ".colibri.cls"
        self.textfile = txtfile
        self.corpusfile = tmp + ".colibri.dat"
        self.classencoder = colibricore.ClassEncoder()
        self.classencoder.build(self.textfile)
        self.classencoder.save(self.classfile)
        self.classencoder.encodefile(self.textfile, self.corpusfile)
        self.classdecoder = colibricore.ClassDecoder(self.classfile)

    def return_counts(self, tokens, length):
        options = colibricore.PatternModelOptions(mintokens = tokens, maxlength = length)
        patternmodel = colibricore.UnindexedPatternModel()
        patternmodel.train(self.corpusfile, options)
        pattern_counts = []
        for pattern, count in patternmodel.items():
            pattern_counts.append(pattern.tostring(self.classdecoder), count)
        sorted_pattern_counts = sorted(pattern_counts, key = lambda k : k[1], reverse = True)r
        return sorted_pattern_counts

    def count_lexicon(self, lexicon, tokens, length):
        patternmodel = colibricore.UnindexedPatternModel()
        patternmodel.train(self.corpusfile, options)
        for key in lexicon:
            querypattern = self.classencoder.buildpattern(key)
            try:
                print(patternmodel[querypattern])
            except KeyError:
                continue

