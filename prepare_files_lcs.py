
import sys
import os

import docreader
import utils
import featurizer

"""
Script to convert documents into the format that can be processed by the LCS implementation of Balanced Winnow

"""

infile = sys.argv[1] # txt or csv
textindex = int(sys.argv[2])
fileprefix = sys.argv[3]
label = sys.argv[4]
filesdirectory = sys.argv[5]
partsdirectory = sys.argv[6]

print('Reading in documents')
dr = docreader.Docreader()
dr.parse_doc(infile)
documents = [line[textindex] for line in dr.lines]

documents_tagged = utils.tokenized_2_tagged(documents)
print(documents_tagged[0])

featuredict = {'token_ngrams': {'n_list': [1, 2, 3]}}
ft = featurizer.Featurizer(documents, documents_tagged, partsdirectory, featuredict)
ft.fit_transform()
print(ft.feats['token_ngrams'])
instances, vocabulary = ft.return_instances(['token_ngrams'])

parts = []
# make chunks of 25000 from the data
if instances.shape[0] > 25000:
    chunks = []
    for i in range(0, instances.shape[0], 25000):
        if not i+25000 > instances.shape[0]:
            chunks.append(range(i, i+25000)) 
        else:
            chunks.append(range(i, instances.shape[0]))
else:
    chunks = [range(instances.shape[0])]
for i, chunk in enumerate(chunks):
    # make subdirectory
    subpart = fileprefix + str(i) + '/'
    subdir = filesdirectory + subpart
    if not os.path.isdir(subdir):
        os.mkdir(subdir)
    for j, index in enumerate(chunk):
        zeros = 5 - len(str(j))
        filename = subpart + ('0' * zeros) + str(j) + '.txt'
        print(index, instances.shape[0])
        features = [vocabulary[x] for x in instances[index].indices]
        with open(filesdirectory + filename, 'w', encoding = 'utf-8') as outfile: 
            outfile.write('\n'.join(features))
        parts.append(filename + ' ' + label)
with open(partsdirectory + label + '.txt', 'w', encoding = 'utf-8') as partsfile:
    partsfile.write('\n'.join(parts))
