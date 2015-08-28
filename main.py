#!/usr/bin/env 

import configparser
import sys
import os
import re

import docreader
import datahandler
import utils

configfile = sys.argv[1]

expdir = '/'.join(configfile.split('/')[:-1]) + '/'

cp = configparser.ConfigParser()
cp.read(configfile)

########################### Formatting data ###########################
fileformats = ['.txt', '.xls', '.csv']
data = [doc for doc in cp.sections() if doc[-4:] in fileformats]
train = []
test = []
for doc in data:
    print(doc)
    dp = cp[doc]
    keys = [k for k in dp.keys()]
    ##### Reading in data #####
    if dp.getboolean('tocsv'):
        columns = [k for k in keys if re.search(r'\d+', k)]
        columndict = {}
        catdict = {}
        for column in columns:
            index = int(column)
            cat = dp[column]
            columndict[index] = cat
            catdict[cat] = index
        date = catdict['date'] if 'date' in catdict.keys() else False
        time = catdict['time'] if 'time' in catdict.keys() else False
        sepdict = {'tab' : '\t', 'space' : ' '}
        delimiter = sepdict[dp['separator']]
        header = dp.getboolean('header')
        reader = docreader.Docreader()
        reader.parse_doc(doc, delimiter, header, date, time)
        new_lines, other_lines = reader.set_lines(columndict)
        csv_doc = doc[:-4] + '_standard.csv'
        utils.write_csv(new_lines, csv_doc)
        doc = csv_doc
        if other_lines:
            if len(other_lines) > 0:
                meta_doc = doc[:-4] + '_meta.csv'
                utils.write_csv(other_lines, meta_doc)
    ##### Tagging data #####
    if dp.getboolean('tag'):
        tagged_csv = doc[:-4] + '_tagged.csv'
        if dp['tagger'] == 'frog':
            os_string = 'python3 frog_data.py ' + doc + ' ' + tagged_csv + ' '
            if dp.getboolean('tweets'):
                os_string += '1'
            else:
                os_string += '0'
            os.system(os_string)
        doc = tagged_csv
    ##### Pre-processing data #####
    dh = datahandler.Datahandler()
    dh.set(doc)
    if dp.getboolean('preprocess'):
        if dp['add_label'] != 'no':
            dh.set_label(dp['add_label'])
        if dp.getboolean('filter_punctuation'):
            dh.filter_punctuation()
        if dp.getboolean('normalize_usernames'):
            dh.normalize_usernames()
        if dp.getboolean('normalize_urls'):
            dh.normalize_urls()
        if dp.getboolean('lower'):
            dh.to_lower()
        if dp['remove_instances'] != 'no':
            remove = dp['remove_instances'].split(' ')
            dh.filter_instances(remove)
        preprocessed_csv = doc[:-4] + '_preprocessed.csv'
        dh.write_csv(preprocessed_csv)
    if dp['train_test'] == 'train':
        train.append(dh)
    elif dp['train_test'] == 'test':
        test.append(dh)

##### Bundling data #####
trainfile = expdir + 'traindata.csv'
utils.bundle_data(train, trainfile)

testfile = expdir + 'testdata.csv'
if len(test) > 0:
    utils.bundle_data(test, testfile)

########################### Experiments ###########################
featuretypes = [featuretype for featuretype in cp.sections() if featuretype[:8] == 'Features']
features = {}
for featuretype in featuretypes:
    fp = cp[featuretype]
    keys = [k for k in fp.keys()]
    feature_dict = {}
    for key in keys:
        values = fp[key].split()
        feature_dict[key] = values
        features[featuretype] = feature_dict

print(features)

vp = cp['Vector']
weight = vp['weight'].split()
select = int(vp['select'])

print(weight)
print(select)

classifiers = [clf for clf in cp.sections() if clf[:3] == 'Clf']
clfs = {}


