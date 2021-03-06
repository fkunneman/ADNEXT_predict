#!/usr/bin/env 

import configparser
import sys
import os
import re

import docreader
import datahandler
import utils
import experimenter

configfile = sys.argv[1]

expdir = '/'.join(configfile.split('/')[:-1]) + '/'

cp = configparser.ConfigParser()
cp.read(configfile)

########################### Formatting data ###########################
fileformats = ['.txt', '.xls', '.csv']
fields = ['label', 'doc_id', 'author_id', 'date', 'time', 'authorname', 'text', 'tagged']
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
        delimiter = sepdict[dp['separator']] if 'separator' in dp.keys() else False
        header = dp.getboolean('header')
        reader = docreader.Docreader()
        reader.parse_doc(doc, delimiter, header, date, time)
        new_lines, other_lines = reader.set_lines(fields, columndict)
        csv_doc = doc[:-4] + '_standard.csv'
        if new_lines[1][-1] != '-':
            new_lines_tags = []
            for line in new_lines:
                t = line[-1]
                line[-1] = '\n'.join(['\t'.join(x.split('|')) for x in t.split(' ')])
                new_lines_tags.append(line)
            utils.write_csv(new_lines_tags, csv_doc)
        else:
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
        if dp.getboolean('add_label'):
            dh.set_label(dp['label'])
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
print('bundling data')
trainfile = expdir + 'traindata.csv'
if len(train) > 0:
    dh_train = utils.bundle_data(train, trainfile)      
else:
    dh_train = datahandler.Datahandler()
    dh_train.set(trainfile)
if 'balance' in cp.sections():
    print('balancing data')
    dh_train = utils.balance_data(dh_train, cp['balance']['outfile'])
train_dataset = dh_train.dataset

testfile = expdir + 'testdata.csv'
if len(test) > 0:
    dh_test = utils.bundle_data(test, testfile)
    test_dataset = dh_test.dataset
else:
    try:
        dh_test = datahandler.Datahandler()
        dh_test.set(testfile)
        test_dataset = dh_test.dataset
    except:
        test_dataset = False

##### Sampling data #####
if 'sample' in cp.sections():
    print('sampling data')
    samples = []
    samplevalues = [int(x) for x in cp['sample']['values'].split()]
    for sample in samplevalues:
        dh_sample = datahandler.Datahandler()
        dh_sample.set_rows(dh_train.rows)
        dh_sample.sample(int((len(dh_sample.rows) / 100) * sample))
        samples.append((str(sample), dh_sample.dataset))
else:
    samples = [('100', train_dataset)]

########################### Experiments ###########################
featuretypes = [featuretype for featuretype in cp.sections() if featuretype[:8] == 'Features']
features = {}
for featuretype in featuretypes:
    featurename = featuretype[9:]
    fp = cp[featuretype]
    keys = [k for k in fp.keys()]
    feature_dict = {}
    for key in keys:
        values = fp[key].split()
        feature_dict[key] = values
    features[featurename] = feature_dict

vp = cp['Vector']
weight = vp['weight'].split()
select = [int(x) for x in vp['select'].split()]

sp = cp['Special']
usersplit = sp.getboolean('usersplit')
print('usersplit is', usersplit)

classifiers = [clf for clf in cp.sections() if clf[:3] == 'Clf']
clfs = {}
for classifier in classifiers:
    clp = cp[classifier]
    keys = [k for k in clp.keys()]
    clf = {}
    for key in keys:
        value = clp[key]
        if re.search(' ', value):
            value = value.split()
        clf[key] = value
    clfs[classifier[4:]] = clf
    
ensemble_clfs = [clf for clf in cp.sections() if clf[:12] == 'Ensemble_clf']
assessor = []
approach = ''
helpers = {}
for classifier in ensemble_clfs:
    clp = cp[classifier]
    keys = [k for k in clp.keys() if not k in ['helper', 'assessor', 'approach']]
    clf = {}
    clf_name = classifier[13:]
    for key in keys:
        value = clp[key]
        if re.search(' ', value):
            value = value.split()
        clf[key] = value
    if cp[classifier].getboolean('helper'):
        helpers[clf_name] = clf
    if cp[classifier].getboolean('assessor'):
        assessor = [clf_name, clf]
        approach = cp[classifier]['approach']
if len(ensemble_clfs) > 0:
    ensemble_clf = {'helpers' : helpers, 'assessor' : assessor, 'approach' : approach}
    clfs['ensemble_clf'] = ensemble_clf

for sample in samples:
    sampledir = expdir + sample[0] + '_percent/'
    if not os.path.isdir(sampledir):
        os.mkdir(sampledir)
    print('sample', sample[0])
    grid = experimenter.Experiment(sample[1], test_dataset, features, weight, select, clfs, sampledir, usersplit)
    print('featurizing data')
    grid.set_features()
    print('running experiment grid')
    grid.run_grid()
