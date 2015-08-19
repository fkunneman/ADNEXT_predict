#!/usr/bin/env
"""
Script to tokenize the last column in a csv file
Presumes frog as python binding
"""

import sys
import frog

import datahandler

infile = sys.argv[1]
outfile = sys.argv[2]
twitter = int(sys.argv[3])

#initialize frog
fo = frog.FrogOptions(parser=False)
if twitter:
    frogger = frog.Frog(fo, '/vol/customopt/lamachine/etc/frog/frog-twitter.cfg')
else:
    frogger = frog.Frog(fo, '/vol/customopt/lamachine/etc/frog/frog.cfg')

#read in file
dh = datahandler.Datahandler()
dh.set(infile)
texts = dh.dataset['text']

#frog lines
l = len(texts)
shows = range(10000, l, 10000) #to write intermediate output
checks = range(0, l, 1000)
frogged_texts = []
for i, text in enumerate(texts):
    if i in checks:
        print('line',i,'of',l)
    #get frogged data
    data = frogger.process(text)
    tokens = []
    sentence = -1
    for token in data:
        if token['index'] == '1':
            sentence += 1
        tokens.append([token['text'], token['lemma'], token['pos'], str(sentence)])
    frogged_texts.append(tokens)
    #write intermediate output to a file
    if i in shows:
        dh.dataset['tagged'] = frogged_texts
        dh.dataset = dataset_2_rows()
        dh.write_csv(outfile)

dh.dataset['tagged'] = frogged_texts
dh.dataset_2_rows()
dh.write_csv(outfile)
