#!/usr/bin/env
"""
Script to tokenize the last column in a csv file
Presumes frog as python binding
"""

import sys
import frog

import datahandler
import utils

infile = sys.argv[1]
outfile = sys.argv[2]

#initialize frog
fo = frog.FrogOptions(parser=False)
frogger = frog.Frog(fo, "/vol/customopt/uvt-ru/etc/frog/frog-twitter.cfg")

#read in file
datahandler = datahandler.Datahandler(infile)
dataset = datahandler.dataset
texts = dataset["text"]

#frog lines
l = len(texts)
shows = range(10000, l, 10000) #to write intermediate output
checks = range(0, l, 1000)
frogged_texts = []
for i, text in enumerate(texts):
    if i in checks:
        print("line",i,"of",l)
    #get frogged data
    data = frogger.process(text)
    tokens = []
    sentence = -1
    for token in data:
        if token["index"] == '1':
            sentence += 1
        tokens.append("\t".join([token["text"], token["pos"], token["lemma"], str(sentence)]))
    frogged_texts.append("\n".join(tokens))
    #write intermediate output to a file
    if i in shows:
        dataset["frogs"] = frogged_texts
        rows = utils.dataset_2_rows(dataset)
        utils.write_csv(rows, outfile)

dataset["frogs"] = frogged_texts
rows = utils.dataset_2_rows(dataset)
utils.write_csv(rows, outfile)
