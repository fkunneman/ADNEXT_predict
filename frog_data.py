#!/usr/bin/env
"""
Script to tokenize the last column in a csv file
Presumes frog as python binding
"""

import sys
import frog

import datareader
import utils

infile = sys.argv[1]
outfile = sys.argv[2]

#initialize frog
fo = frog.FrogOptions(parser=False)
frogger = frog.Frog(fo, frog_config)

#read in file
datareader = datareader.Datareader()
datareader.set(infile)
dataset = datareader.get()
texts = dataset["texts"]

#frog lines
l = len(texts)
shows = range(10000, l, 10000) #to write intermediate output
checks = range(0, l, 1000)
frogged_texts = []
for i, texts in enumerate(texts):
    if i in checks:
        print("line",i,"of",l)
    data = frogger.process(texts)
    tokens = []
    sentence = -1
    for token in data:
        if token["index"] == '1':
            sentence += 1
        tokens.append("\t".join([token["text"], token["pos"], token["lemma"], str(sentence)])
    frogged_texts.append("\n".join(tokens))
    #write intermediate output to a file
    if i in shows:
        write_dataset["frogged"] = frogged_texts
        utils.write_data_csv(write_dataset, outfile, ["frogged"])

dataset["frogged"] = frogged_texts
utils.write_data_csv(dataset, outfile, ["frogged"])