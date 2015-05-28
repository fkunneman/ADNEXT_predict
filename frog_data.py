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

#initialize frog
fo = frog.FrogOptions(parser=False)
frogger = frog.Frog(fo, "/vol/customopt/uvt-ru/etc/frog/frog-twitter.cfg")

#read in file
dh = datahandler.Datahandler(infile)
texts = dh.dataset["text"]

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
        tokens.append([token["text"], token["pos"], token["lemma"], str(sentence)])
    frogged_texts.append(tokens)
    #write intermediate output to a file
    if i in shows:
        dh.dataset["frogs"] = frogged_texts
        dh.dataset = dataset_2_rows()
        dh.write_csv(outfile)

dh.dataset["frogs"] = frogged_texts
dh.dataset_2_rows()
dh.write_csv(outfile)
