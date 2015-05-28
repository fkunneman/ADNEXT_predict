#!/usr/bin/env 

import argparse
import datahandler

"""
Script to format classification files by filtering instances and adding labels
"""
parser = argparse.ArgumentParser(description = "Script to format " +
    "classification files by filtering instances and adding labels")
parser.add_argument('-i', action = 'store', required = True, nargs = '+'
    help = "The input files")  
parser.add_argument('-l', action = 'store', required = False, nargs = '+'
    help = "The label for each file")
parser.add_argument('-o', action = 'store', required = True, nargs = '+'
    help = "The output files")
parser.add_argument('-b', action = 'store', required = False, nargs = '+',
    help = "Remove instances if they contain one of the given words")
parser.add_argument('--punctuation', action = 'store_true', 
    help = "Choose to filter punctuation")
parser.add_argument('--us', action = 'store_true', 
    help = "Choose to normalize usernames")
parser.add_argument('--ur', action = 'store_true', 
    help = "Choose to normalize urls")
args = parser.parse_args()

fl = list(zip(args.i, args.l, args.o))
for infile, label, outfile in fl:
    dh = datahandler.Datahandler()
    dh.set(infile)
    dh.set_label(label)
    if args.b:
        dh.filter_instances(b)
    if args.punctuation:
        dh.filter_punctuation()
    if args.us:
        dh.normalize_usernames()
    if args.ur:
        dh.normalize_urls()
    dh.write_csv(outfile)
