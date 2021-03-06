#!/usr/bin/env 

import argparse
import utils
import datahandler
import csv

"""
Converts docs to standard format
======

Script to convert a document to a standard csv file
Doc-types: txt & excel
Columns: label, doc_id, user_id, username, date, time, meta (sparse), text, frogged 

Parameters
------
i : document
o : outfile
c : file that specifies column-values

"""

parser = argparse.ArgumentParser(description = "Converts docs to standard format")
parser.add_argument('-i', action = 'store', required = True, help = "Document")  
parser.add_argument('-o', action = 'store', required = True, help = "Outfile")
parser.add_argument('-c', action = 'store', required = False, 
    help = "Columnfile in which the value of each column is specified (needed for " +
        ".xls(x) and .txt files, not for .json files")
parser.add_argument('--header', action = 'store_true', 
    help = "Specify if the file contains a header")
args = parser.parse_args() 

# read in file
if args.i[-3:] == "xls" or args.i[-4:] == "xlsx":
    # make sure date and time fields are correctly processed
    indexline = utils.read_columnfile(args.c)
    date, time = False, False
    if indexline[3] != "-":
        date = indexline[3]
    if indexline[4] != "-":
        time = indexline[4]  
    lines = utils.read_excel(args.i, args.header, date, time)
elif args.i[-4:] == "json":
    csvrows = utils.read_json(args.i)
else: # txt file
    with open(args.i, encoding="utf-8") as fn:
        lines = [x.strip().split("\t") for x in fn.readlines()]
        if args.header:
            lines = lines[1:]

# with open(args.o, 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     for line in lines:
#         writer.writerow(line)
# quit()

# set columns of lines in right order
if args.c: 
    indexline = utils.read_columnfile(args.c)
    csvrows = []
    for line in lines:
        csvrow = []
        for i in indexline:
            if i == "-":
                csvrow.append("-")
            else:
                csvrow.append(line[i])
        csvrows.append(csvrow)

# write to csv
dh = datahandler.Datahandler()
dh.set_rows(csvrows)
dh.write_csv(args.o)
