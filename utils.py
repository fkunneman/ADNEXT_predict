#!/usr/bin/env 

import xlrd
import json
import csv
import re
import datetime
import functools
from collections import Counter

def write_csv(rows, outfile, header = True):
    """
    CSV writer
    =====
    Function to write rows to a file in csv format

    Parameters
    -----
    rows : list of lists
    outfile : str
        The name of the file to write the rows to
    """
    with open(outfile, 'w') as csvfile:
        if header:
            writer = csv.writer(csvfile, fieldnames = rows[0])
        writer.writeheader()
        for row in rows[1:]:
            writer.writerow(row)

def read_json(filename):
    """
    Reader
    =====
    Function to read in a json file

    Parameters
    -----
    filename : Name of the json formatted file
        function presumes json output of twiqs

    Returns
    -----
    rows : a list of lists
        each list corresponds to a row with values (in the right order)
    """

    month = {"Jan" : "01", "Feb" : "02", "Mar" : "03", "Apr" : "04", 
        "May" : "05", "Jun" : "06", "Jul" : "07", "Aug" : "08", 
        "Sep" : "09", "Oct" : "10", "Nov" : "11", "Dec" : "12"}
    date_time = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)" + 
        r" (\d+) (\d{2}:\d{2}:\d{2}) \+\d+ (\d{4})")

    rows = []
    with open(filename, encoding = "utf-8") as js:
        for line in js.readlines():
            decoded = json.loads(line)
            if "twinl_lang" in decoded and decoded["twinl_lang"] != "dutch":
                continue
            tweet_id = decoded["id"]
            user_id = decoded["user"]["id"]
            dtsearch = date_time.search(decoded["created_at"]).groups()
            date = dtsearch[3] + "-" + month[dtsearch[0]] + "-" + dtsearch[1]
            time = dtsearch[2]
            username = decoded["user"]["screen_name"]
            text = decoded["text"]
            rows.append(["-"] + [tweet_id, user_id, date, time, username, text] + ["-"])
    return rows


def return_folds(instances, n = 10):
    folds = []
    for i in range(n):
        j = i
        fold = []
        while j < len(instances):
            fold.append(instances[j])
            j += n
        folds.append(fold)
    runs = []
    foldindices = range(n)
    for run in foldindices:
        testindex = run
        trainindices = list(set(foldindices) - set([testindex]))
        trainfolds = [folds[i] for i in trainindices]
        train = functools.reduce(lambda y, z: y+z, trainfolds)
        test = folds[testindex]
        runs.append([train, test])
    return runs

def find_ngrams(input_list, n):
    """
    Calculate n-grams from a list of tokens/characters with added begin and end
    items. Based on the implementation by Scott Triglia
    http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
    """
    for x in range(n-1):
        input_list.insert(0, '')
        input_list.append('')
    return zip(*[input_list[i:] for i in range(n)])

def freq_dict(text):
    """
    Returns a frequency dictionary of the input list
    """
    c = Counter()
    for word in text:
        c[word] += 1
    return c

def format_table(data, widths):
    output = []
    for row in data:
        try:
            output.append("".join("%-*s" % i for i in zip(widths, row)))
        except:
            print('Format table: number of width values and row values do not correspond, using tabs instead')
            output.append('\t'.join(row))
    return output
