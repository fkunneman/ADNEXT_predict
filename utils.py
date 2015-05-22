#!/usr/bin/env 

import xlrd
import json
import re

def read_columnfile(columnfile):
    """
    Reader
    =====
    Function to convert a columnfile into a dict of columns

    Parameters
    -----
    columnfile : Name of the column file, a file that specifies the 
        fieldname of each column

    Returns
    -----
    defaultline : list of the index of each key in the file 
    """

    column_sequence = "label, doc_id, user_id, username, date, time, text, " +
        "frogged".split()
    defaultline = ["-", "-", "-", "-", "-", "-", "-", "-"]

    #initialize columndict
    columndict = {
        "Label" : 0,
        "Tweet_id" : 1,
        "User_id" : 2,
        "Date" : 3,
        "Time" : 4,
        "Username" : 5,
        "Text" : 6
    }

    with open(filename) as cf:
        for line in sf.readlines():
            key, value = line.strip().split(": ")
            try:
                defaultline[columndict[key]] = int(value)
            except:
                print("Columnfile not correctly formatted, exiting " +
                    "program")
                exit()
    return defaultline

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
            tweet_id = [decoded["id"]
            user_id = decoded["user"]["id"]
            dtsearch = date_time.search(decoded["created_at"]).groups()
            date = dtsearch[1] + "-" + month[dtsearch[0]] + "-" + dtsearch[3]
            time = dtsearch[2]
            username = decoded["user"]["screen_name"]
            text = decoded["text"]
            rows.append(["-"] + [unicode(x) for x in [tweet_id, user_id, date, time, 
                username, text]])
    return rows

def read_excel(filename):
    """
    Excel reader
    =====
    Function to read in an excel file

    Parameters
    -----
    filename : Name of the excel file

    Returns
    -----
    rows : list of lists
        each list corresponds to the cell values of a row
    """
    workbook = xlrd.open_workbook(filename)
    wbsheet = workbook.sheets()[0]
    rows = []
    for rownum in range(wbsheet.nrows):
        rowvals = wbsheet.row_values(rownum)
        rows.append([unicode(x) for x in rowvals])
    return rows

def write_csv(rows, outfile):
    """
    CSV writer
    =====
    Function to write rows to a file in csv format

    Parameters
    -----
    rows : list of lists (rows and columns respectively)
    outfile : the name of the file to write the rows to

    """
    #write lines to outfile
    with open(outfile, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)