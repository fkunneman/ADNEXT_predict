#!/usr/bin/env 

import configparser
import sys

import docreader

configfile = sys.argv[1]

cp = configparser.ConfigParser()
cp.read(configfile)

# Reading in data
fileformats = ['.txt', '.xls', '.txt']
data = [doc for doc in cp.sections() if doc[-4:] in fileformats]
for doc in data:
	dp = cp[doc]
	keys = [k for k in dp.keys()]
	if dp.getboolean('tocsv'):
		print('Reading', doc)
		columns = [k for k in keys if re.search(r'\d+', k)]
		columndict = {}
		for column in columns:
			columndict[int(column)] = dp[column]  
		reader = docreader.Docreader()
		reader.parse_doc(doc)
		new_lines, other_lines = reader.set_lines(columndict)
		


