#!/usr/bin/env 

import configparser
import sys
import os

import docreader
import utils

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
		csv_doc = doc[:-4] + '_standard.csv'
		utils.write_csv(new_lines, csv_doc)
		if len(other_lines) > 0:
			meta_doc = doc[:-4] + '_meta.csv'
			utils.write_csv(other_lines, meta_doc)
	quit()
	if dp.getboolean('tag'):
		tagged_csv = doc[:-4] + '_tagged.csv'
		if dp['tagger'] == 'frog':
			os_string = os.system('python3 frog_data.py ' + doc + ' ' + tagged_csv + ' ')
			if dp.getboolean('tweets'):
				os_string += '1'
			else:
				os_string += '0'
			os.system(os_string)





