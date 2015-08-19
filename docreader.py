
import sys
import xlrd
import csv

class Docformatter:

	def __init__(self):
		self.lines = []

	def parse_doc(self, doc):
		form = doc[-4:]
		if form == '.txt':
			self.lines = self.parse_txt(doc)
		elif form == '.xls':
			self.lines = self.parse_xls(doc)
		else:
			self.lines = self.parse_csv(doc)

	def parse_txt(self, doc, delimiter = '\t', header = False):
	    with open(doc, encoding = 'utf-8') as fn:
	        lines = [x.strip().split(delimiter) for x in fn.readlines()]
	    if args.header:
	        lines = lines[1:]
	    return lines

	def parse_xls(self, doc, header = False, date = 0, time = 0):
	    """
	    Excel reader
	    =====
	    Function to read in an excel file

	    Parameters
	    -----
	    doc : str
	    	Name of the excel file
	    header : bool
	    	Indicate if the file contains a header
	    date : bool / int
	    	If one of the excel fields is in date format, specify the index of the column, give False otherwise
	    time : bool / int
	    	If one of the excel fields is in time format, specify the index of the column, give False otherwise
	    Returns
	    -----
	    lines : list of lists
	        Each list corresponds to the cell values of a row
	    """
	    workbook = xlrd.open_workbook(filename)
	    wbsheet = workbook.sheets()[0]
	    rows = []
	    begin = 0
	    if header:
	        begin = 1
	    for rownum in range(begin, wbsheet.nrows):
	        values = wbsheet.row_values(rownum)
	        if date == 0 or date:
	           try:
	               datefields = xlrd.xldate_as_tuple(wbsheet.cell_value(rownum, date), workbook.datemode)[:3]
	               values[date] = datetime.date(*datefields)
	           except TypeError:
	               values[date] = values[date]           
	        if time == 0 or time:
	           try:
	               timefields = xlrd.xldate_as_tuple(wbsheet.cell_value(rownum, time), workbook.datemode)[3:]
	               values[time] = datetime.time(*timefields)
	           except TypeError:
	               values[time] = values[time]        
	        rows.append(values)
	    return rows

	def parse_csv(self, doc):
        """
        Csv reader
        =====
        Function to read in a csv file

        Parameters
        -----
        doc : str
            The name of the csv file

        Returns
        -----
	    lines : list of lists
	        Each list corresponds to the cell values of a row
        """
        csv.field_size_limit(sys.maxsize)
        lines = []
        try:
            with open(doc, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                for line in csv_reader:
                    lines.append(line)
        except:
            csvfile = open(doc, 'r')
            csv_reader = csv.reader(line.replace('\0','') for line in csvfile.readlines())       
            for line in csv_reader:
                lines.append(line)

        return lines

	def set_lines(self, columndict):
		"""
		Columnformatter
		=====
		Function to set columns in the standard format

		Parameters
		-----
		columndict : dict
			dictionary to specify the column for the present categories

		Attributes
		-----
		columns : dict
			Dictionary with the standard column for each category
		defaultline : list
			Standard line that is copied for each new line
			Categories that are not present are left as '-'

		Returns
		-----
		new_lines : list of lists
			The correctly formatted lines

		"""
	    columns = {
	       'label' : 0,
	       'tweet_id' : 1,
	       'author_id' : 2,
	       'date' : 3,
	       'time' : 4,
	       'authorname' : 5,
	       'text' 	: 6
	       'tagged'	: 7
	    }
	    standard_cats = columns.keys()
	    defaultline = ["-", "-", "-", "-", "-", "-", "-", "-"]
		other_header = []
		for key, value in sorted(columndict.items()):
			if not value in standard_cats:
				other_header.append(value)
		if len(other_header) > 0:
			other = True
			other_lines = [other_header]
		else:
			other = False
			other_lines = False

		new_lines = []
		for line in self.lines:
			new_line = defaultline[:]
			if other:
				other_line = []
			for key, value in sorted(columndict.items()):
				if value in standard_cats:
					new_line[columns[value]] = line[key]
				else:
					other_line.append(line[key])
			new_lines.append(new_line)
			if other:
				other_lines.append(other_line)
		return new_lines, other_lines
