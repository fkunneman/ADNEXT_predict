
import os
import sys
#import frogger
#import datahandler

level = int(sys.argv[1])

jsonfile = "/home/fkunneman/test_predict/tweets_ecoli.json"
xlsfile = "/home/fkunneman/test_predict/politiehorst.xlsx"
xlsfile2 = "/home/fkunneman/test_predict/tweets_mazelen_2.xls"
txtfile = "/home/fkunneman/test_predict/tweets_zinin.txt"

xlsconfig = "/home/fkunneman/test_predict/columns_xls.txt"
xlsconfig2 = "/home/fkunneman/test_predict/columns_xls2.txt"
txtconfig = "/home/fkunneman/test_predict/columns_txt.txt"

jsoncsv = "/home/fkunneman/test_predict/tweets_ecoli.csv"
xlscsv = "/home/fkunneman/test_predict/politiehorst.csv"
xls2csv = "/home/fkunneman/test_predict/tweets_mazelen.csv"
txtcsv = "/home/fkunneman/test_predict/tweets_zinin.csv"

jsoncsv_fr = "/home/fkunneman/test_predict/tweets_ecoli_frogged.csv"
xlscsv_fr = "/home/fkunneman/test_predict/politiehorst_frogged.csv"
xls2csv_fr = "/home/fkunneman/test_predict/tweets_mazelen_frogged.csv"
txtcsv_fr = "/home/fkunneman/test_predict/tweets_zinin_frogged.csv"

jsoncsv_ins = "/home/fkunneman/test_predict/tweets_ecoli_filtered.csv"
xlscsv_ins = "/home/fkunneman/test_predict/politiehorst_filtered.csv"
xls2csv_ins = "/home/fkunneman/test_predict/tweets_mazelen_filtered.csv"
txtcsv_ins = "/home/fkunneman/test_predict/tweets_zinin_filtered.csv"

if level <= 1:
	print("Read in json")
	os.system("python3 doc2csv.py -i " + jsonfile + " -o " + jsoncsv)
	print("Read in xls")
	os.system("python3 doc2csv.py -i " + xlsfile + " -o " + xlscsv + 
		" -c " + xlsconfig)
	print("Read in xls2")
	os.system("python3 doc2csv.py -i " + xlsfile2 + " -o " + xls2csv + 
		" -c " + xlsconfig2 + " --header")
#print("Read in txt")
#os.system("python3 doc2csv.py -i " + txtfile + " -o " + txtcsv + 
#    " -c " + txtconfig)

if level <= 2:
	print("frog json")
	os.system("python3 frog_data.py " + jsoncsv + " " + jsoncsv_fr)
	print("frog xls")
	os.system("python3 frog_data.py " + xlscsv + " " + xlscsv_fr)
	print("frog xls2")
	os.system("python3 frog_data.py " + xls2csv + " " + xls2csv_fr)
#print("frog txt")
#os.system("python3 frog_data.py " + txtcsv + " " + txtcsv_fr)

if level <= 3:
	print("setting all")
	os.system("python3 format_instances.py -i " + jsoncsv_fr + " " + xlscsv_fr + 
		" " + xls2csv_fr + " -l ziekte plisie pokken -o " + jsoncsv_ins + " " +
		xlscsv_ins + " " + xls2csv_ins + " -b rt --punctuation --us --ur")


#print("setting json")


