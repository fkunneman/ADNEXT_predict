
import os
import sys
import datahandler
import featurizer

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
txt1csv = "/home/fkunneman/test_predict/tweets_zinin_n.csv"
txt2csv = "/home/fkunneman/test_predict/tweets_geenzinin.csv"

jsoncsv_fr = "/home/fkunneman/test_predict/tweets_ecoli_frogged.csv"
xlscsv_fr = "/home/fkunneman/test_predict/politiehorst_frogged.csv"
xls2csv_fr = "/home/fkunneman/test_predict/tweets_mazelen_frogged.csv"
txt1csv_fr = "/home/fkunneman/test_predict/tweets_zinin_frogged.csv"
txt2csv_fr = "/home/fkunneman/test_predict/tweets_geenzinin_frogged.csv"

jsoncsv_ins = "/home/fkunneman/test_predict/tweets_ecoli_filtered.csv"
xlscsv_ins = "/home/fkunneman/test_predict/politiehorst_filtered.csv"
xls2csv_ins = "/home/fkunneman/test_predict/tweets_mazelen_filtered.csv"
txt1csv_ins = "/home/fkunneman/test_predict/tweets_zinin_filtered.csv"
txt2csv_ins = "/home/fkunneman/test_predict/tweets_geenzinin_filtered.csv"

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
#	os.system("python3 frog_data.py " + jsoncsv + " " + jsoncsv_fr)
	print("frog xls")
#	os.system("python3 frog_data.py " + xlscsv + " " + xlscsv_fr)
	print("frog xls2")
#	os.system("python3 frog_data.py " + xls2csv + " " + xls2csv_fr)
	print("frog txt")
	os.system("python3 frog_data.py " + txt1csv + " " + txt1csv_fr)
	os.system("python3 frog_data.py " + txt2csv + " " + txt2csv_fr)

if level <= 3:
	print("setting all")
	os.system("python3 format_instances.py -i " + jsoncsv_fr + " " + xlscsv_fr + 
		" " + xls2csv_fr + " " + txt1csv_fr + " " + txt2csv_fr + " -l ziekte plisie pokken zinin geenzinin -o " + jsoncsv_ins + " " +
		xlscsv_ins + " " + xls2csv_ins + " " + txt1csv_ins + " " + txt2csv_ins + " -b rt --punctuation --us --ur")

if level <= 4:
    print("extracting features")
    #read data
    dh_zin = datahandler.Datahandler()
    dh_zin.set(txt1csv_ins)
    dh_geenzin = datahandler.Datahandler()
    dh_geenzin.set(txt2csv_ins)
    raw = dh_zin.dataset['text'] + dh_geenzin.dataset['text']
    frogs = dh_zin.dataset['frogs'] + dh_geenzin.dataset['frogs']
    features = {'token_ngrams': {'n_list': [1, 2, 3], 'max_feats': 1500}}
    f = featurizer.Featurizer(raw, frogs, features)
    feats = f.fit_transform()
