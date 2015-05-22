
import os

jsonfile = "/home/fkunneman/test_predict/tweets_ecoli.json"
xlsfile = "/home/fkunneman/test_predict/politiehorst.xlsx"
txtfile = "/home/fkunneman/test_predict/tweets_zinin.txt"

xlsconfig = "/home/fkunneman/test_predict/columns_xls.txt"
txtconfig = "/home/fkunneman/test_predict/columns_txt.txt"

jsonout = "/home/fkunneman/test_predict/tweets_ecoli.csv"
xlsout = "/home/fkunneman/test_predict/politiehorst.csv"
txtout = "/home/fkunneman/test_predict/tweets_zinin.csv"

print("Read in json")
os.system("python3 doc2csv.py -i " + jsonfile + " -o " + jsonout)
print("Read in xls")
os.system("python3 doc2csv.py -i " + xlsfile + " -o " + xlsout + 
    " -c " + xlsconfig)
print("Read in txt")
os.system("python3 doc2csv.py -i " + txtfile + " -o " + txtout + 
    " -c " + txtconfig)
