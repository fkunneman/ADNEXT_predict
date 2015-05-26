
import os

jsonfile = "/home/fkunneman/test_predict/tweets_ecoli.json"
xlsfile = "/home/fkunneman/test_predict/politiehorst.xlsx"
xlsfile2 = "/home/fkunneman/test_predict/tweets_mazelen_2.xls"
txtfile = "/home/fkunneman/test_predict/tweets_zinin.txt"

xlsconfig = "/home/fkunneman/test_predict/columns_xls.txt"
xlsconfig2 = "/home/fkunneman/test_predict/columns_xls2.txt"
txtconfig = "/home/fkunneman/test_predict/columns_txt.txt"

jsonout = "/home/fkunneman/test_predict/tweets_ecoli.csv"
xlsout = "/home/fkunneman/test_predict/politiehorst.csv"
xls2out = "/home/fkunneman/test_predict/tweets_mazelen.csv"
txtout = "/home/fkunneman/test_predict/tweets_zinin.csv"

print("Read in json")
os.system("python3 doc2csv.py -i " + jsonfile + " -o " + jsonout)
print("Read in xls")
os.system("python3 doc2csv.py -i " + xlsfile + " -o " + xlsout + 
    " -c " + xlsconfig)
print("Read in xls2")
os.system("python3 doc2csv.py -i " + xlsfile2 + " -o " + xls2out + 
    " -c " + xlsconfig2)
print("Read in txt")
os.system("python3 doc2csv.py -i " + txtfile + " -o " + txtout + 
    " -c " + txtconfig)
