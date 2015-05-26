
import os

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

print("frog json")
os.system("python3 frog_data.py " + jsoncsv + " " + jsoncsv_fr)
print("frog xls")
os.system("python3 frog_data.py " + xlscsv + " " + xlscsv_fr)
print("frog xls2")
os.system("python3 frog_data.py " + xls2csv + " " + xls2csv_fr)
#print("frog txt")
#os.system("python3 frog_data.py " + txtcsv + " " + txtcsv_fr)
