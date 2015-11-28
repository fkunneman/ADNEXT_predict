
import ucto
import sys

import docreader
import linewriter

infile = sys.argv[1]
textindex = int(sys.argv[2])

dr = docreader.Docreader()
dr.parse_doc(infile)

tokenizer = ucto.Tokenizer('/vol/customopt/lamachine/etc/ucto/tokconfig-nl-twitter')

outfile = infile[:-4] + '_tokenized.csv'
l = len(dr.lines)
shows = range(10000, l, 10000) #to write intermediate output
checks = range(0, l, 1000)
lines_tokenized = []
for i, line in enumerate(dr.lines):
    if i in checks:
        print('line', i, 'of', l)
    tokenizer.process(line[textindex])
    tokenized = ' '.join([x.text for x in tokenizer])
    line[textindex] = tokenized
    lines_tokenized.append(line)
    if i in shows:
        lw = linewriter.Linewriter(lines_tokenized)
        lw.write_csv(outfile)

lw = linewriter.Linewriter(lines_tokenized)
lw.write_csv(outfile)

