
import sys

import featurizer
import datahandler

infile = sys.argv[1]
outdir = sys.argv[2]

dh = datahandler.Datahandler()
dh.set(infile)

tokenized = [' '.join([t[0] for t in instance]) + '\n' for instance in dh.dataset['tagged']])
cn = featurizer.CocoNgrams([1,2,3],[])
cn.fit(tmp, tokenized)
i, v = cn.transform(write = outdir + 'sparse.txt')
with open(outdir + 'vocabulary.txt', 'w', encoding = 'utf-8') as v_out:
    v_out.write('\n'.join(v))
