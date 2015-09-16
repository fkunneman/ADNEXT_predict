
import sys
import os

import featurizer
import datahandler

infile = sys.argv[1]
outdir = sys.argv[2]

def write_sparse(dh, outdir):
    tokenized = [' '.join([t[0] for t in instance]) + '\n' for instance in dh.dataset['tagged']])
    cn = featurizer.CocoNgrams([1,2,3],[])
    cn.fit(tmp, tokenized)
    i, v = cn.transform(write = outdir + 'sparse.txt')
    with open(outdir + 'vocabulary.txt', 'w', encoding = 'utf-8') as v_out:
        v_out.write('\n'.join(v))

dh100 = datahandler.Datahandler()
dh100.set(infile)
outdir100 = outdir + '100/'
os.mkdir(outdir100)
write_sparse(dh100, outdir100)

dh10 = datahandler.Datahandler()
dh10.set_rows(dh100.rows)
dh10.sample(int(len(dh100.rows) / 10))
outdir10 = outdir + '10/'
os.mkdir(outdir10)
write_sparse(dh10, outdir10)

dh1 = datahandler.Datahandler()
dh1.set_rows(dh10.rows)
dh1.sample(int(len(dh10.rows) / 10))
outdir1 = outdir + '1/'
os.mkdir(outdir1)
write_sparse(dh1, outdir1)
