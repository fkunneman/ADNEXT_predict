
import sys

standardfile = sys.argv[1] # filename
standard_idcolumn = int(sys.argv[2]) # column of the identifier per document
tagged_bool = int(sys.argv[3])
t_column = int(sys.argv[4])
tagged_file = sys.argv[5] # filename
tagged_idcolumn = int(sys.argv[6]) # column of the identifier per document
tagged_column = int(sys.argv[7]) # column of the tagged information
tagged_type = int(sys.argv[8]) # 0 for token, 1 for lemma, 2 for pos, 3 for sentence index 
outfile = sys.argv[9]

standard_lines = []
tagged_dict = {}

with open(standardfile, 'r', encoding = 'utf-8') as sf:
    for line in sf.readlines():
        columns = line.strip().split('\t')
        idcolumn = columns[standard_idcolumn]
        standard_lines.append(idcolumn, columns)

with open(tagged_file, 'r', encoding = 'utf-8') as tf:
    for line in tf.readlines():
        columns = line.strip().split('\t')
        idcolumn = columns[tagged_idcolumn]
        taggedcol = columns[tagged_column]
        tags = taggedcol.split()
        tagged = []
        for tag in tags:
            taglist = ['-'] * 4
            taglist[tagged_type] = tag
            tagged.append(taglist)
        tagged_dict[idcolumn] = tagged

with open(outfile, 'w', encoding = 'utf-8') as out:
    for line in standard_lines:
        new_tags = tagged_dict[line[0]]
        if tagged_bool:
            tagged_col = [x.split('|') for x in line[1][t_column].split()]
            for i, tag in enumerate(tagged_col):
                tag[tagged_type] = new_tags[i][tagged_type]
            line[1][t_column] = ' '.join(['|'.join(taglist) for taglist in tagged_col])
        else:
            line[1].append(' '.join(['|'.join(taglist) for taglist in new_tags]))
        out.write('\t'.join(line[1]) + '\n')

        