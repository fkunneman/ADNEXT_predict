
import sys
import pickle

import ucto

model = sys.argv[1]
vocabulary = sys.argv[2]
weights = sys.argv[3]

with open(model, 'rb') as model_open:
    clf = pickle.load(model_open)
tokenizer = ucto.Tokenizer('/vol/customopt/lamachine/etc/ucto/tokconfig-nl-twitter')
vocabulary = {}
keys = []
with open(vocab, 'r', encoding = 'utf-8') as vocabularyfile:
    keys = [x.strip() for x in vocabularyfile.readlines()]
vocabulary_length = len(self.keys)
vocabulary = {x:i for i, x in enumerate(self.keys)}



while True:
    sentence = input('Please enter some input...\n--> ')
    print(sentence)

