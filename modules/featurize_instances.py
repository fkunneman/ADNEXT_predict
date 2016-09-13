
import numpy
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter

import featurizer

class Featurize_tokens(Task):

    in_tokenized = InputSlot()

    token_ngrams = Parameter()
    blackfeats = Parameter()

    def out_features(self):
        return self.outputfrominput(inputformat='tokenized', stripextension='.tok.txt', addextension='.features')

    def run(self):
        
        # generate dictionary of features
        features = {'token_ngrams':{'n_list':self.token_ngrams.split(), 'blackfeats':self.blackfeats}}
        
        # read in file and put in right format
        with open(self.in_tokenized().path, 'r', encoding = 'utf-8') as file_in:
            documents = file_in.readlines()

        ft = featurizer.Featurizer(documents, features)
        ft.fit_transform()
        instances, vocabulary = ft.return_instances(['token_ngrams'])

        numpy.savez(self.out_features().path, data=instances.data, indices=instances.indices, indptr=instances.indptr, shape=instances.shape)
        vocabulary = list(vocabulary)
        print(' '.join(vocabulary[:10]).encode('utf-8'))
        with open('vocabulary.txt','w',encoding='utf-8') as vocab_out:
            vocab_out.write('\n'.join(vocabulary))
        
@registercomponent
class Featurize(StandardWorkflowComponent):

    token_ngrams = Parameter(default='1 2 3')
    blackfeats = Parameter(default = False)
    
    def accepts(self):
        return InputFormat(self, format_id='tokenized', extension='tok.txt')
        
    def autosetup(self):
        return Featurize_tokens
