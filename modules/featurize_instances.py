
import numpy
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter

import featurizer
from tokenize_instances import Tokenize

class Featurize_tokens(Task):

    in_tokenized = InputSlot()

    token_ngrams = Parameter()
    blackfeats = Parameter()
    standard_vocabulary = Parameter(default = False)

    def out_features(self):
        return self.outputfrominput(inputformat='tokenized', stripextension='.tok.txt', addextension='.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='tokenized', stripextension='.tok.txt', addextension='.vocabulary.txt')

    def run(self):
        
        print('start run')
        # generate dictionary of features
        features = {'token_ngrams':{'n_list':self.token_ngrams.split(), 'blackfeats':self.blackfeats}}
        
        # read in file and put in right format
        with open(self.in_tokenized().path, 'r', encoding = 'utf-8') as file_in:
            documents = file_in.readlines()

        ft = featurizer.Featurizer(documents, features)
        ft.ft.fit_transform()
        instances, vocabulary = ft.return_instances(['token_ngrams'], )

        numpy.savez(self.out_features().path, data=instances.data, indices=instances.indices, indptr=instances.indptr, shape=instances.shape)

        vocabulary = list(vocabulary)
        with open(self.out_vocabulary().path,'w',encoding='utf-8') as vocab_out:
            vocab_out.write('\n'.join(vocabulary))
        
@registercomponent
class Featurize(StandardWorkflowComponent):

    token_ngrams = Parameter(default='1 2 3')
    blackfeats = Parameter(default=False)
    
    tokconfig = Parameter(default=False)
    strip_punctuation = BoolParameter(default=True)

    def accepts(self):
        return InputFormat(self, format_id='tokenized', extension='tok.txt'), InputComponent(self, Tokenize, config=self.tokconfig, strip_punctuation=self.strip_punctuation)
                    

#    def setup(self, workflow, input_feeds):
#        featurizertask = workflow.new_task('Featurize_tokens', Featurize_tokens, autopass=True)
#        featurizertask.in_tokenized = input_feeds['toktxt']
#        return Featurize_tokens

    def autosetup(self):
        return Featurize_tokens
