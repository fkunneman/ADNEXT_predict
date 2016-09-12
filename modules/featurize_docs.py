
from pynlpl.formats import folia
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, Parameter, InputSlot
from luiginlp.modules.ucto import Ucto

import simple_featurizer

class FeaturizeTask(Task):
    """Featurizes a single file"""

    in_folia = InputSlot() #input slot for a FoLiA document

    def out_featuretxt(self):
        """Output slot -- outputs a single *.features.txt file"""
        return self.outputfrominput(inputformat='folia', stripextension='.folia', addextension='.features')

    def run(self):
        """Run the featurizer"""
        doc = folia.Document(file=self.in_folia().path, encoding = 'utf-8')

        ft = simple_featurizer.Featurizer()
        features = ft.extract_words(doc)
        with open(self.out_featuretxt().path,'w',encoding = 'utf-8') as f_out:
            f_out.write(' '.join(features))

@registercomponent
class FeaturizeComponent(StandardWorkflowComponent):

    language = Parameter()

    def autosetup(self):
        return FeaturizeTask

    def accepts(self):
        return InputFormat(self, format_id='folia', extension='folia.xml'), InputComponent(self, Ucto, language = self.language)
