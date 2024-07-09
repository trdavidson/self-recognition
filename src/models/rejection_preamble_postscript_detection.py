import os.path

from attr import define, field
import pandas as pd
import sys
sys.path.append("src/")

from utils import read_txt, load_yaml

try:
    from model_utils import AIMessage, SystemMessage, HumanMessage, BaseMessage
    from openai_model import OpenAIModel
except:
    from .model_utils import AIMessage, SystemMessage, HumanMessage, BaseMessage
    from .openai_model import OpenAIModel


@define
class PrefixSuffixRejectClassifier:
    
    model: OpenAIModel = field(default=None)
    prefix_examples_path: str = field(default="data/prompts/preamble_prompt.yaml")
    prefix_examples: dict = field(factory=dict)
    preamble_prefix: str = field(default="Extract the preamble of the following text, if it's present: ")
    suffix_examples_path: str = field(default="data/prompts/postscript_prompt.yaml")
    suffix_examples: dict = field(factory=dict)
    postscript_prefix: str = field(default="Extract the postscript of the following text, if it's present: ")
    rejection_examples_path: str = field(default="data/prompts/reject_prompt.yaml")
    rejection_examples: dict = field(factory=dict)
    rejection_prefix: str = field(default='Is the following text a rejection?')

    def __attrs_post_init__(self):
        if self.prefix_examples_path is not None and os.path.exists(self.prefix_examples_path):
            self.prefix_examples = [BaseMessage(**msg) for msg in load_yaml(self.prefix_examples_path)['prompt']]

        if self.suffix_examples_path is not None and os.path.exists(self.suffix_examples_path):
            self.suffix_examples = [BaseMessage(**msg) for msg in load_yaml(self.suffix_examples_path)['prompt']]
            
        if self.rejection_examples_path is not None and os.path.exists(self.rejection_examples_path):
            self.rejection_examples = [BaseMessage(**msg) for msg in load_yaml(self.rejection_examples_path)['prompt']]

    def __call__(self, text):
        prefix_resp = self.extract_prefix()
        rejection_resp = self.extract_rejection()
        
        return prefix_resp, rejection_resp

    def extract_prefix(self, text):
        if self.prefix_examples is not None: 
            temp_text = HumanMessage(self.preamble_prefix + text )
            t = self.preamble_prefix + [temp_text]
            prefix_resp = self.model(t)
        else:
            prefix_resp = None 
        return prefix_resp
            
    def extract_rejection(self, text):
        if self.rejection_examples is not None: 
            temp_text = HumanMessage(self.rejection_prefix + text)
            t = self.rejection_examples + [temp_text]
            rejection_resp = self.model(t)
        else:
            rejection_resp = None 
            
        return rejection_resp

    
    
if __name__=='__main__':
    model = OpenAIModel(model_name='gpt-4-turbo')
    psrc = PrefixSuffixRejectClassifier(model)
    
    df = pd.read_csv('golden/random_45_qr100/responses.csv')
    df['response'].apply(lambda x: psrc.extract_prefix(x))
    
    out = model(prompts)
    print(out)
    