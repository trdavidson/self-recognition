import requests
from attr import define, field, attrs
from typing import Tuple
import sys
sys.path.append('src/') 

from utils import get_api_key


@attrs
class OpenAIEmbedding:
    dimensions: int = field(default=None)
    model_name: str = field(default='text-embedding-3-small')
    model_provider: str = field(default='openai')
    model_key: str = field(default=None)
    model_key_path: str = field(default='secrets.json')
    model_key_name: str = field(default=None)
    request_url: str = field(default='https://api.openai.com/v1/embeddings')

    # response info
    prompt_tokens: int = field(default=None)
    total_tokens: int = field(default=None)
    
    def __attrs_post_init__(self):
        if self.model_key is None:
            self.model_key = get_api_key(fname=self.model_key_path, provider=self.model_provider,
                                         key=self.model_key_name)

    def embed(self, text: str) -> list:
        headers, package = self._preprocess(text)
        response = requests.post(url=self.request_url, headers=headers, json=package) 
        embedding = self._postprocess(response)
        return embedding        

    def _preprocess(self, text: str) -> Tuple[dict, dict]:
        headers = {
            "Content-type": "application/json",
            "Authorization": f"Bearer {self.model_key}"

        }

        package = { 
            "input": text,
            "model": self.model_name,
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            package['dimensions'] = self.dimensions
        return headers, package
    
    def _postprocess(self, response: str) -> list:
        data = response.json()
        embed = data['data'][0]['embedding']
        self.prompt_tokens = data['usage']['prompt_tokens']
        self.total_tokens = data['usage']['total_tokens']
        return embed
    
    def to_lean_dict(self):
        lean_dict = {
            'model_name': self.model_name,
            'model_provider': self.model_provider,
            'dimensions': self.dimensions
        }
        return lean_dict
           
if __name__=='__main__':
    sample_text = "Let's embed this sentence!!!"
    
    embed_model = OpenAIEmbedding()
    
    embedding = embed_model.embed(sample_text) 
    print(embedding)