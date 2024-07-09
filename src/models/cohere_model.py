"""
source documentation: https://docs.cohere.com/reference/chat
"""
from typing import List
from attr import attrs, field
import requests
from retry import retry
import numpy as np
import sys
sys.path.append("src/")

try:
    from model_utils import ChatModel, AIMessage, SystemMessage, HumanMessage, BaseMessage
except: 
    from .model_utils import ChatModel, AIMessage, SystemMessage, HumanMessage, BaseMessage


@attrs
class CohereModel(ChatModel):
    model_provider: str = field(default="cohere")
    model_name: str = field(default="command-r-plus")  # command-light is alternative
    api_endpoint: str = field(default="https://api.cohere.ai/v1/chat")
    role_mapping = field(default={'role': 'role', 'content': 'message', 'assistant': 'CHATBOT', 'user': 'USER',
                                  'system': 'SYSTEM'})
    embed_model_name: str = field(default="embed-english-light-v3.0")
    embedding_types: list = ['float']

    @retry(requests.exceptions.RequestException, tries=3, delay=2, backoff=2)
    def _generate(self, data) -> AIMessage:
        url, headers, package = self._preprocess(data)
        response = requests.post(url=url, headers=headers, json=package)
        ai_msg = self._postprocess(response)

        return ai_msg

    def _preprocess(self, data: List[BaseMessage]) -> (str, dict, dict):
        next_msg = data[-1].content
        messages = [m.prepare_for_generation(role_mapping=self.role_mapping) for m in data]
        messages = messages[:-1]

        url = self.api_endpoint
        headers = {
            "Authorization": f"Bearer {self.model_key}",
            "Content-type": "application/json"
        }

        package = {
            "chat_history": messages,
            "message": next_msg,
            "temperature": self.temperature,
            "preamble": ""  # clear default system context @20240320
        }

        return url, headers, package

    def _postprocess(self, response) -> AIMessage:
        content = ""
        try:
            body = response.json()
            content = body['text']
            self.prompt_tokens = body["meta"]["tokens"]["input_tokens"]
            self.completion_tokens = body["meta"]["tokens"]["output_tokens"]
        except Exception as e:
            print(f'error: failed to unpack cohere API response - {e} - {response.content}')
            raise e

        return AIMessage(content)

    def embed(self, data: List[str]):
        """
        Obtain embeddings for a list of strings.

        Note: each text in data should be less than 512 tokens.
              number of texts per call is 96
        source: https://docs.cohere.com/reference/embed

        :param data: list of N text to embed
        :return: embeddings: N x d
        """

        url, headers, packages = self._embed_preprocess(data)
        embeddings = []
        for package in packages:
            response = requests.post(url=url, headers=headers, json=package)
            _embeddings = self._embed_postprocess(response)
            embeddings.append(_embeddings)

        embeddings_grouped = {}
        for em in self.embedding_types:
            em_group = [e[em] for e in embeddings if e is not None]
            if len(em_group) > 0:
                em_group = np.concatenate(em_group, axis=0)
            embeddings_grouped[em] = em_group

        return embeddings_grouped

    def _embed_preprocess(self, data):
        url = "https://api.cohere.ai/v1/embed"
        headers = {
            "Authorization": f"Bearer {self.model_key}",
            "Content-type": "application/json",
            "X-Client-Name": "ycm"
        }

        packages = []
        num_packages = int(np.ceil(len(data) / 96))
        for i in range(num_packages):
            start = i * 96
            end = (i+1) * 96
            package = {
                "model": self.embed_model_name,  # d=384
                "texts": data[start:end],
                "input_type": "clustering",
                "embedding_types": self.embedding_types  # int8, uiint8, binary, ubinary
            }
            packages.append(package)
        return url, headers, packages

    @staticmethod
    def _embed_postprocess(response):
        try:
            body = response.json()
            embeddings = body["embeddings"]
            input_token_cost = body["meta"]["billed_units"]["input_tokens"]
            output_token_cost = body["meta"]["billed_units"].get("output_tokens")
            warnings = input_token_cost = body["meta"].get("warnings")
        except Exception as e:
            embeddings = None
            print(f'[error] failed to unpack cohere embedding response: {e}')

        return embeddings


if __name__ == "__main__":
    test_messages = [SystemMessage("This is fun, right?"), HumanMessage("Test 1, 2, 3.")]

    test_model = CohereModel()
    test_model.model_name = 'command-r-plus'
    print(test_model(test_messages))
