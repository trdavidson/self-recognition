"""
source documentation, azure: https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line%2Cpython-new&pivots=rest-api
source documentation, openai: https://platform.openai.com/docs/api-reference/chat/create
"""
from typing import Tuple, Union
from retry import retry
import attr
from attr import attrs, field
import requests
import sys

sys.path.append("src/")
from utils import get_api_key

try:
    from model_utils import ChatModel, AIMessage, SystemMessage, HumanMessage
except: 
    from .model_utils import ChatModel, AIMessage, SystemMessage, HumanMessage

@attrs
class OpenAIModel(ChatModel):
    model_provider: str = field(default='azure')
    model_name: str = field(default='gpt-3.5-turbo')
    role_mapping = field(default={'role': 'role', 'content': 'content', 'assistant': 'assistant', 'user': 'user',
                                  'system': 'system'})

    def _custom_post_init(self):
        if self.model_provider == 'azure':
            self.model_key = get_api_key(fname=self.model_key_path,
                                         provider=self.model_provider,
                                         key=("dlab_key-1-18k-loc2" if ('gpt-35' in self.model_name.replace('.', ''))
                                              else "dlab_key-1-18k"))

    def _get_model_remap(self):
        if self.model_provider == 'azure':
            model_name = self.model_name
            # model_name = self.model_name.replace(".", "")  # for the azure naming struct.
            if 'gpt-4-turbo' == self.model_name:
                model_name = 'gpt-4-turbo-2024-04-09'
            if 'gpt-4' == self.model_name:
                model_name = 'gpt-4-0613'
            if 'gpt-3.5-turbo-instruct' == self.model_name:
                model_name = 'gpt-35-turbo-0613'
            if 'gpt-3.5-turbo' == self.model_name:
                model_name = 'gpt-35-turbo-0125'
            return model_name
        else:
            return self.model_name

    def _get_api_base(self, model_name):
        if self.model_provider == 'azure':
            if model_name == 'gpt-4-turbo-2024-04-09':
                api_base = 'key-1-18k'
            elif model_name == 'gpt-4-0613':
                api_base = 'key-1-18k'
            elif model_name == 'gpt-35-turbo-0613':
                api_base = 'key-1-18k'
            elif model_name == 'gpt-35-turbo-0125':
                api_base = 'key-1-18k-loc2'
            else:
                raise ValueError(f'Invalid model name: {model_name}')
            api_base = f'https://{api_base}.openai.azure.com'
            return api_base

        else:
            return self.api_info["api_base"]

    @retry(Exception, tries=3, delay=8, backoff=2)
    def _generate(self, data, return_logprobs: bool = False, return_top_logprobs: int = 0):

        url, headers, package = self._preprocess(data)

        if self.model_provider == 'azure':
            return_logprobs = False
            return_top_logprobs = 0

        if return_logprobs:
            package['logprobs'] = True
        if isinstance(return_top_logprobs, int) and return_top_logprobs > 0:
            package['logprobs'] = True
            package['top_logprobs'] = min(return_top_logprobs, 20)

        response = requests.post(url=url, headers=headers, json=package)
        processed_response = self._postprocess(response)

        return processed_response

    def _preprocess(self, data) -> Tuple[str, dict, dict]:
        _messages = [m.prepare_for_generation(role_mapping=self.role_mapping) for m in data]

        headers = {
            "Content-type": "application/json"
        }

        package = {
            "messages": _messages,
            "temperature": self.temperature,
        }

        if self.model_provider == 'azure':
            model_name = self._get_model_remap()
            api_base = self._get_api_base(model_name)
            api_version = self.api_info["api_version"]

            url = f'{api_base}/openai/deployments/{model_name}/chat/completions?api-version={api_version}'
            headers["api-key"] = self.model_key
        else:
            url = self.api_info["api_base"]
            headers["Authorization"] = f"Bearer {self.model_key}"
            package["model"] = self.model_name

        return url, headers, package

    def _postprocess(self, data):
        """
        content is the chat completion
        logprobs is returned as:
            None if return_logprobs is False
            list if return_logprobs is True
                -> [{token: str, logprob: float, bytes: int,
                    'top_logprobs': [{token: str, logprob: float, bytes: int}]}]
        :param data: API response package
        :return: AIMessage if no logprobs, else dict with content and logprobs
        """
        try:
            data = data.json()
            content = data["choices"][0]["message"]["content"]
            logprobs_data = data["choices"][0].get('logprobs', None)
            self.prompt_tokens = data['usage']['prompt_tokens']
            self.completion_tokens = data['usage']['completion_tokens']
        except Exception as e:
            print(f'[error] failed to generate response - {data} - {e}')
            raise e

        if logprobs_data is not None:
            return AIMessage(content), logprobs_data

        return AIMessage(content)


if __name__ == "__main__":
    messages_ = [SystemMessage("This is fun, right?"), HumanMessage("Test 1, 2, 3.")]
    test_model = OpenAIModel(model_provider='azure', model_name='gpt-4-turbo')
    print(test_model(messages_, return_logprobs=False, return_top_logprobs=0))
