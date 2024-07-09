"""
API Reference: https://docs.together.ai/docs/inference-rest
"""
from retry import retry
import requests
from attr import attrs, field

import sys

sys.path.append("src/")
from models.model_utils import ChatModel, BaseMessage, AIMessage, SystemMessage, HumanMessage


@attrs
class TogetherAIModel(ChatModel):
    model_provider: str = field(default='together_ai')
    model_name: str = field(default='mistralai/Mixtral-8x7B-Instruct-v0.1')
    role_mapping = field(default={'role': 'role', 'content': 'content', 'assistant': 'assistant', 'user': 'user',
                                  'system': 'system'})

    @retry(Exception, tries=2, delay=2, backoff=2)
    def _generate(self, data, return_logprobs: bool = False, return_prompt_logprobs: bool = False):

        url, headers, package = self._preprocess(data)
        if return_logprobs:
            package['logprobs'] = 1
        if return_prompt_logprobs:
            package['logprobs'] = 1
            package['echo'] = True

        response = requests.post(url=url, headers=headers, json=package)
        # response = self.client.chat.completions.create(**kwargs)
        processed_response = self._postprocess(response)

        return processed_response

    def _preprocess(self, data):
        _messages = [m.prepare_for_generation(role_mapping=self.role_mapping) for m in data]

        url = self.api_info["api_base"]

        headers = {
            "Content-type": "application/json",
            "Authorization": f"Bearer {self.model_key}"
        }

        package = {
            "messages": _messages,
            "temperature": self.temperature,
            "model": self.model_name
        }

        return url, headers, package

    def _postprocess(self, data):

        try:
            data_ = data.json()
            content = data_['choices'][0]['message']['content']
            logprobs = []
            if len(data_['prompt']) > 0:
                prompt_logprobs = data_['prompt'][0].get('logprobs', None)
            else:
                prompt_logprobs = None
            if isinstance(prompt_logprobs, dict) and prompt_logprobs.get('token_logprobs', None) is not None:
                for t, tlp in zip(prompt_logprobs.get('tokens', []), prompt_logprobs.get('token_logprobs', [])):
                    logprobs.append({'token': t, 'logprob': tlp, 'prompt': True})
            completion_logprobs = data_["choices"][0]['logprobs']

            # standardize response to OpenAI response format
            if isinstance(completion_logprobs, dict) and completion_logprobs.get('token_logprobs', None) is not None:
                for t, tlp in zip(completion_logprobs.get('tokens', []), completion_logprobs.get('token_logprobs', [])):
                    logprobs.append({'token': t, 'logprob': tlp})
            self.prompt_tokens = data_['usage']['prompt_tokens']
            self.completion_tokens = data_['usage']['completion_tokens']
        except Exception as e:
            print(f'[error] failed to generate response - {e} - {data} - {data.content}')
            raise e

        if len(logprobs) > 0:
            return AIMessage(content), logprobs

        return AIMessage(content)


if __name__ == "__main__":
    messages_ = [SystemMessage("This is fun, right?"), HumanMessage("Test 1, 2, 3.")]
    model_ = TogetherAIModel()
    print(model_(messages_, return_logprobs=True, return_prompt_logprobs=True))
