"""
API reference: https://docs.anthropic.com/claude/reference/messages_post
"""
from typing import List
import requests
from retry import retry
import sys
sys.path.append("src/")
from attr import attrs, field
from models.model_utils import ChatModel, BaseMessage, AIMessage, SystemMessage, HumanMessage


@attrs
class AnthropicModel(ChatModel):
    model_provider: str = field(default='anthropic')
    model_name: str = field(default='claude-3-haiku-20240307')
    role_mapping = field(default={'role': 'role', 'content': 'content', 'assistant': 'assistant', 'user': 'user',
                                  'system': 'system'})

    @retry(requests.exceptions.RequestException, tries=3, delay=2, backoff=2)
    def _generate(self, data):
        url, headers, package = self._preprocess(data)
        response = requests.post(url, headers=headers, json=package)
        ai_msg = self._postprocess(response)

        return ai_msg

    def _preprocess(self, messages: List[BaseMessage]):
        # @2024-03-20
        # - API currently supports a top-level system msg.
        # - API allows user to "constrain" part of response, by having latest msg be assistant msg
        # - API requires first message to be a user-message
        url = self.api_info["api_base"]

        headers = self.api_info["headers"]
        headers["x-api-key"] = self.model_key

        package = {"max_tokens": self.gen_max_tokens,
                   "temperature": self.temperature,
                   "model": self.model_name
                   }
        if len(messages) > 1 and messages[0].role == SystemMessage("").role:
            system_msg = messages[0].content
            package["system"] = system_msg
            messages = messages[1:]

        if len(messages) > 0 and messages[0].role != HumanMessage("").role:
            messages = [HumanMessage("Ok, I'm ready!")] + messages

        package["messages"] = [m.prepare_for_generation(role_mapping=self.role_mapping) for m in messages]

        # legacy
        if self.model_name == "claude-2":
            package["model"] = "claude-2.0"

        return url, headers, package

    def _postprocess(self, response) -> AIMessage:
        content = ""
        try:
            body = response.json()
            content = body['content'][0]['text']
            self.prompt_tokens = body['usage']['input_tokens']
            self.completion_tokens = body['usage']['output_tokens']
        except Exception as e:
            print(f'error: failed to parse response - {body} - {response} - {e}')
            raise e

        return AIMessage(content)


if __name__ == "__main__":
    messages_test = [SystemMessage("This is fun, right?"), HumanMessage("Test 1, 2, 3.")]

    _model = AnthropicModel()
    _model.model_name = "claude-2.0"
    _model.model_name = "claude-3-opus-20240229"
    print(_model(messages_test))
