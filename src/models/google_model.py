"""
-> source documentation: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-chat
-> models: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
"""
from typing import List, Tuple
from attr import attrs, field
import requests
import jwt
import json
import time
from retry import retry
import sys
import fire
import numpy as np

sys.path.append("src/")
try:
    from model_utils import ChatModel, BaseMessage, AIMessage, SystemMessage, HumanMessage
except ImportError:
    from .model_utils import ChatModel, BaseMessage, AIMessage, SystemMessage, HumanMessage


@attrs
class GoogleModel(ChatModel):
    model_provider: str = field(default='google')
    model_name: str = field(default='chat-bison')
    project_id: str = field(default="whoami-420808")
    api_endpoint: str = field(default="us-central1-aiplatform.googleapis.com")
    role_mapping = field(default={'role': 'author', 'content': 'content', 'assistant': 'bot', 'user': 'user'})
    role_mapping_anthropic = field(default={'role': 'role', 'content': 'content', 'assistant': 'assistant',
                                            'user': 'user', 'system': 'system'})
    key_updated_at: float = 0.

    def _custom_post_init(self):
        if self.model_provider == 'google':
            try:
                self.model_key, self.key_updated_at = self._refresh_token()
            except Exception as e:
                print(f'warning: failed to refresh GCP token - {e}')

        if 'gemini' in self.model_name:
            self.role_mapping = {'role': 'role', 'content': 'text', 'assistant': 'model', 'user': 'user',
                                 'system': 'user'}

    def _get_model_remap(self):
        m_map = {
            'chat-bison': 'chat-bison',
            'claude-3-haiku-20240307': 'claude-3-haiku@20240307',
            'claude-3-sonnet-20240229': 'claude-3-sonnet@20240229',
            'claude-3-opus-20240229': 'claude-3-opus@20240229',
            'claude-3-5-sonnet-20240620': 'claude-3-5-sonnet@20240620'
        }

        model_name = m_map.get(self.model_name)
        if model_name is None:
            raise NotImplementedError(f'model {self.model_name} not implemented')
        return model_name

    @retry(Exception, tries=3, delay=8, backoff=2)
    def _generate(self, data) -> AIMessage:

        if self.model_provider == 'google' and (time.time() - self.key_updated_at) > 60*58:
            self.model_key, self.key_updated_at = self._refresh_token()

        is_gemini = 'gemini' in self.model_name

        url, headers, package = self._preprocess_v2(data) if is_gemini else self._preprocess(data)
        response = requests.post(url=url, headers=headers, json=package)
        ai_msg = self._postprocess_v2(response) if is_gemini else self._postprocess(response)

        return ai_msg

    def _preprocess(self, data: List[BaseMessage]) -> (str, dict, dict):

        if 'claude-3' in self.model_name:
            return self._preprocess_anthropic(data)

        system_msg = ''
        # NOTE 1: moved away from isinstance(msg, SystemMessage) since relative imports might invalidate this evaluation
        # example: src.model_utils.SystemMessage will return False for msg instantiated using model_utils.SystemMessage
        # NOTE 2: Google API expects at least 1 message in the package body -> only parse system message to context if
        # more than 1 message in list
        if len(data) > 1 and data[0].role == SystemMessage("").role:
            system_msg = data[0].content
            data = data[1:]

        _messages = [m.prepare_for_generation(role_mapping=self.role_mapping) for m in data]

        model_name = self._get_model_remap()
        url = f'https://{self.api_endpoint}/v1/projects/{self.project_id}/locations/us-central1/publishers/google/' \
              f'models/{model_name}:predict'

        headers = {
            "Authorization": f"Bearer {self.model_key}",
            "Content-type": "application/json"
        }
        package = {
            "instances": [
                {
                    "context": system_msg,
                    "examples": [],
                    "messages": _messages
                }
            ],
            "parameters": {
                "candidateCount": 1,
                "maxOutputTokens": 1024,
                "temperature": self.temperature,
                "topP": 0.8,
                "topK": 5
            }
        }

        return url, headers, package

    def _preprocess_anthropic(self, data):
        # https://console.cloud.google.com/vertex-ai/publishers/anthropic/model-garden/claude-3-haiku
        system_msg = ''

        if len(data) > 1 and data[0].role == SystemMessage("").role:
            system_msg = data[0].content
            data = data[1:]
        _messages = [m.prepare_for_generation(role_mapping=self.role_mapping_anthropic) for m in data]

        model_name = self._get_model_remap()

        # api_endpoint = self.api_endpoint
        # location = 'us-central1'
        if 'opus' in model_name:
            location = 'us-east5'
            api_endpoint = f'{location}-aiplatform.googleapis.com'
        elif 'sonnet' in model_name:
            if '-3-5-' in model_name:
                locations = ['us-east5', 'europe-west1']
                np.random.shuffle(locations)
                location = locations[0]  # random location
            else:
                location = 'us-central1'
            api_endpoint = f'{location}-aiplatform.googleapis.com'
        else:
            locations = ['us-central1', 'europe-west4']
            np.random.shuffle(locations)
            location = locations[0]  # random location
            api_endpoint = f'{location}-aiplatform.googleapis.com'

        url = f'https://{api_endpoint}/v1/projects/{self.project_id}/locations/{location}/publishers/anthropic/' \
              f'models/{model_name}:rawPredict'
        headers = {
            "Authorization": f"Bearer {self.model_key}",
            "Content-type": "application/json"
        }

        package = {
            "anthropic_version": "vertex-2023-10-16",
            "system": system_msg,
            "messages": _messages,
            "max_tokens": 4096,
            "stream": False
        }

        return url, headers, package

    def _postprocess(self, response) -> AIMessage:

        if "claude-3" in self.model_name:
            return self._postprocess_anthropic(response)

        body = response.json()
        content = body['predictions'][0]['candidates'][0]['content']
        self.prompt_tokens = body["metadata"]['tokenMetadata']['inputTokenCount']['totalTokens']
        self.completion_tokens = body["metadata"]['tokenMetadata']['outputTokenCount']['totalTokens']

        return AIMessage(content)

    def _postprocess_anthropic(self, response):
        # {'id': 'msg_vrtx_01MvSZgEc6C1NaGwprAs23YD', 'type': 'message', 'role': 'assistant',
        #  'model': 'claude-3-haiku-20240307', 'stop_sequence': None, 'usage': {'input_tokens': 16, 'output_tokens': 17},
        #  'content': [{'type': 'text', 'text': "I'm ready to assist you. How can I help you today?"}],
        #  'stop_reason': 'end_turn'}
        try:
            body = response.json()
            content = body["content"][0]["text"]
            self.prompt_tokens = body["usage"]["input_tokens"]
            self.completion_tokens = body["usage"]["output_tokens"]
        except Exception as e:
            print(f'[error] {response.json()} - {e}')
            raise e

        return AIMessage(content)

    def _preprocess_v2(self, data: List[BaseMessage]):
        messages = [m.prepare_for_generation(role_mapping=self.role_mapping) for m in data]
        # somehow not able to figure out where the API expects the system instruction...
        # system_instruction = None
        # if len(messages) > 0 and messages[0]['role'] == 'system':
        #     system_instruction = messages[0]['text']
        #     messages = messages[1:]
        messages = [{'role': m['role'], 'parts': [{'text': m['text']}]} for m in messages]

        url = f'{self.api_info["api_base"]}{self.model_name}:generateContent'
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': self.model_key
        }
        package = {
            "contents": messages,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.gen_max_tokens,
                # "topP": 0.8,
                # "topK": 10
            }
        }
        # if system_instruction is not None:
        #     package['generationConfig']['system_instruction'] = system_instruction

        return url, headers, package

    def _postprocess_v2(self, response) -> AIMessage:
        try:
            body = response.json()
            content = body['candidates'][0]['content']['parts'][0]['text']
            self.prompt_tokens = body['usageMetadata']['promptTokenCount']
            self.completion_tokens = body['usageMetadata']['candidatesTokenCount']
        except Exception as e:
            print(f'error: failed to parse response - {response.content} - {e}')
            raise e

        return AIMessage(content)

    @staticmethod
    def _refresh_token(json_filename: str = 'gcp_secrets.json', expires_in: int = 3600) -> Tuple[str, float]:
        # https://developers.google.com/identity/protocols/oauth2/service-account
        # https://www.jhanley.com/blog/
        # https://stackoverflow.com/a/53926983/3723434

        with open('secrets.json', 'r') as f:
            local_key_data = json.load(f)

        # first check if key was recently updated
        last_updated = local_key_data.get('google', {}).get('updated_at', -1)
        if last_updated is not None and last_updated > 0:
            since_update = time.time() - last_updated
            if since_update < 60 * 30:
                print(f'using existing gcp key - created {since_update/60.: .2f} min ago')
                return local_key_data['google']['key'], last_updated

        scopes = "https://www.googleapis.com/auth/cloud-platform"
        with open(json_filename, 'r') as f:
            cred = json.load(f)

        # Google Endpoint for creating OAuth 2.0 Access Tokens from Signed-JWT
        auth_url = "https://www.googleapis.com/oauth2/v4/token"
        issued = int(time.time())
        expires = issued + expires_in  # expires_in is in seconds

        # JWT Headers
        additional_headers = {
            'kid': cred['private_key'],
            "alg": "RS256",
            "typ": "JWT"  # Google uses SHA256withRSA
        }

        # JWT Payload
        payload = {
            "iss": cred['client_email'],  # Issuer claim
            "sub": cred['client_email'],  # Issuer claim
            "aud": auth_url,  # Audience claim
            "iat": issued,  # Issued At claim
            "exp": expires,  # Expire time
            "scope": scopes  # Permissions
        }

        # Encode the headers and payload and sign creating a Signed JWT (JWS)
        signed_jwt = jwt.encode(payload, cred['private_key'], algorithm="RS256", headers=additional_headers)
        auth_url = "https://www.googleapis.com/oauth2/v4/token"
        params = {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": signed_jwt
        }

        res = requests.post(url=auth_url, data=params)
        try:
            token = res.json()['access_token']
        except Exception as e:
            token = None
            print(f'error: unable to retrieve access token - {e}')
        else:
            print(f'successfully refreshed token')

        # update the key
        local_key_data['google']['key'] = token
        updated_at = time.time()
        local_key_data['google']['updated_at'] = updated_at

        # write the new version of the dictionary
        with open('secrets.json', 'w') as f:
            json.dump(local_key_data, f, indent=4)

        return token, updated_at


def test_model(model_name='chat-bison', system_msg='', msg='Test 1, 2, 3', provider='google'):
    print(f'perform google api test...:{provider}, {model_name}: {system_msg}, {msg}')
    if len(system_msg) > 0:
        test_messages = [SystemMessage(system_msg), HumanMessage(msg)]
    else:
        test_messages = [HumanMessage(msg)]
    model = GoogleModel(model_name=model_name, model_provider=provider)
    print(model(test_messages))


if __name__ == "__main__":
    fire.Fire(test_model)
