from typing import List, Union, Any
import tiktoken
import yaml
import inspect
from datetime import datetime as dt
import attr
from attr import attrs, field, define
import sys
sys.path.append("src/")

from utils import get_api_key, printv


@attrs
class BaseMessage:
    role: str = attr.ib()
    content: str = attr.ib()
    ext_visible: bool = field(default=True)
    alt_role: str = field(default=None)

    def format_prompt(self, before, to):
        self.content = self.content.replace("{" + before + "}", f"{to}")
        return self

    def prepare_for_generation(self, role_mapping=None):
        default_mapping = {'role': 'role', 'content': 'content', 'assistant': 'assistant', 'user': 'user',
                           'system': 'system'}
        if role_mapping is None:
            role_mapping = default_mapping

        _role = role_mapping.get(self.role, default_mapping.get(self.role))
        msg = {role_mapping.get('role'): _role, role_mapping.get('content'): self.content}

        return msg
    
    def prepare_for_completion(self):
        return self.content

    def text(self):
        return self.content

    def copy(self):
        if isinstance(self, SystemMessage):
            c = SystemMessage
        elif isinstance(self, AIMessage):
            c = AIMessage
        elif isinstance(self, HumanMessage):
            c = HumanMessage
        else:
            c = BaseMessage

        return c(role=self.role, content=self.content, alt_role=self.alt_role)

    def __str__(self):
        return self.content

    def __getitem__(self, key):
        return self.__dict__[key]

    def __eq__(self, other):
        return self.role == other.role and self.content == other.content


@attrs
class AIMessage(BaseMessage):
    role: str = attr.ib(default="assistant")


@attrs
class HumanMessage(BaseMessage):
    role: str = attr.ib(default="user")


@attrs
class SystemMessage(BaseMessage):
    role: str = attr.ib(default="system")

    def __add__(self, otherSystem):
        return SystemMessage(self.content + "\n" + otherSystem.content)


@define
class ChatModel:
    """
    Basic LLM API model wrapper class

    # TODO: (1) error handling
    #       (2) expand support
    #       (3) move to aiohttp REST API calls
    #       (4) add streaming mode
    # prepare payload
    # make the call
    # process output
    """
    model_provider: str = field(default='azure')
    model_name: str = field(default='gpt-3.5-turbo')
    model_key: Any = field(default=None)
    model_key_path: Any = field(default='secrets.json')
    model_key_name: Any = field(default=None)
    model: Any = field(default=None)
    role_mapping: dict = field(factory=dict)

    debug_mode: bool = field(default=False)
    temperature: float = field(default=0.0)
    generation_params: dict = field(factory=dict)
    context_max_tokens: int = field(default=1024)
    gen_max_tokens: int = 4096
    prompt_cost: float = 0
    completion_cost: float = 0
    tpm: int = 0
    rpm: int = 0
    api_info: dict = field(factory=dict)
    messages: list = field(factory=list)
    response: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # keep track of costs
    budget: float = 10.
    session_prompt_costs: float = 0.
    session_completion_costs: float = 0.

    def __attrs_post_init__(self):

        if self.model_key is None:
            self.model_key = get_api_key(fname=self.model_key_path, provider=self.model_provider,
                                         key=self.model_key_name)

        # get model details
        model_details = get_model_details(self.model_name)
        self.context_max_tokens = model_details['max_tokens']
        self.gen_max_tokens = model_details.get("max_gen_tokens", self.context_max_tokens)
        self.prompt_cost = model_details['prompt_cost']
        self.completion_cost = model_details['completion_cost']
        self.tpm = model_details['tpm']
        self.rpm = model_details['rpm']
        # get api info
        self.api_info = get_api_settings(self.model_provider)

        # only single generations currently implemented
        if self.generation_params.get("n", 1) >= 2:
            raise NotImplementedError("Need to implement for more than one generation.")

        self._custom_post_init()

    def _custom_post_init(self):
        pass

    def __call__(self, messages: List[BaseMessage], return_logprobs: bool = False, return_top_logprobs: int = 0,
                 return_prompt_logprobs: bool = False) -> Union[str, tuple]:
        """
        Generate tokens and optionally logprobs per token. Note, not all API models support logprobs!.
        """
        if self.debug_mode:
            # time.sleep(0.2)  # small wait to see sensible msg timestamps
            return f"<{dt.strftime(dt.now(), '%H%M%S_%f')}> lorem ipsum dolor sit amet"

        self.prompt_tokens, self.completion_tokens = -1, -1

        # check if subclass supports logprobs and top_logprobs
        signature = inspect.signature(self._generate)
        supports_logprobs = 'return_logprobs' in signature.parameters
        supports_top_logprobs = 'return_top_logprobs' in signature.parameters
        supports_prompt_logprobs = 'return_prompt_logprobs' in signature.parameters

        if self.model_provider == 'azure':
            supports_logprobs = False
            supports_prompt_logprobs = False
            supports_top_logprobs = False

        # NOTE: comment out for now for ease of experimenting
        # if return_logprobs and not supports_logprobs:
        #     raise ValueError(f"Model {self.model_name} does not support logprobs.")
        # if return_prompt_logprobs and not supports_prompt_logprobs:
        #     raise ValueError(f"Model {self.model_name} does not support prompt logprobs.")
        # if return_top_logprobs and not supports_top_logprobs:
        #     raise ValueError(f"Model {self.model_name} does not support top logprobs.")

        if supports_logprobs and supports_top_logprobs and supports_prompt_logprobs:
            response = self._generate(messages, return_logprobs=return_logprobs,
                                      return_top_logprobs=return_top_logprobs,
                                      return_prompt_logprobs=return_prompt_logprobs)
        elif supports_logprobs and supports_prompt_logprobs:
            response = self._generate(messages, return_logprobs=return_logprobs,
                                      return_prompt_logprobs=return_prompt_logprobs)
        elif supports_logprobs and supports_top_logprobs:
            response = self._generate(messages, return_logprobs=return_logprobs,
                                      return_top_logprobs=return_top_logprobs)
        else:
            response = self._generate(messages)

        # update internal books:
        response_content = response[0] if isinstance(response, tuple) else response.content
        self._update_token_counts(prompt_messages=messages, completion=response_content)

        # update message history
        self.messages = messages

        if isinstance(response, tuple):
            return response[0].content, response[1]
        else:
            return response.content

    def _generate(self, data, **kwargs) -> Union[AIMessage, dict]:
        pass

    def _preprocess(self, data: List[BaseMessage]):
        pass

    def _postprocess(self, data) -> AIMessage:
        pass

    def _update_token_counts(self, prompt_messages, completion):
        try:
            prompt_tokens = self.estimate_tokens(messages=prompt_messages) if self.prompt_tokens < 0 else self.prompt_tokens
            completion_tokens = self.estimate_tokens(completion) if self.completion_tokens < 0 else self.completion_tokens

            # keep track of budget and costs
            pc, cc, _ = self.estimate_cost(input_tokens=prompt_tokens, output_tokens=completion_tokens)
            self.session_prompt_costs += pc
            self.session_completion_costs += cc
            self.budget -= (pc + cc)
        except Exception as e:
            print(f'warning: unable to update token counts - {e}')

    def history(self):
        # optionally keep a history of interactions
        return self.messages

    def estimate_cost(self, input_tokens: int, output_tokens: int):
        """
        Basic cost estimation
        """
        input_cost = (input_tokens / 1000) * self.prompt_cost
        output_cost = (output_tokens / 1000) * self.completion_cost
        total_cost = input_cost + output_cost

        return input_cost, output_cost, total_cost

    def estimate_tokens(self, messages: Union[List[BaseMessage], str]):
        if "llama" in self.model_name:
            return 0
        try:
            enc = tiktoken.encoding_for_model(self.model_name)
        except KeyError as e:
            enc = tiktoken.encoding_for_model("gpt-4")
        msg_token_penalty = 8
        str_to_estimate = ''

        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict):
                    str_to_estimate += m["content"]
                elif isinstance(m, BaseMessage):
                    str_to_estimate += m.text()
                else:
                    str_to_estimate += ''
        elif isinstance(messages, str):
            str_to_estimate = messages
        else:
            str_to_estimate = ''

        tokens = len(enc.encode(str_to_estimate)) + msg_token_penalty

        return tokens

    def check_context_len(self, context: List[BaseMessage], max_gen_tokens: int) -> bool:
        # TODO: currently not checking context length anymore!!!
        """Calculate how many tokens we have left. 
        
        messages: List[system_msg, msg_history, note_history, prev_game_history) + note/msg]
        
        Returns:
            got_space (bool)
        """
        # 1. enough context token space to generate note?
        # 2. enough context token space to generate msg?

        # somde models prepend tokens, e.g., openai adds 8 default tokens per request¨
        msg_token_penalty = 8
        context_tokens = self.estimate_tokens(context)
        tokens_left = self.context_max_tokens - (context_tokens + msg_token_penalty + max_gen_tokens)
        got_space = tokens_left > 0

        return got_space

    def __repr__(self):
        return f'ChatModel("model_name"={self.model_name}, "model_provider"={self.model_provider})'

    def to_lean_dict(self):
        lean_dict = {
            'model_name': self.model_name,
            'model_provider': self.model_provider,
            'temperature': self.temperature
        }
        return lean_dict

    def __eq__(self, other):
        # TODO: figure out why this doesn't work
        return (self.model_name == other.model_name and self.model_provider == other.model_provider
                and self.temperature == other.temperature)

    def copy(self):
        m = self.__class__(model_provider=self.model_provider, model_name=self.model_name, model_key=self.model_key,
                           model_key_path=self.model_key_path, model_key_name=self.model_key_name, model=self.model,
                           role_mapping=self.role_mapping, debug_mode=self.debug_mode, temperature=self.temperature,
                           generation_params=self.generation_params, context_max_tokens=self.context_max_tokens,
                           prompt_cost=self.prompt_cost, completion_cost=self.completion_cost,
                           tpm=self.tpm, rpm=self.rpm, api_info=self.api_info, messages=self.messages,
                           response=self.response, prompt_tokens=self.prompt_tokens,
                           completion_tokens=self.completion_tokens, budget=self.budget,
                           session_prompt_costs=self.session_prompt_costs,
                           session_completion_costs=self.session_completion_costs)
        return m


def get_model_pricing(model_name):
    model_details = get_model_details(model_name=model_name)
    return model_details['prompt_cost'], model_details['completion_cost']


def get_model_details(model_name, fpath='data/llm_model_details.yaml'):
    try:
        with open(fpath) as f:
            details = yaml.safe_load(f)
            # print(details)
    except Exception as e:
        print(f'error: unable to load model details - {e}')
        details = {}

    models = details.keys()
    if model_name not in models:
        raise KeyError(f'error: no details available for model {model_name} - pick one of {models}')

    return details[model_name]


def get_api_settings(api_provider, fpath='data/api_settings/apis.yaml'):
    try:
        with open(fpath) as f:
            details = yaml.safe_load(f)
    except Exception as e:
        print(f'error: unable to load model details - {e}')
        details = {}

    models = details.keys()
    if api_provider not in models:
        raise KeyError(f'error: no details available for model {api_provider} - pick one of {models}')

    return details[api_provider]
    

class MessagesList(list):
    def append(self, message: BaseMessage):
        if not isinstance(message, BaseMessage):
            raise ValueError(f'error: message must be of type BaseMessage - got {type(message)}')
        if len(self) > 0 and self[-1].role == message.role:
            self[-1].content += "\n" + message.content
        else:  
            super().append(message)