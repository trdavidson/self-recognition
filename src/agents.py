from copy import deepcopy
from attr import attrs, field
from models.model_utils import SystemMessage, HumanMessage, AIMessage, BaseMessage, ChatModel, MessagesList
from utils import printv, read_txt
from prompts import QuestionPrompt, AnswerPrompt, DiscriminatorPrompt


@attrs
class BaseAgent:
    model: ChatModel = field(default=None)
    verbosity: int = field(default=1)
    debug_mode: bool = field(default=False)
    init_settings = locals()
    history: MessagesList = field(factory=MessagesList)

    def __attrs_post_init__(self):
        self.__sub_post_init__()

    def generate(self, messages) -> str:
        content = self.model(messages)
        return content

    def get_settings(self):
        return self.init_settings

    def __getitem__(self, key):
        return getattr(self, key)
    
    def pretty_print(self, init_msg: str = None, post_msg: str = None):
        if init_msg is not None:
            printv(init_msg, v=1, c='green')
        for msg in self.history:
            if isinstance(msg, SystemMessage):
                printv(msg, v=1, c='red')
            elif isinstance(msg, HumanMessage):
                printv(msg, v=1, c='white')
            elif isinstance(msg, AIMessage):
                printv(msg, v=1, c='blue')
        if post_msg is not None:
            printv(post_msg, v=1, c='green')

    @staticmethod
    def _flatten_messages(messages):
        # convert messages from BaseMessage -> {role: "", content: ""}
        return [{'role': msg.role, 'content': msg.content} for msg in messages]

    @staticmethod
    def _unflatten_messages(messages):
        history = MessagesList()
        for msg in messages:
            history.append(BaseMessage(**msg))
        return history
    
    def __sub_post_init__(self):
        # add any additional post init steps here in the subclasses - do NOT use __attrs_post_init__!
        pass

    def copy(self):
        c = BaseAgent()
        c = self.base_copy(c)

        return c

    def base_copy(self, c):
        c.model = self.model.copy()
        c.verbosity = self.verbosity
        c.debug_mode = self.debug_mode
        c.history = deepcopy(self.history)

        return c
    
    def __eq__(self, other):
        return self.model == other.model

        
@attrs
class Contestant(BaseAgent):
    """
    A contestant agent will be called in the GeneratorProtocol to generate responses to questions.
    (base case) The contestant will generate a response to a question.
    (instruct case) The contestant will generate a response to a question with an instruction.
    """
    initialization_prompt_path: str = field(default='data/prompts/basic_judge_static_single_response.txt')
    initialization_prompt: str = field(default=None)
    n_words: int = field(default=None)
    instruction: str = field(default=None)
    answers: list = field(factory=list)
    answers_logprobs: list = field(factory=list)
    
    def generate_response(self, question: str, return_logprobs=False) -> str:

        a_prompt = AnswerPrompt(instruction=self.instruction).format_prompt(question)
        self.history.append(a_prompt)
        if self.debug_mode:
            answer = ('dummy', {'logprobs': [0.99]}) if return_logprobs else 'dummy'
        else:
            answer = self.model(self.history, return_logprobs=return_logprobs)

        if return_logprobs:
            answer, logprobs = answer
            self.answers_logprobs.append(logprobs)

        self.answers.append(answer)
        self.history.append(AIMessage(answer))
        return answer

    def reset(self):
        self.answers = []
        self.answers_logprobs = []
        self.history = []

    def copy(self):
        c = Contestant(instruction=self.instruction, answers=deepcopy(self.answers))
        c = self.base_copy(c)

        return c


@attrs
class Judge(BaseAgent):
    initialization_prompt_path: str = field(default='data/prompts/basic_judge_static_single_response.txt')
    initialization_prompt: str = field(default=None)
    questions: list = field(factory=list)
    questions_logprobs: list = field(factory=list)
    verdicts: list = field(factory=list)
    verdicts_logprobs: list = field(factory=list)

    def __sub_post_init__(self):
        self.initialization_prompt = read_txt(self.initialization_prompt_path)

    def copy(self):
        c = Judge(initialization_prompt_path=self.initialization_prompt_path,
                  questions=deepcopy(self.questions),
                  verdicts=deepcopy(self.verdicts)
                  )
        c = self.base_copy(c)
        return c

    def generate_question(self, return_logprobs=False) -> str:
        q_prompt = self.initialization_prompt
        q_prompt = QuestionPrompt().format_prompt(q_prompt)
        self.history.append(q_prompt)
        question = self.model(self.history, return_logprobs=return_logprobs)
        if return_logprobs:
            question, question_logprobs = question
            self.questions_logprobs.append(question_logprobs)
        self.questions.append(question)
        self.history.append(AIMessage(question))
            
        return question

    def do_verdict(self, responses: list, task_desc: str, instructions, return_logprobs=True):

        if self.debug_mode:
            return {'verdict': 'dummy', 'verdict_logprobs': {0.99}}

        rs = [r['response'] for r in responses]
        v_prompt = DiscriminatorPrompt(**instructions).format_prompt(rs, preamble=task_desc)
        # printv(v_prompt, debug=True)
        verdict = self.model([v_prompt], return_logprobs=return_logprobs)
        verdict_logprobs = None
        if return_logprobs and isinstance(verdict, tuple):
            verdict, verdict_logprobs = verdict

        output = {
            'verdict': verdict,
        }

        if return_logprobs:
            output['verdict_logprobs'] = verdict_logprobs

        return output

    def reset(self):
        self.questions = []
        self.questions_logprobs = []
        self.verdicts = []
        self.verdicts_logprobs = []
        self.history = []
