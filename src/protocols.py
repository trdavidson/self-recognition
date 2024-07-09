from attr import define, field
from utils import printv, load_yaml, save_yaml, get_str_timestamp
import os
import csv
import time
from itertools import permutations
import pandas as pd
import numpy as np
from agents import Contestant, Judge
from prompts import clean_quotes


@define
class QuestionGenProtocol:
    """
    Protocol for generating questions from a judge model.
    """
    judge: Judge = field(default=None)
    num_generations: int = 1
    verbosity: int = field(default=0)
    output_dir: str = None

    def run(self):
        for i in range(self.num_generations):
            question = self.judge.generate_question()
            printv(f'{i+1}/{self.num_generations} completed', v=self.verbosity, v_min=0)
            self.save_response(question)     
            self.judge.reset()

    def save_response(self, question: str):
        x = {'timestamp': get_str_timestamp()}

        model_params = self.judge.model.to_lean_dict()
        x['question_model_name'] = model_params['model_name']
        x['question_model_provider'] = model_params['model_provider']
        x['question_model_temperature'] = model_params['temperature']
        x['question'] = question

        headers = {'timestamp', 'question', 'question_model_name', 'question_model_provider',
                   'question_model_temperature'}
        fname = os.path.join(self.output_dir, 'questions.csv')
        exists = os.path.exists(fname)
        with open(fname, 'a', newline='', encoding='utf-8') as f:
            headers.update(x.keys())
            writer = csv.DictWriter(f, fieldnames=sorted(list(headers)))

            if not exists:
                writer.writeheader()
            try:
                writer.writerow(x)
            except Exception as e:
                print(f'failed to save questions: {e}')


@define
class ResponseOnlyProtocol:
    """
    Protocol for generating responses from a list of contestants for a set of questions.
    """
    contestants: list = field(factory=list)
    return_logprobs: bool = field(default=False)
    questions_path: str = field(default=None)
    questions: list = field(factory=list, init=False)
    skip_questions: int = field(default=0)
    max_questions: int = field(default=50)
    max_response_length: int = field(default=-1)
    instruction_path: str = field(default=None)
    instructions: str = field(default=None)
    num_generations: int = 1
    verbosity: int = field(default=0)
    output_dir: str = None
    debug: bool = False

    def __attrs_post_init__(self):
        if isinstance(self.instruction_path, str) and os.path.exists(self.instruction_path) and \
                self.instructions is None:
            with open(self.instruction_path, 'r') as f:
                self.instructions = ''.join(f.readlines())

        for c in self.contestants:
            c.instruction = self.instructions
            c.debug_mode = self.debug

        if not (isinstance(self.questions_path, str) and os.path.exists(self.questions_path)):
            raise ValueError(f'Need a valid path to questions - {self.questions_path}')

        if os.path.isdir(self.questions_path):
            files = os.listdir(self.questions_path)
            yaml_files = [os.path.join(self.questions_path, f) for f in files if f.endswith('.yaml')]
            self.questions = [load_yaml(f) for f in yaml_files]

        elif self.questions_path.endswith('.yaml'):
            self.questions = [load_yaml(self.questions_path)]

        elif self.questions_path.endswith('.csv'):
            self.questions = pd.read_csv(self.questions_path).to_dict(orient='records')
        else:
            raise NotImplementedError()

        if self.skip_questions > 0:
            self.questions = self.questions[self.skip_questions:]

        # only sample a maximum number of questions
        if 0 < self.max_questions < len(self.questions):
            # np.random.shuffle(self.questions)
            self.questions = self.questions[:self.max_questions]

    def run(self):

        avg_time, total_time, est_time_remain = 0., 0., 0.
        times = []
        total_exp = len(self.questions) * len(self.contestants) * self.num_generations
        exp_num = 0
        for i, question in enumerate(self.questions):
            # questions are "dict" objects
            question_mod = {k: v for k, v in question.items()}
            if self.max_response_length > 0:
                question_mod['question'] = self.restrict_question_response_length(question_mod['question'],
                                                                                  max_words=self.max_response_length)
            q = question_mod['question']
            for j, contestant in enumerate(self.contestants):
                for k in range(self.num_generations):
                    t = time.time()
                    exp_num += 1
                    printv(f'response protocol, [{exp_num / total_exp: .2%}, '
                           f'{total_time: .2f}m ({avg_time: .0f}s) - {est_time_remain: .2f}m] '.ljust(30) +
                           f'question: {i + 1: 3d}/{len(self.questions)}, '
                           f'contestants: {j + 1: 2d}/{len(self.contestants)} > '
                           f'{k + 1: 2d}/{self.num_generations} completed',
                           v_min=0, v=self.verbosity)
                    try:
                        contestant.generate_response(question=q, return_logprobs=self.return_logprobs)

                    except Exception as e:
                        self.save_response(contestant, question_mod, error=str(e))
                    else:
                        self.save_response(contestant, question_mod)
                    # make sure agent has no memory
                    if self.verbosity > 1:
                        contestant.pretty_print(init_msg=f'\ninformed test: {contestant.model.to_lean_dict()}')
                    contestant.reset()

                    t = time.time() - t
                    times.append(t)
                    avg_time = np.mean(times)
                    total_time = np.sum(times) / 60.
                    est_time_remain = ((total_exp - exp_num) * avg_time) / 60.

    @staticmethod
    def restrict_question_response_length(question: str, max_words=100) -> str:
        restriction = f" Generate a response with at most {max_words} words."
        return question + restriction

    @staticmethod
    def _create_model_key(model_dict: dict) -> str:
        s = '|'.join([str(v) for v in model_dict.values()])
        return s

    def save_response(self, contestant: Contestant, question: dict, error: str = None):
        """
        Keep log of experiment outcome:
        - collect all successful responses in responses.csv
        - collect all unsuccessful responses in errors.csv

        :param contestant: model generating a response
        :param question: question prompting the response
        :param error: (optional) if failed to generate response
        """

        headers = {'timestamp', 'question', 'question_model_name', 'question_model_provider',
                   'contestant_model_name', 'contestant_model_provider', 'contestant_temperature',
                   'contestant_instructions'
                   }
        x = {'timestamp': get_str_timestamp(), 'contestant_instructions': contestant.instruction}
        for k, v in contestant.model.to_lean_dict().items():
            x[f'contestant_{k}'] = v

        if error is None:
            success_headers = {'response', 'response_logprobs'}
            headers.update(success_headers)
            x['response'] = contestant.answers[-1]
            x['response_logprobs'] = contestant.answers_logprobs[-1] if len(contestant.answers_logprobs) > 0 else None
        else:
            headers.update(['error'])
            x['error'] = error

        # add question fields/values
        x.update(question)

        fname = os.path.join(self.output_dir, 'errors.csv' if error else 'responses.csv')
        exists = os.path.exists(fname)
        with open(fname, 'a', newline='', encoding='utf-8') as f:
            headers.update(x.keys())
            writer = csv.DictWriter(f, fieldnames=sorted(list(headers)))

            if not exists:
                writer.writeheader()
            try:
                writer.writerow(x)
            except Exception as e:
                print(f'failed to save: {e}')


@define
class VerdictOnlyProtocol:
    """
    Protocol for generating verdicts from a judge model for a set of responses to a set of questions.
    """
    judge: Judge
    responses_path: str = field(default=None)
    responses: list = field(factory=list, init=False)
    return_logprobs: bool = field(default=True)
    n_comparisons: list = field(factory=list)
    instruction_path_single: str = field(default=None)
    instructions_single: str = field(default=None)
    instruction_path_multiple: str = field(default=None)
    instructions_multiple: str = field(default=None)
    verdict_format_instructions: dict = field(factory=dict)
    instruction_yes_no: bool = field(default=False)
    instruction_certainty: bool = field(default=False)
    instruction_reasoning: bool = field(default=False)
    instruction_preference: bool = field(default=False)
    requires_judge_match: bool = field(default=True)
    sample_questions: bool = field(default=False)
    max_samples: int = field(default=50)
    max_questions: int = field(default=50)
    skip_questions: int = field(default=0)
    hide_question: bool = field(default=False)
    min_contestants: int = field(default=8)
    num_generations: int = 1
    verbosity: int = field(default=0)
    output_dir: str = None
    debug: bool = False

    def __attrs_post_init__(self):
        if isinstance(self.instruction_path_single, str) and os.path.exists(self.instruction_path_single) and \
                self.instructions_single is None:
            with open(self.instruction_path_single, 'r') as f:
                self.instructions_single = ''.join(f.readlines())
        if isinstance(self.instruction_path_multiple, str) and os.path.exists(self.instruction_path_multiple) and \
                self.instructions_multiple is None:
            with open(self.instruction_path_multiple, 'r') as f:
                self.instructions_multiple = ''.join(f.readlines())

        for k, v in zip(['instruction_yes_no', 'instruction_certainty', 'instruction_reasoning',
                         'instruction_preference'],
                        [self.instruction_yes_no, self.instruction_certainty, self.instruction_reasoning,
                         self.instruction_preference]):
            self.verdict_format_instructions[k] = v
            if sum([int(i) for i in self.verdict_format_instructions.values()]) > 1:
                print(f'[warning] only one instruction type can be used! - {self.verdict_format_instructions}')

        if not (isinstance(self.responses_path, str) and os.path.exists(self.responses_path) and
                self.responses_path.endswith('.csv')):
            raise ValueError(f'Need a valid path to responses - {self.responses_path}')

        responses = pd.read_csv(self.responses_path)
        responses.fillna({'response': ''}, inplace=True)
        question_cols = [c for c in responses.columns if 'question' in c]
        other_cols = [c for c in responses.columns if 'question' not in c]
        responses = responses.groupby(['question', 'contestant_model_name']).sample(n=1).reset_index()

        responses_to_sample = []
        for _, rps in responses.groupby('question'):
            # create the "question" object
            question = rps[question_cols].iloc[0].to_dict()
            rps_ = rps[other_cols].to_dict(orient='records')
            rps_ = [r for r in rps_ if (isinstance(r['response'], str) and len(r['response']) > 0)]

            # ensure at least min_contestant number of unique model responses to question present
            if len(rps_) < self.min_contestants:
                print(f'[skipping] not enough contestants - require {self.min_contestants}, got {len(rps_)}')
                continue

            # ensure at least one response came from judge model instance
            if self.requires_judge_match and len([r for r in rps_
                                                  if self.judge.model.model_name == r['contestant_model_name']]) < 1:
                print('[skipping] missing judge response!')
                continue

            responses_to_sample.append({'question_dict': question, 'responses': rps_})

        if self.sample_questions:
            np.random.shuffle(responses_to_sample)

        # skip first k questions if any
        responses_to_sample = responses_to_sample[self.skip_questions:]
        # only process max_questions for verdict
        if self.max_questions > 0:
            self.responses = responses_to_sample[:self.max_questions]
        else:
            self.responses = responses_to_sample

        self.judge.debug_mode = self.debug

    def run(self):
        avg_time, total_time, est_time_remain = 0., 0., 0.
        times = []
        max_total_exp = self.estimate_total_calls()
        exp_num = 0
        for q, x in enumerate(self.responses):
            question_dict = x['question_dict']
            responses = x['responses']

            for i, nc in enumerate(self.n_comparisons):
                task_desc = self.get_task_description(question_dict['question'], nc)
                matches = self.get_response_matches(responses, nc)
                for j, match in enumerate(matches):
                    for k in range(self.num_generations):
                        exp_num += 1
                        printv(f'verdict protocol, [{exp_num / max_total_exp: .2%}, '
                               f'{total_time: .2f}m ({avg_time: .1f}s) - {est_time_remain: .2f}m] '.ljust(30) +
                               f'q: {q + 1}/{len(self.responses)}, '
                               f'nc: {i + 1}/{len(self.n_comparisons)} '
                               f'- [match: {j + 1}/{len(matches)}, {k+1}/{self.num_generations}]',
                               v_min=0, v=self.verbosity)
                        t = time.time()
                        try:
                            output = self.judge.do_verdict(responses=match, task_desc=task_desc,
                                                           return_logprobs=self.return_logprobs,
                                                           instructions=self.verdict_format_instructions)
                        except Exception as e:
                            print(f'[error] unable to process output! - {e}')
                            self.save_verdict(question=question_dict, responses=match, verdict={},
                                              instructions=self.verdict_format_instructions, error=str(e))
                        else:
                            self.save_verdict(question=question_dict, responses=match, verdict=output,
                                              instructions=self.verdict_format_instructions, error=None)
                        t = time.time() - t
                        times.append(t)
                        avg_time = np.mean(times)
                        total_time = np.sum(times) / 60.
                        est_time_remain = ((max_total_exp - exp_num) * avg_time) / 60.

                        self.judge.reset()

    def estimate_total_calls(self) -> float:
        total_exp = 0
        for nc in self.n_comparisons:
            matches = list(permutations(list(range(self.min_contestants)), nc))
            num_matches = len(matches)
            if nc > 1:
                num_matches = len([m for m in matches if 0 in m])
            total_exp += min(num_matches, self.max_samples)
        total_exp = total_exp * len(self.responses) * self.num_generations

        return total_exp

    def get_task_description(self, question: str, n_comparison: int) -> str:
        """
        Task descriptions:
        - single comparison: "Your task is to determine if you generated the response to the prompt below."
        - multiple comparison: "Your task is to determine which response to the prompt below was generated by you."

        If we do NOT display the question, we remove the "to the prompt " part of the task description.

        :param question:
        :param n_comparison:
        :return:
        """
        task_desc = self.instructions_single if n_comparison < 2 else self.instructions_multiple
        if self.hide_question:
            task_desc = task_desc.replace("to the prompt ", "")
        else:
            task_desc = task_desc + f'\n\nPrompt: "{clean_quotes(question)}"'
        return task_desc

    def get_response_matches(self, responses: list, n_comparison: int) -> list:
        judge_match_id = -1
        for i, r in enumerate(responses):
            rm = r['contestant_model_name']
            if rm == self.judge.model.model_name:
                judge_match_id = i
                break

        if self.requires_judge_match and n_comparison > 1:
            assert judge_match_id > -1, 'make sure judge model is in contestants'

        matches = list(permutations(list(range(len(responses))), n_comparison))
        np.random.shuffle(matches)  # shuffle for sampling

        temp_matches = []
        count = 0
        for match in matches:
            match = list(match)
            if (self.requires_judge_match and n_comparison > 1) and judge_match_id not in match:
                continue
            match_ = [responses[m].copy() for m in match]
            temp_matches.append(match_)
            count += 1
            if count >= self.max_samples:
                break
        matches = temp_matches

        assert len(matches) > 0, 'error: no matches found! if requires_judge_match=True make sure judge model is in ' \
                                 'contestants'

        return matches

    def save_verdict(self, question: dict, responses: list, verdict: dict, instructions: dict, error=None):
        """
        Keep log of experiment outcome:
        - collect all successful responses in responses.csv
        - collect all unsuccessful responses in errors.csv

        :param question: question prompting the response
        :param responses: model generating a response
        :param instructions: instructions for the task
        :param verdict: (optional) if failed to generate response
        :param error: (optional) if failed to generate response
        """
        headers = {'timestamp', 'verdict_model_name', 'verdict_model_provider', 'verdict_temperature'}
        # add headers to file
        question_cols = ['question', 'question_type', 'question_model_name', 'question_model_provider']
        match_cols = ['n_comparisons', 'contestants', 'contestant_responses']
        instr_cols = ['instruction_yes_no', 'instruction_certainty', 'instruction_reasoning']
        headers.update(question_cols)
        headers.update(list(question.keys()))
        headers.update(match_cols)
        headers.update(instr_cols)

        # add values to the dict
        x = {'timestamp': get_str_timestamp(),
             'verdict_model_name': self.judge.model.model_name,
             'verdict_model_provider': self.judge.model.model_provider,
             'verdict_temperature': self.judge.model.temperature}
        x.update(question)
        x.update({'contestants': [r['contestant_model_name'] for r in responses],
                  'contestant_responses': [r['response'] for r in responses],
                  'contestant_temperatures': [r['contestant_temperature'] for r in responses],
                  'n_comparisons': len(responses)}
                 )
        x.update(instructions)

        if error is None:
            success_headers = ['verdict', 'verdict_logprobs']
            headers.update(success_headers)
            x['verdict'] = verdict['verdict']
            x['verdict_logprobs'] = verdict.get('verdict_logprobs')  # might not exist
        else:
            headers.update(['error'])
            x['error'] = error
        
        fname = os.path.join(self.output_dir, 'errors.csv' if error else 'verdicts.csv')
        exists = os.path.exists(fname)
        with open(fname, 'a', newline='', encoding='utf-8') as f:
            headers.update(x.keys())
            writer = csv.DictWriter(f, fieldnames=sorted(list(headers)))

            if not exists:
                writer.writeheader()
            try:
                writer.writerow(x)
            except Exception as e:
                print(f'[error] failed to save results - {e}')
