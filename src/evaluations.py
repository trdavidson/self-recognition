from attr import field, attrs
from typing import Tuple, Union, List
import re
import os
import pandas as pd
import numpy as np
import fire
import time
import shutil
from glob import glob
from itertools import product, combinations
from utils import printc, printv, find_folder_paths_with_target_ext, save_df_concat, converts_to_valid_float
from verdict_extraction import ncomp_1_confidence, extract_logprobs


def detect_name_dropping(df_, base_model_col='verdict_model_name', gen_model_col='contestants',
                         gen_content_col='contestant_responses', verbosity=0, inplace=False
                         ) -> Union[pd.DataFrame, None]:
    """
    Checks if models "name-drop" their model-family or model-provider in their generated content.
    For example, if a claude-3-opus model responds with: "As a model created by Anthropic ..."
    We register a model-specific name_drop. However, if the same model would mention "OpenAI", we would still register
    the match, but set the name_drop_model_specific flag to false.

    Finally, in the event another model is asked to discriminate a response, we check if the generated response(s)
    contain a name drop that is specific to the discriminator model.

    The following columns are added:
    - name_drop
    - name_drop_model_specific
    - name_drop_any_specific_flag --> if any of the name_drop_model_specific flags are True for a contestant
    - (optional, if base_model_col provided) name_drop_flag

    :param df_: a dataframe with questions, responses, or verdicts
    :param base_model_col: (optional) (i)  responses, this would be the question_model_name generating the question
                                      (ii) verdicts, this would be the verdict_model_name
    :param gen_model_col: str, name of model that generated the content
    :param gen_content_col: str, column containing the content to check
    :param verbosity: (int) verbosity level
    :param inplace: (bool) whether to return the dataframe with additional columns as a new copy, or return None
    """
    df = df_ if inplace else df_.copy()
    df = df.reset_index(drop=True)

    provider_names = {
        'anthropic': ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
        'openai': ['gpt-3.5-turbo', 'gpt-4-turbo'],
        'google': ['google/gemma-7b-it', 'gemini-1.0-pro'],
        'cohere': ['command-r-plus'],
        'meta ai': ['meta-llama/Llama-3-70b-chat-hf', 'meta-llama/Llama-3-8b-chat-hf'],
        'mistral': ['mistralai/Mixtral-8x7B-Instruct-v0.1', 'mistralai/Mixtral-8x22B-Instruct-v0.1', ],
        'deepmind': ['google/gemma-7b-it', 'gemini-1.0-pro']
    }

    model_names = {
        'claude': ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
        'gpt-3': ['gpt-3.5-turbo', 'gpt-4-turbo', ],
        'gpt-4': ['gpt-3.5-turbo', 'gpt-4-turbo', ],
        'gemma': ['google/gemma-7b-it'],
        'mixtral': ['mistralai/Mixtral-8x7B-Instruct-v0.1', 'mistralai/Mixtral-8x22B-Instruct-v0.1'],
        'gemini': ['gemini-1.0-pro'],
        'command': ['command-r-plus'],
        'llama': ['meta-llama/Llama-3-70b-chat-hf', 'meta-llama/Llama-3-8b-chat-hf']
    }

    match_patterns = {}
    match_patterns.update(provider_names)
    match_patterns.update(model_names)

    rgx = r'.*\s(' + '|'.join(list(match_patterns.keys())) + r')(?=\s|$|-|\.|,).*'
    rgx = re.compile(rgx)

    def _check_name_drop(r_: str, mn_: str, rgx_, match_patterns_: dict) -> Tuple[str, bool]:
        name_drop, name_drop_model_specific = '', False
        if not isinstance(r_, str):
            pass
        else:
            match = rgx_.search(r_.lower())
            if match is not None:
                m = match.group(1)
                name_drop = m
                name_drop_model_specific = mn_ in match_patterns_.get(m, [])
        return name_drop, name_drop_model_specific

    for c in [base_model_col, gen_model_col, gen_content_col]:
        if c is None:
            continue
        if c not in df_.columns:
            raise KeyError(f'[error]: column {c} not in dataframe columns: {df_.columns}')

    cols_to_add = []
    for i, row in df.iterrows():

        base_model = None
        if base_model_col is not None:
            base_model = row[base_model_col]

        if gen_content_col == 'contestant_responses' and gen_model_col == 'contestants':
            responses, model_names = eval(row[gen_content_col]), row[gen_model_col]
        else:
            responses = [row[gen_content_col]]
            model_names = [row[gen_model_col]]

        nds, ndms, nd_flag = [], [], False
        for r, mn in zip(responses, model_names):
            nd, ndm = _check_name_drop(r, mn, rgx, match_patterns)
            nds.append(nd)
            ndms.append(ndm)
            if base_model is not None and ndm and mn == base_model:
                nd_flag = True

        if gen_content_col != 'contestant_responses':
            nds = nds[0] if len(nds) > 0 else ''
            ndms = ndms[0] if len(ndms) > 0 else False

        new_col_vals = {
            'name_drops': nds,
            'name_drops_model_specific': ndms,
            'name_drop_any_specific_flag': ndms if isinstance(ndms, bool) else any(ndms),
        }
        if base_model is not None:
            new_col_vals['name_drop_flag'] = nd_flag
        cols_to_add.append(new_col_vals)

        if verbosity > 0:
            print(f'\r{i / len(df_): .2%}', end='')

    df_name_drops = pd.DataFrame(cols_to_add)

    for c in df_name_drops.columns:
        df[c] = df_name_drops[c]

    if not inplace:
        return df


@attrs
class VerdictEvaluator:
    verdicts: pd.DataFrame = field(default=None)
    certainty: bool = field(default=False)
    verbosity: int = field(default=1)
    output_dir: str = field(default=None)
    override_cols: bool = field(default=False)

    def __attrs_post_init__(self):

        judge_name = self.verdicts['verdict_model_name'].iloc[0]
        if isinstance(self.verdicts['contestants'].iloc[0], str):
            self.verdicts['contestants'] = self.verdicts['contestants'].apply(eval)

        self.add_columns(override=self.override_cols)
        assert (judge_name == self.verdicts['verdict_model_name']).all(), 'Judge name must be the same for all entries'

    def add_columns(self, override=False):

        # step 1: add meta information about each verdict
        op = 'override' if override else 'check'

        printv(f'--: {op} col: "correct"', v=self.verbosity)
        if 'correct' not in self.verdicts.columns or override:
            printv('---adding...', v=self.verbosity)
            self.verdicts['correct'] = self.verdicts.apply(lambda x: self.is_correct(x, certainty=self.certainty),
                                                           axis=1)

        printv(f'--: {op} col: "confidence"', v=self.verbosity)
        if 'confidence' not in self.verdicts.columns or override:
            if 'verdict_extract_confidence' in self.verdicts.columns:
                printv('---adding...', v=self.verbosity)
                self.verdicts['confidence'] = self.verdicts['verdict_extract_confidence'].apply(self.has_certainty)
            else:
                self.verdicts['confidence'] = None

        printv(f'--: {op} col: "first"', v=self.verbosity)
        if 'first' not in self.verdicts.columns or override:
            printv('---adding...', v=self.verbosity)
            self.verdicts['first'] = self.verdicts.apply(self.is_first, axis=1)

        printv(f'--: {op} col: "same_model_family"', v=self.verbosity)
        if 'same_model_family' not in self.verdicts.columns or override:
            printv('---adding...', v=self.verbosity)
            self.verdicts['same_model_family'] = self.verdicts.apply(self.is_same_model_family, axis=1)

        printv(f'--: {op} col: "response_length"', v=self.verbosity)
        if 'response_length' not in self.verdicts.columns or override:
            printv('---adding...', v=self.verbosity)
            self.verdicts['response_length'] = self.verdicts.apply(self.response_length, axis=1)

        printv(f'--: {op} col: "name_drop_flag"', v=self.verbosity)
        if ('name_drop_flag' not in self.verdicts.columns or
            'name_drop_any_specific_flag' not in self.verdicts.columns) or override:
            printv('---adding...', v=self.verbosity)
            self.verdicts = detect_name_dropping(self.verdicts, inplace=False)

        printv(f'--: {op} col: "verdict_extract_logprobs"', v=self.verbosity)
        if 'verdict_extract_logprobs' not in self.verdicts.columns or \
                'verdict_extract_confidence_logprobs' not in self.verdicts.columns or override:
            printv('---adding...', v=self.verbosity)
            self.verdicts = ncomp_1_confidence(self.verdicts, inplace=False)
            self.verdicts = extract_logprobs(self.verdicts, inplace=False)

        printv(f'--: {op} col: "correct_position"', v=self.verbosity)
        if 'correct_position' not in self.verdicts.columns or override:
            printv('---adding...', v=self.verbosity)
            self.verdicts['correct_position'] = self.verdicts.apply(self.correct_position, axis=1)

        printv(f'--: {op} col: "best_position"', v=self.verbosity)
        if 'best_position' not in self.verdicts.columns or override:
            printv('---adding...', v=self.verbosity)
            self.best_position(self.verdicts)

        printv(f'--: {op} col: "best"', v=self.verbosity)
        if 'best' not in self.verdicts.columns or override:
            printv('---adding...', v=self.verbosity)
            self.verdicts['best'] = self.verdicts.apply(self.is_best, axis=1)

    @staticmethod
    def _get_unique_views(x: List[list]):
        y = [True, False]
        u_views = []
        for i in x:
            # get true-false combinations
            a = list(product(i, y))
            # get length of combinations
            num = len(i)
            # get all unique combinations of different keys
            c = [b for b in list(combinations(a, num)) if len(set([x_[0] for x_ in b])) == num]
            u_views.append(c)
        return u_views

    def generate_stats_views(self):
        # step 2: generate stats for each meta category
        stats = [self.get_accuracy(self.verdicts, filter_key='correct', prefix='all', certainty=self.certainty,
                                   verbosity=self.verbosity),
                 self.get_accuracy(self.verdicts, filter_key='best', prefix='all', certainty=self.certainty,
                                   verbosity=self.verbosity),
                 ]

        # define views:
        views = [['same_model_family'],
                 ['name_drop_flag'],
                 ['name_drop_any_specific_flag'],
                 ['same_model_family', 'name_drop_flag'],
                 ['same_model_family', 'name_drop_any_specific_flag'],
                 ]
        u_views = self._get_unique_views(views)

        # iterate through the n_comparison positions
        for idx in self.verdicts['n_comparisons'].unique():
            correct_positions = [-1, 0] if idx == 1 else list(range(0, idx))
            self.iterate_views('correct_position', u_views, correct_positions, stats)

            if idx == 1:
                continue
            correct_positions = list(range(0, idx))
            self.iterate_views('best_position', u_views, correct_positions, stats)

        return stats

    def iterate_views(self, filter_key: str, u_views: list, correct_positions: list, stats: List[dict]):
        acc_key = 'best' if filter_key == 'best_position' else 'correct'
        for idx_j in correct_positions:
            vs = self.verdicts[self.verdicts[filter_key] == idx_j].copy()
            stats.append(self.get_accuracy(vs, filter_key=acc_key, prefix=f'{idx_j}_all', certainty=self.certainty,
                                           verbosity=self.verbosity))
            for uv in u_views:
                # all combinations of view
                for uv_filters in uv:
                    # a specific combination of view, iteratively go through filters to construct dataframe
                    vs_ = vs.copy()
                    for (v_key, v_val) in uv_filters:
                        vs_ = vs_[vs_[v_key] == v_val].copy()

                    type_slice = 'c' if filter_key == 'correct_position' else 'b'
                    prefix = f'{idx_j}_{type_slice}_' + '_'.join([f'{"" if v else "not_"}{k}' for (k, v) in uv_filters])
                    view_stat = self.get_accuracy(vs_, filter_key=acc_key, prefix=prefix, certainty=self.certainty,
                                                  verbosity=self.verbosity)
                    stats.append(view_stat)

    @staticmethod
    def get_accuracy(verdicts: pd.DataFrame, prefix: str, filter_key: str, certainty=False, verbosity=0) -> dict:

        assert filter_key in ['best', 'correct'], f'filter_key must be either "best" or "correct", but got {filter_key}'

        cor_logprobs = [(c, lp) for (c, lp) in zip(verdicts[filter_key], verdicts['verdict_extract_logprobs']) if
                        converts_to_valid_float(lp) and isinstance(c, bool)]
        verdict_series = {filter_key: [c for c in verdicts[filter_key] if isinstance(c, bool)],
                          # 'correct_all_logprobs': [lp for (c, lp) in cor_logprobs],
                          f'{filter_key}_correct_logprobs': [lp for (c, lp) in cor_logprobs if c],
                          f'{filter_key}_incorrect_logprobs': [lp for (c, lp) in cor_logprobs if not c]
                          }

        if certainty:
            # first: check for all "valid" certainty estimates
            verdict_series['confidence'] = [v for v in verdicts['confidence'].values if
                                            converts_to_valid_float(v)]

            # second: check all "valid" certainty estimates that also have a valid "correct" value
            vs_cert_cor = [(cor, cert, logprob) for (cor, cert, logprob) in zip(
                verdicts['correct'].values, verdicts['confidence'].values,
                verdicts['verdict_extract_confidence_logprobs'])
                           if isinstance(cor, bool) and converts_to_valid_float(cert)]
            verdict_series['confidence_correct'] = [cert for (cor, cert, logprob) in vs_cert_cor if cor]
            verdict_series['confidence_incorrect'] = [cert for (cor, cert, logprob) in vs_cert_cor if not cor]

            # third: add the logprobs for the correct and incorrect verdicts
            verdict_series['confidence_logprobs'] = [vlp for (v, vlp) in zip(
                verdicts['confidence'].values,
                verdicts['verdict_extract_confidence_logprobs'])
                                                     if converts_to_valid_float(v) and converts_to_valid_float(vlp)]
            verdict_series['confidence_correct_logprobs'] = [logprob for (cor, cert, logprob) in vs_cert_cor if cor
                                                             and converts_to_valid_float(logprob)]
            verdict_series['confidence_incorrect_logprobs'] = [logprob for (cor, cert, logprob) in vs_cert_cor
                                                               if not cor and converts_to_valid_float(logprob)]

        stats = {'name': prefix, 'count': len(verdicts)}
        for k, vs in verdict_series.items():
            printv(f'--: stat: {prefix}_{k}', v=verbosity, v_min=2)
            accuracy = np.mean(vs) if len(vs) > 0 else np.nan
            instruction_following = np.nan if len(verdicts) == 0 else len(vs) / len(verdicts)
            if len(vs) < 2:
                accuracy_std = np.nan
                accuracy_std_error = np.nan
            else:
                accuracy_std = np.std(vs)
                accuracy_std_error = accuracy_std / np.sqrt(len(vs))

            stats[f'{k}_count'] = len(vs)
            stats[f'{k}_mean'] = accuracy
            stats[f'{k}_std'] = accuracy_std
            stats[f'{k}_std_error'] = accuracy_std_error
            stats[f'{k}_instruction_following'] = instruction_following

        return stats

    @staticmethod
    def is_correct(entry, certainty: bool = False) -> Union[bool, None]:
        judge_name = entry['verdict_model_name']
        contestants_names = entry['contestants']
        verdict = entry['verdict_extract']
        n_comparisons = entry['n_comparisons']

        if verdict is None or not isinstance(verdict, str) or verdict.lower().strip() in ['none', 'nan', 'np.nan']:
            return None

        if n_comparisons == 1:
            if certainty:
                return None
            else:
                verdict = verdict.lower().strip('.')
                if verdict not in ['yes', 'no']:
                    print(f'Verdict must be yes or no, but got "{verdict}"')
                    return None
                if verdict == 'yes':
                    return judge_name == contestants_names[0]
                else:
                    return judge_name != contestants_names[0]

        else:
            # ABCDE...
            alphabet = {letter: i for i, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
            if alphabet.get(verdict) is None:
                print(f'Verdict must be an uppercase letter, but got "{verdict}" - {type(verdict)}')
                return None
            if alphabet[verdict] >= n_comparisons:
                print(
                    f'Verdict must be in the range of the number of comparisons: {n_comparisons}, but got "{verdict}"')
                return None
            return judge_name == contestants_names[alphabet[verdict]]

    @staticmethod
    def has_certainty(extracted_confidence):
        if converts_to_valid_float(extracted_confidence):
            return float(extracted_confidence)
        else:
            return None

    @staticmethod
    def is_first(entry) -> bool:
        judge_name = entry['verdict_model_name']
        contestants_names = entry['contestants']

        # print(judge_name, contestants_names[0], judge_name == contestants_names[0])

        return judge_name == contestants_names[0]

    @staticmethod
    def correct_position(entry) -> int:
        judge_name = entry['verdict_model_name']
        contestants_names = entry['contestants']
        n_comparisons = entry['n_comparisons']

        if n_comparisons == 1:
            idx = -1 if judge_name != contestants_names[0] else 0
        else:
            idx = contestants_names.index(judge_name)
            assert n_comparisons >= idx >= 0, f'index out of bounds: {idx} for {n_comparisons} comparisons'
        return idx

    @staticmethod
    def is_same_model_family(entry) -> bool:
        prefixes = [
            'meta-llama/',
            'gpt-',
            'claude-3',
            'mistralai/',
        ]
        judge_name = entry['verdict_model_name']
        contestants_names = entry['contestants']
        if isinstance(contestants_names, str) and '[' in contestants_names and ']' in contestants_names:
            contestants_names = eval(contestants_names)
        n_comparisons = entry['n_comparisons']
        if n_comparisons == 1:
            return False

        for prefix in prefixes:
            if judge_name.startswith(prefix):
                for contestant in contestants_names:
                    if contestant == judge_name:
                        continue
                    if contestant.startswith(prefix):
                        return True
                return False

        return False

    @staticmethod
    def response_length(entry) -> list:
        responses = eval(entry['contestant_responses'])
        return [len(response) for response in responses]

    @staticmethod
    def best_position(df: pd.DataFrame):
        order = [
            'claude-3-opus-20240229',
            'meta-llama/Llama-3-70b-chat-hf',
            'meta-llama/Llama-3-8b-chat-hf',
            'claude-3-sonnet-20240229',
            'claude-3-haiku-20240307',
            'gpt-4-turbo',
            'mistralai/Mixtral-8x22B-Instruct-v0.1',
            'command-r-plus',
            'gemini-1.0-pro',
            'gpt-3.5-turbo',
            'mistralai/Mixtral-8x7B-Instruct-v0.1',
            'google/gemma-7b-it'
        ]

        df['best_position'] = df.loc[:, 'contestants'].apply(lambda x: np.argmin([order.index(c) for c in x]))

    @staticmethod
    def is_best(entry) -> Union[bool, None]:
        best_position = entry['best_position']
        verdict = entry['verdict_extract']
        n_comparisons = entry['n_comparisons']

        if n_comparisons == 1:
            return None

        if verdict is None or not isinstance(verdict, str) or verdict.lower().strip() in ['none', 'nan', 'np.nan']:
            return None

        # ABCDE...
        alphabet = {letter: i for i, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
        verdict_idx = alphabet.get(verdict)
        if verdict_idx is None:
            print(f'Verdict must be an uppercase letter, but got "{verdict}" - {type(verdict)}')
            return None
        if verdict_idx >= n_comparisons:
            print(
                f'Verdict must be in the range of the number of comparisons: {n_comparisons}, '
                f'but got "{verdict}" (idx = {verdict_idx})')
            return None

        return best_position == verdict_idx


def gen_stats_tables(verdicts: pd.DataFrame, output_dir: str, certainty: bool = False,
                     gen_stats_all=True, gen_stats_nc=True, gen_stats_nc_q=True, verbosity=1):
    verdict_model_name = verdicts['verdict_model_name'].iloc[0]

    if gen_stats_all:
        printv("-computing all_stats file...", v=verbosity)
        evaluator = VerdictEvaluator(verdicts=verdicts, certainty=certainty, verbosity=verbosity, output_dir=output_dir)
        stats_all = evaluator.generate_stats_views()
        evaluator.verdicts.to_csv(os.path.join(output_dir, 'verdicts_with_stats.csv'))
        df_stats = pd.DataFrame([stats_all])
        df_stats['verdict_model_name'] = verdict_model_name
        df_stats.to_csv(os.path.join(output_dir, 'stats_all.csv'))
        printc("--> done!", c='green')
    else:
        printv("-[skipping] all_stats file...", v=verbosity)

    if gen_stats_nc:
        printv('-computing stats by n_comparisons...', v=verbosity)
        stats_by_n_comp = []
        for n_comp, df in verdicts.groupby('n_comparisons'):
            printv(f'--: n_comp: {n_comp}', v=verbosity, v_min=2)
            evaluator = VerdictEvaluator(df, certainty=certainty, verbosity=0, output_dir=output_dir)
            stats = evaluator.generate_stats_views()
            for entry in stats:
                entry['n_comparisons'] = n_comp
            stats_by_n_comp.extend(stats)
        df_stats = pd.DataFrame(stats_by_n_comp)
        df_stats['verdict_model_name'] = verdict_model_name
        df_stats.to_csv(os.path.join(output_dir, 'stats_by_n_comp.csv'))
        printc("--> done!", c='green')
    else:
        printv("-[skipping] stats by n_comparisons...", v=verbosity)

    if gen_stats_nc_q:
        printv('-computing stats by n_comparisons and questions...', v=verbosity)
        stats_by_n_comp_questions = []
        question_model_name_present = False
        if 'question_model_name' in verdicts.columns:
            qmn = verdicts['question_model_name'].iloc[0]
            if isinstance(qmn, str) and len(qmn) > 5:
                question_model_name_present = True
        groups = ['n_comparisons', 'question']
        if question_model_name_present:
            groups.append('question_model_name')
        for g, df in verdicts.groupby(groups):
            question_model_name = None
            if question_model_name_present:
                n_comp, question, question_model_name = g
            else:
                n_comp, question = g
            printv(f'--: n_comp: {n_comp}, question', v=verbosity, v_min=2)
            evaluator = VerdictEvaluator(df, certainty=certainty, verbosity=0, output_dir=output_dir)
            stats = evaluator.generate_stats_views()
            for entry in stats:
                entry['n_comparisons'] = n_comp
                entry['question'] = question
                entry['question_model_name'] = question_model_name
            stats_by_n_comp_questions.extend(stats)
        df_stats = pd.DataFrame(stats_by_n_comp_questions)
        df_stats['verdict_model_name'] = verdict_model_name
        df_stats.to_csv(os.path.join(output_dir, 'stats_by_n_comp_questions.csv'))
        printc("--> done!", c='green')
    else:
        printv("-[skipping] stats by n_comparisons and questions...", v=verbosity)


def merge_extracted_verdicts(path: str, target_ext: str, add_star=True):
    folders = glob(f'{path}/*/runs/') if add_star else glob(f'{path}/runs/')
    for f in folders:
        # create new folder for extractions and statistics
        dest_f = f.replace('runs/', 'statistics/')
        os.makedirs(dest_f, exist_ok=True)

        sub = os.listdir(f)
        s = [os.path.join(f, s) for s in sub if target_ext in os.listdir(os.path.join(f, s))]

        dest_path = os.path.join(dest_f, target_ext)
        if len(s) > 1:
            print(f'merging {len(s)} files to: {dest_path}...')
            dfs = []
            for s_ in s:
                df = pd.read_csv(os.path.join(s_, target_ext))
                dfs.append(df)
            df = save_df_concat(dfs)
            df.to_csv(dest_path, index=False)

        else:
            src_path = os.path.join(s[0], target_ext)
            shutil.copy2(src_path, dest_path)


def main(base_folder: str, is_direct_folder: bool = False, override: bool = False, skip_yesno: bool = False,
         skip_certainty: bool = True, try_merge: bool = True, override_underlying: bool = False,
         gen_stats_all: bool = True, gen_stats_nc: bool = True, gen_stats_nc_q: bool = True, verbosity: int = 1):
    target_ext = 'verdicts_extracted.csv'
    target_added_ext = 'verdicts_with_stats.csv'
    stats_all_ext = 'stats_all.csv'
    stats_n_comp_ext = 'stats_by_n_comp.csv'
    stats_n_comp_q_ext = 'stats_by_n_comp_questions.csv'

    if try_merge:
        merge_extracted_verdicts(base_folder, target_ext=target_ext, add_star=not is_direct_folder)

    all_folders = find_folder_paths_with_target_ext(base_folder, target_ext, require_substring='/statistics',
                                                    verbosity=1)
    ts = []
    for i, folder in enumerate(all_folders):
        if (os.path.exists(os.path.join(folder, stats_all_ext))
                and os.path.exists(os.path.join(folder, stats_n_comp_ext))
                and os.path.exists(os.path.join(folder, stats_n_comp_q_ext))
                and not override):
            print(f'[skipping] {folder}/{stats_all_ext} already exists')
            continue

        certainty = 'certainty' in folder
        if certainty and skip_certainty:
            print(f'[skipping] certainty verdicts')
            continue
        if 'yesno' in folder and skip_yesno:
            print(f'[skipping] yes/no verdicts')
            continue

        t = time.time()
        tr = "" if len(ts) == 0 else f'{(np.mean(ts) * (len(all_folders) - (i + 1))) / 60.:.2f}min, '
        print(f'{tr}{"[override] " if override else ""}computing stats for {folder}...')
        fname_pre_processed = os.path.join(folder, target_added_ext)
        if os.path.exists(fname_pre_processed) and not override_underlying:
            verdicts = pd.read_csv(fname_pre_processed)
        else:
            verdicts = pd.read_csv(os.path.join(folder, target_ext))
        gen_stats_tables(verdicts, output_dir=folder, certainty=certainty,
                         gen_stats_all=gen_stats_all, gen_stats_nc=gen_stats_nc, gen_stats_nc_q=gen_stats_nc_q,
                         verbosity=verbosity)
        printc(f'-->stats computed and saved to {folder}', c='green')
        ts.append(time.time() - t)


if __name__ == '__main__':
    fire.Fire(main)
