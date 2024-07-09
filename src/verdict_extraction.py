from attr import define, field
import pandas as pd
import yaml
import re
from typing import Tuple, Union
from tqdm import tqdm
import numpy as np
import fire
import os
import sys
sys.path.append('src/')

from models.model_utils import HumanMessage, BaseMessage, ChatModel
from models.openai_model import OpenAIModel
from models.anthropic_model import AnthropicModel
from models.cohere_model import CohereModel
from models.togetherai_model import TogetherAIModel
from models.google_model import GoogleModel
from utils import extract_dictionary, printc, find_folder_paths_with_target_ext, converts_to_valid_float


prompt_mapping = {
    'multi': {
        'no_conf': 'data/prompts/answer_extraction_multi.yaml',
        'conf': 'data/prompts/answer_extraction_multi_conf.yaml'
    },
    'single': {
        'no_conf': 'data/prompts/answer_extraction_single.yaml',
        'conf': 'data/prompts/answer_extraction_multi_conf.yaml'
    }
}

alphabet = {letter: i for i, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}


@define
class ExtractModel:
    verdicts_path: str
    model_name: str = field(default='gpt-3.5-turbo')
    model_provider: str = field(default='azure')
    model: ChatModel = field(default=None, init=False)
    verdicts: pd.DataFrame = field(default=None)
    certainty: bool = field(default=True)
    debug: bool = field(default=False)

    def __attrs_post_init__(self):
        self.verdicts = pd.read_csv(self.verdicts_path)

        if self.model_provider in ['azure', 'openai']:
            model_f = OpenAIModel
        elif self.model_provider == 'anthropic':
            model_f = AnthropicModel
        elif self.model_provider == 'cohere':
            model_f = CohereModel
        elif self.model_provider == 'google':
            model_f = GoogleModel
        elif self.model_provider == 'together_ai':
            model_f = TogetherAIModel
        else:
            raise NotImplementedError('feel free to extend to with custom models')

        self.model = model_f(model_name=self.model_name, model_provider=self.model_provider)

    def run(self):
        verdicts = []
        confs = []
        for i, row in tqdm(self.verdicts.iterrows()):
            confidence = row['instruction_certainty']
            n_comparisons = row['n_comparisons']
            verdict, confidence = do_verdict_extraction(row['verdict'],
                                                        model=self.model,
                                                        n_comparisons=n_comparisons,
                                                        confidence=confidence,
                                                        )
            if self.debug:
                printc(f'original verdict: {row["verdict"]}\n--[exv] {verdict}\n--[confi] {confidence}', c='yellow')

            verdicts.append(verdict)
            confs.append(confidence)

        self.verdicts['verdict_extract'] = verdicts
        self.verdicts['verdict_extract_confidence'] = confs
        self.verdicts['verdict_extract_model_name'] = self.model_name

        return self.verdicts


def logprob_to_dict(x, exp=True):
    if isinstance(x, dict):
        return x

    try:
        x_ = eval(x)
    except (TypeError, ValueError) as e:
        d = {}
    else:
        d = {}
        for lp in x_:
            token = lp['token']
            token_lp = lp['logprob']
            res = d.get(token.strip())
            # only add if not already in dict
            if res is None:
                d[token.strip()] = np.exp(token_lp) if exp else token_lp
    return d


def lookup_logprob(key, logprobs, exp=False, round_lp=None):
    if key is None:
        return None
    logprobs = logprob_to_dict(logprobs, exp=exp)
    key_ = str(key).strip().replace('.0', '').replace('.', '')
    key_lp = logprobs.get(key_)
    if key_lp is not None and round_lp is not None:
        key_lp = round(key_lp, round_lp)
    return key_lp


def ncomp_1_confidence(df, inplace=False):
    df_ = df if inplace else df.copy()

    k1 = 'verdict_extract'
    k2 = 'verdict_extract_confidence'
    cols = [k1, k2]
    ves, vecs = [], []
    for ve, vec in df_[df_['n_comparisons'] == 1][cols].values:
        if converts_to_valid_float(ve) and not converts_to_valid_float(vec):
            vec = float(ve)
            ve = None
        ves.append(ve)
        vecs.append(vec)
    df_.loc[df_['n_comparisons'] == 1, k1] = ves
    df_.loc[df_['n_comparisons'] == 1, k2] = vecs
    if not inplace:
        return df_


def extract_logprobs(df, inplace=False):
    df_ = df if inplace else df.copy()

    if 'verdict_logprobs' not in df_.columns:
        df_['verdict_logprobs'] = None
        df_['verdict_extract_logprobs'] = None
        df_['verdict_extract_confidence_logprobs'] = None
    else:
        vs_lps, v_conf_lps = [], []
        for (v, vc, vlp) in df_[['verdict_extract', 'verdict_extract_confidence', 'verdict_logprobs']].values:
            v_lp = lookup_logprob(v, vlp)
            vc_lp = lookup_logprob(vc, vlp)

            vs_lps.append(v_lp)
            v_conf_lps.append(vc_lp)

        df_['verdict_extract_logprobs'] = vs_lps
        df_['verdict_extract_confidence_logprobs'] = v_conf_lps

    if not inplace:
        return df_


def do_simple_verdict_extraction(verdict: str, n_comparisons: int, confidence: bool):
    if verdict == '```' or not isinstance(verdict, str) or len(verdict) == 0:
        return True, ('None', None)
    verdict = verdict.strip().strip('.')
    if not confidence:
        if n_comparisons == 1:
            # yes/no, no confidence
            if verdict.lower() in ['yes', 'no']:
                return True, (verdict.lower(), None)
        else:
            # multiple, no confidence
            if verdict.upper() in alphabet:
                return True, (verdict.upper(), None)
    else:
        if n_comparisons == 1:
            # yes/no, confidence
            if verdict in '12345':
                return True, (verdict, None)
        else:
            # multiple, confidence

            if len(verdict) <= 1:
                return True, ('None', None)

            rgx = r'^([A-Z])([-\W ]){0,5}([1-5])$'
            rgx = re.compile(rgx)
            res = rgx.search(verdict)
            if res is not None:
                label = res.group(1)
                score = res.group(3)
                return True, (label, score)

    return False, None


def do_verdict_extraction(verdict: str, n_comparisons: int, confidence: bool, model: ChatModel
                          ) -> Tuple[Union[str, None], Union[str, None]]:

    can_simple_extract, simple_result = do_simple_verdict_extraction(verdict, n_comparisons, confidence)
    if can_simple_extract:
        return simple_result

    conf_str = 'conf' if confidence else 'no_conf'
    n_comp_str = 'multi' if n_comparisons > 1 else 'single'

    fname = prompt_mapping[n_comp_str][conf_str]
    prompts = yaml.safe_load(open(fname))['prompt']
    prompts = [BaseMessage(**pm) for pm in prompts]
    # format task
    prompts.append(HumanMessage(content=verdict))

    try:
        output = model(prompts)
    except Exception as e:
        output = None
        print(f'error: failed to extract verdict from message - {e}')

    if output is None:
        return None, None

    if confidence:
        output = extract_dictionary(output)
        ex_verdict, confidence = None, None
        if isinstance(output, dict):
            ex_verdict, confidence = output.get('verdict'), output.get('confidence')
        if ex_verdict is None and confidence is None:
            return None, None

        if isinstance(ex_verdict, str):
            ex_verdict = ex_verdict.strip()
        if isinstance(confidence, str):
            confidence = confidence.strip()

        return ex_verdict, confidence

    if isinstance(output, str):
        output = output.strip()

    return output, None


def main(base_folder: str, override=False):
    """
    Convert verdicts.csv files to verdicts_extracted.csv files. Adds the following columns:
    - verdict_extract: extracted verdict
    - verdict_extract_confidence: confidence score
    - verdict_extract_model_name: model name used for extraction

    Example usage:
        python src/verdict_extraction.py --base_folder="data/verdicts_to_process_random"
    :param base_folder: root folder path containing verdicts.csv files
    :param override: weather to override existing extracted files
    """
    target_ext = 'verdicts.csv'
    extracted_ext = 'verdicts_extracted.csv'

    folder_paths = find_folder_paths_with_target_ext(base_folder, target_ext=target_ext, verbosity=1)
    for folder in folder_paths:
        if os.path.exists(os.path.join(folder, extracted_ext)) and not override:
            print(f'[skipping] {folder}/{extracted_ext} already exists')
            continue
        print(f'{"[override] " if override else ""}extracting from {folder}...')
        model = ExtractModel(os.path.join(folder, target_ext))
        df_extracted = model.run()
        ncomp_1_confidence(df_extracted, inplace=True)
        if 'verdict_logprobs' not in df_extracted.columns:
            df_extracted['verdict_logprobs'] = None
        else:
            extract_logprobs(df_extracted, inplace=True)

        df_extracted.to_csv(os.path.join(folder, extracted_ext), index=False)
        printc(f'-->extracted to {os.path.join(folder, extracted_ext)}', c='green')


if __name__ == '__main__':
    fire.Fire(main)
