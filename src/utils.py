import json
import pandas as pd
import yaml
import omegaconf
import re
import os
from typing import Union, List
from datetime import datetime as dt
import scipy.stats as sps
import numpy as np

odict = (dict, omegaconf.dictconfig.DictConfig)
olist = (list, omegaconf.listconfig.ListConfig)

model_order = [
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
]

model_mapping = {
    'claude-3-haiku-20240307': 'Claude Haiku',
    'claude-3-opus-20240229': 'Claude Opus',
    'claude-3-sonnet-20240229': 'Claude Sonnet',
    'command-r-plus': 'Command R+',
    'gemini-1.0-pro': 'Gemini 1.0 Pro',
    'google_gemma-7b-it': 'Gemma-7B',
    'gpt-3.5-turbo': 'GPT-3.5-turbo',
    'gpt-4-turbo': 'GPT-4-turbo',
    'meta-llama/Llama-3-70b-chat-hf': 'Llama 3 70B',
    'meta-llama/Llama-3-8b-chat-hf': 'Llama 3 8B',
    'mistralai/Mixtral-8x22B-Instruct-v0.1': 'Mixtral-8x22B',
    'mistralai/Mixtral-8x7B-Instruct-v0.1': 'Mixtral-8x7B'
}

plot_colors = {
    0: '#1B998B',  # Teal
    1: '#E84855',  # Red
    2: '#800080',  # Purple
    3: '#FAA613',  # Orange
    4: '#2D3047',  # Dark Blue
    5: '#3F88C5',  # Blue
    6: '#F4A261',  # Peach
    7: '#8D8741',  # Olive
    8: '#A23E48',  # Dark Red
    9: '#F45B69'   # Pink
}


def read_txt(path):
    with open(path, 'r') as f:
        return f.read()


def get_str_timestamp():
    return dt.now().strftime("%Y%m%d_%H%M%S_%f")


def extract_dictionary(x):
    if isinstance(x, str):
        regex = r"{.*?}"
        match = re.search(regex, x, re.MULTILINE | re.DOTALL)
        if match:
            try:
                json_str = match.group()
                json_str = json_str.replace("'", '"')
                dict_ = json.loads(json_str)
                return dict_
            except Exception as e:
                print(f"unable to extract dictionary - {e}")
                return None

        else:
            return None
    else:
        return None


def save_yaml(content, path):
    with open(path, 'w') as f:
        yaml.dump(content, f, default_flow_style=False)


def load_yaml(path):
    with open(path, 'r') as f:
        content = yaml.safe_load(f)
    return content


def printv(msg, v=0, v_min=0, c=None, debug=False):
    # convenience print function
    if debug:
        c = 'yellow' if c is None else c
        v, v_min = 1, 0
        printc('\n\n>>>>>>>>>>>>>>>>>>>>>>START DEBUG\n\n', c='yellow')
    if (v > v_min) or debug:
        if c is not None:
            printc(msg, c=c)
        else:
            print(msg)
    if debug:
        printc('\n\nEND DEBUG<<<<<<<<<<<<<<<<<<<<<<<<\n\n', c='yellow')


def printc(x, c='r'):
    m1 = {'r': 'red', 'g': 'green', 'y': 'yellow', 'w': 'white',
          'b': 'blue', 'p': 'pink', 't': 'teal', 'gr': 'gray'}
    m2 = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'pink': '\033[95m',
        'teal': '\033[96m',
        'white': '\033[97m',
        'gray': '\033[90m'
    }
    reset_color = '\033[0m'
    print(f'{m2.get(m1.get(c, c), c)}{x}{reset_color}')


def pretty_print_model_names(m):
    model_map = {'command-r-plus': 'command-r-plus',
                 'gemini-1.0-pro': 'gemini-1.0-pro',
                 'claude-3-sonnet-20240229': 'claude-3-sonnet',
                 'claude-3-haiku-20240307': 'claude-3-haiku',
                 'claude-3-opus-20240229': 'claude-3-opus',
                 'gpt-3.5-turbo': 'gpt-3.5-turbo',
                 'meta-llama/Llama-3-8b-chat-hf': 'llama-3-8b',
                 'gpt-4-turbo': 'gpt-4-turbo',
                 'meta-llama/Llama-3-70b-chat-hf': 'llama-3-70b',
                 'mistralai/Mixtral-8x22B-Instruct-v0.1': 'mixtral-8x22b',
                 'mistralai/Mixtral-8x7B-Instruct-v0.2': 'mixtral-8x7b',
                 'google/gemma-7b-it': 'gemma-7b-it',
                 }
    return model_map.get(m, m)


def paper_plot_model_mapping(m):
    paper_model_mapping = {
        'claude-3-haiku-20240307': 'Claude 3 Haiku',
        'claude-3-opus-20240229': 'Claude 3 Opus',
        'claude-3-sonnet-20240229': 'Claude 3 Sonnet',
        'command-r-plus': 'Command R+',
        'gemini-1.0-pro': 'Gemini 1.0 Pro',
        'google/gemma-7b-it': 'Gemma-7B',
        'google_gemma-7b-it': 'Gemma-7B',
        'gpt-3.5-turbo': 'GPT-3.5-turbo',
        'gpt-4-turbo': 'GPT-4-turbo',
        'meta-llama_Llama-3-70b-chat-hf': 'Llama 3 70B',
        'meta-llama/Llama-3-70b-chat-hf': 'Llama 3 70B',
        'meta-llama_Llama-3-8b-chat-hf': 'Llama 3 8B',
        'meta-llama/Llama-3-8b-chat-hf': 'Llama 3 8B',
        'mistralai_Mixtral-8x22B-Instruct-v0.1': 'Mixtral-8x22B',
        'mistralai/Mixtral-8x22B-Instruct-v0.1': 'Mixtral-8x22B',
        'mistralai_Mixtral-8x7B-Instruct-v0.1': 'Mixtral-8x7B',
        'mistralai/Mixtral-8x7B-Instruct-v0.1': 'Mixtral-8x7B'
    }

    return paper_model_mapping.get(m, m)


def converts_to_valid_float(x) -> bool:
    """
    Check if a string can be converted to a float or is already a float
    :param x: variable to check
    :return: bool
    """
    valid = True
    if x is None:
        valid = False
    if isinstance(x, float) and np.isnan(x):
        valid = False
    if isinstance(x, str) and not x.replace('.0', '').replace('.', '').isnumeric():
        valid = False

    try:
        float(x)
    except (ValueError, TypeError) as e:
        valid = False

    return valid


def get_api_key(fname='secrets.json', provider='openai', key='dlab_key'):
    try:
        with open(fname) as f:
            keys = json.load(f)[provider]
            if key is not None:
                api_key = keys[key]
            else:
                api_key = list(keys.values())[0]
    except Exception as e:
        print(f'error: unable to load {provider} api key {key} from file {fname} - {e}')
        return None

    return api_key


def read_json(path_name: str):
    with open(path_name, "r") as f:
        json_file = json.load(f)
    return json_file


def format_dictionary(dictionary, indent=0):
    result = ""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            result += f"{' ' * indent}{key}:\n{format_dictionary(value, indent + 4)}"
        else:
            result += f"{' ' * indent}{key}: {value}\n"
    return result


def dictionary_to_string(dictionary):
    return format_dictionary(dictionary)


def find_folder_paths_with_target_ext(path, target_ext=None, verbosity=0, require_substring=None):

    folders = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == target_ext:
                folders.append(root)

    if require_substring is not None:
        folders = [f for f in folders if require_substring in f]

    printv(f'found {len(folders)} folders with {target_ext} files\n', v=verbosity)

    return folders


def save_df_concat(dfs: List[pd.DataFrame], drop_index=True, verbosity: int = 0) -> pd.DataFrame:
    cols = set()

    if len(dfs) == 0:
        printv(f'[warning] list is empty', v=verbosity)
        return pd.DataFrame()

    if len(dfs) == 1:
        printv(f'[warning] list has only one element', v=verbosity)
        return dfs[0]

    for x in dfs:
        cols.update(x.columns)

    dfs_ = []
    for x in dfs:
        missing_cols = [c for c in cols if c not in x.columns]
        if len(missing_cols) > 0:
            # Create a DataFrame with the missing columns initialized to None
            missing_df = pd.DataFrame({mc: [None] * len(x) for mc in missing_cols})
            # Concatenate the original DataFrame with the missing columns DataFrame
            x = pd.concat([x, missing_df], axis=1)
        dfs_.append(x.copy())  # avoid fragmentation

    df = pd.concat(dfs_).reset_index(drop=drop_index)

    return df


def unpack_nested_yaml(x):
    def _update_yaml_path(fpath, project_folder=re.compile(r'(^.*)YCM')):
        """Ensures local paths are updated for experiments that were run on different machines"""
        from_path = project_folder.search(fpath)
        to_path = project_folder.search(os.path.abspath(os.getcwd()))
        if to_path is None or from_path is None:
            return fpath

        return fpath.replace(from_path[1], to_path[1])

    def _unpack_list(source: Union[None, dict, list], package: list, package_idx: int = None, package_key: str = None):
        force_update = 0
        for i, v in enumerate(package.copy()):
            if isinstance(v, odict):
                _unpack_dict(source=package, package=v, package_idx=i)
            elif isinstance(v, olist):
                _unpack_list(source=package, package=v, package_idx=i)
            elif isinstance(v, str):
                force_update += _unpack_str(source=package, package=v, package_idx=i)

        if isinstance(source, olist) and package_idx is not None:
            source[package_idx] = package
        if isinstance(source, odict) and package_key is not None:
            source[package_key] = package

        if force_update > 0:
            _unpack_list(source=source, package=package, package_idx=package_idx, package_key=package_key)

    def _unpack_dict(source: Union[None, dict, list], package: dict, package_idx: int = None, package_key: str = None):
        force_update = 0
        for k, v in package.copy().items():
            if isinstance(v, odict):
                _unpack_dict(source=package, package=v, package_key=k)
            elif isinstance(v, olist):
                _unpack_list(source=package, package=v, package_key=k)
            elif isinstance(v, str):
                force_update += _unpack_str(source=package, package=v, package_key=k)

        if isinstance(source, olist) and package_idx is not None:
            source[package_idx] = package
        if isinstance(source, odict) and package_key is not None:
            source[package_key] = package

        if force_update > 0:
            _unpack_dict(source=source, package=package, package_key=package_key, package_idx=package_idx)

    def _unpack_str(source: Union[None, dict, list], package: str, package_idx: int = None, package_key: str = None
                    ) -> int:
        if not package.endswith('.yaml'):
            return 0
        # update to local machine path
        package = _update_yaml_path(package)

        with open(package, 'r') as file:
            yaml_data = yaml.safe_load(file)
        # do not override keys that already exist
        yaml_data = {k: v for k, v in yaml_data.items() if (
                (isinstance(source, odict) and k not in source.keys()) or
                (isinstance(source, odict) and str(source.get(k)).endswith('.yaml')) or
                (not isinstance(source, odict)))
                     }
        if isinstance(source, odict):
            source.update(yaml_data)
            if isinstance(source[package_key], str) and source[package_key].endswith('.yaml') \
                    and package_key not in yaml_data.keys():
                del source[package_key]
        elif isinstance(source, olist):
            source[package_idx] = yaml_data

        # check if any of the values are still yaml references
        if any([v.endswith('.yaml') for v in yaml_data.values() if isinstance(v, str)]):
            return 1

        return 0

    if isinstance(x, odict):
        _unpack_dict(source=None, package=x)
    elif isinstance(x, olist):
        _unpack_list(source=None, package=x)
    elif isinstance(x, str):
        recursion = _unpack_str(source=None, package=x)
        if recursion > 0:
            unpack_nested_yaml(x)

    return x


def get_inference_root_overrides(cfg_, inference_root_path='src/configs/inference_root.yaml'):
    """Parse inference_root level overrides, e.g., verbosity, max_rounds, etc."""
    root = yaml.safe_load(open(inference_root_path))
    root_keys = root.keys()
    overrides = {}
    for k, v in cfg_.items():
        if k in root_keys and v != root[k]:
            overrides[k] = v

    return overrides


def _update_model_constructor_hydra(model_provider):
    model_target = "models."

    if model_provider in ['azure', 'openai']:
        model_target += "OpenAIModel"
    elif model_provider == 'anthropic':
        model_target += "AnthropicModel"
    elif model_provider == 'cohere':
        model_target += "CohereModel"
    elif model_provider == 'google' or model_provider == 'google_gemini':
        model_target += "GoogleModel"
    elif model_provider == 'llama':
        model_target += "HuggingFaceModel"
    elif model_provider == 'together_ai':
        model_target += 'TogetherAIModel'
    else:
        raise NotImplementedError('feel free to extend to with custom models')

    return model_target


def update_model_constructor_hydra(cfg_exp):
    for a in cfg_exp['agents']:
        a['model']['_target_'] = _update_model_constructor_hydra(a['model']['model_provider'])


class LatentModel:
    def __init__(self, n_comparisons, n_samples):
        self.n_comparisons = n_comparisons
        self.n_samples = n_samples

        my_latents = sps.norm.rvs(size=n_samples)
        other_latents = sps.norm.rvs(size=(n_comparisons - 1, n_samples))
        other_latents_max = np.max(other_latents, axis=0)

        self.shift = other_latents_max - my_latents
        self.shift = np.sort(self.shift)

    def get_shift(self, accuracy):
        if accuracy == 1.0:
            return float('inf')
        elif accuracy == 0:
            return float('-inf')
        return self.shift[int(accuracy * self.n_samples)]

    def get_accuracy(self, shift):
        return np.searchsorted(self.shift, shift) / self.n_samples
