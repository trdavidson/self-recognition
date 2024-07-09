from attr import attrs, field
from typing import List, Dict, Any, Union
import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from utils import find_folder_paths_with_target_ext, save_df_concat, printv, pretty_print_model_names, plot_colors


@attrs
class AnalysisTool:
    """
    AnalysisTool class for analyzing the results of the Verdict Models.
    """
    base_folders: List[str] = field(factory=list)
    skip_yesno: bool = field(default=False)
    skip_certainty: bool = field(default=False)
    stats_yesno: Dict[str, Any] = field(factory=dict, init=False)
    stats_certainty: Dict[str, Any] = field(factory=dict, init=False)
    model_params: dict = field(default=None, init=False)
    model_type: dict = field(default=None, init=False)

    def __attrs_post_init__(self):
        # command: https://huggingface.co/CohereForAI/c4ai-command-r-plus
        # claude speculation: https://lifearchitect.substack.com/p/the-memo-special-edition-claude-3
        # gpt-3.5-turbo leak: https://arxiv.org/abs/2310.17680
        # gpt-4/turbo rumor: https://en.wikipedia.org/wiki/GPT-4
        # gemini 1.0 rumors: https://www.merca20.com/gemini-googles-new-artificial-intelligence-explained-in-5-key-facts/
        self.model_params = {
            'claude-3-haiku-20240307': 20,
            'claude-3-sonnet-20240229': 70,
            'claude-3-opus-20240229': 2000,
            'meta-llama/Llama-3-70b-chat-hf': 70,
            'mistralai/Mixtral-8x22B-Instruct-v0.1': 176,
            'meta-llama/Llama-3-8b-chat-hf': 8,
            'command-r-plus': 104,
            'gemini-1.0-pro': 1600,
            'google/gemma-7b-it': 7,
            'gpt-3.5-turbo': 20,
            'gpt-4-turbo': 10000
        }

        self.model_type = {
            'claude-3-haiku-20240307': 'small',
            'claude-3-sonnet-20240229': 'large',
            'claude-3-opus-20240229': 'large',
            'meta-llama/Llama-3-70b-chat-hf': 'large',
            'mistralai/Mixtral-8x22B-Instruct-v0.1': 'large',
            'meta-llama/Llama-3-8b-chat-hf': 'small',
            'command-r-plus': 'large',
            'gemini-1.0-pro': 'large',
            'google/gemma-7b-it': 'small',
            'gpt-3.5-turbo': 'small',
            'gpt-4-turbo': 'large'
        }

        for folder in self.base_folders:
            if not os.path.exists(folder):
                raise ValueError(f'{folder} does not exist.')

        self.merge_files()

    @staticmethod
    def _column_reorder(df):
        start = ['verdict_model_name', 'dynamic', 'name', 'n_comparisons']
        start = [c for c in start if c in df.columns]
        end = [c for c in df.columns if c not in start]
        df = df[start + end]
        return df

    @staticmethod
    def filter_df(df_, model_name: Union[None, str, list] = None, dynamic: Union[None, bool] = None,
                  n_comparisons: Union[None, list, int] = 2,
                  name_cats: Union[None, list] = ['first', 'not_first'],
                  swap_type: Union[None, str, list] = None, question_type: Union[None, str, list] = None
                  ) -> pd.DataFrame:

        df_f = df_.copy()

        if model_name is not None:
            if isinstance(model_name, list):
                df_f = df_f[df_f['verdict_model_name'].isin(model_name)].copy()
            else:
                df_f = df_f[df_f['verdict_model_name'] == model_name].copy()
        else:
            df_f = df_f[df_f['verdict_model_name'].isin(df_f['verdict_model_name'].unique())].copy()

        if name_cats is not None:
            if isinstance(name_cats, str):
                name_cats = [name_cats]
            missing_types = set(name_cats) - set(df_f['name'].unique())
            if len(missing_types) > 0:
                print(f'[warning] missing name categories!: {missing_types}')
            df_f = df_f[df_f['name'].isin(name_cats)].copy()

        if question_type is not None and isinstance(question_type, (list, str)):
            if isinstance(question_type, str):
                question_type = [question_type]

            df_f = df_f[df_f['question_type'].isin(question_type)].copy()
            missing_types = set(question_type) - set(df_f['question_type'].unique())
            if len(missing_types) > 0:
                print(f'[warning] missing question types!: {missing_types}')

        if n_comparisons is not None:
            if isinstance(n_comparisons, list):
                df_f = df_f[df_f['n_comparisons'].isin(n_comparisons)].copy()
            else:
                df_f = df_f[df_f['n_comparisons'] == n_comparisons].copy()

        if dynamic is not None:
            df_f = df_f[df_f['dynamic'] == dynamic].copy()

        if swap_type is not None:
            if isinstance(swap_type, list):
                df_f = df_f[df_f['q_swap_type'].isin(swap_type)].copy()
            else:
                df_f = df_f[df_f['q_swap_type'] == swap_type].copy()

        return df_f

    @staticmethod
    def filter_out_own_questions(df, question_col: str = 'question_model_name', verdict_col: str = 'verdict_model_name',
                                 question_type: str = 'random', only_keep_own: bool = False):

        if isinstance(question_type, str):
            question_type = [question_type]

        df_ = df[
            ((df['question_type'].isin(question_type)) & ((df[verdict_col] == df[question_col]) == only_keep_own)) | (
                ~df['question_type'].isin(question_type))].copy()
        return df_

    def get_model_names(self, use_yesno=True, question_type=None) -> List[str]:
        stats = self.stats_yesno if use_yesno else self.stats_certainty
        y = stats['stats_by_n_comp_questions.csv']
        y = self.filter_df(y, model_name=None, dynamic=None, n_comparisons=None, name_cats=None,
                           swap_type=None, question_type=question_type)
        return y['verdict_model_name'].unique().tolist()

    def _annotate(self, df: pd.DataFrame, f: str):

        if len(df) < 1:
            print(f)

        random = 'random' in f
        dynamic = ('_ii' in f or '_iu' in f or 'dynamic' in f) and not random
        topk = 'topk' in f
        hide_q = 'hide_q' in f
        swap_same = 'swap_same' in f
        swap_diff = 'swap_diff' in f
        swap_answer = 'swap_answer' in f
        titan = 'titans' in f
        titan_swap = 'titans_swap' in f
        titan_hidden = 'titans_hide_q' in f
        titan_qr100 = 'titans_qr100' in f

        swap_response_flag = 'swap_answer' in f
        hide_question_flag = 'hide_q' in f or swap_response_flag
        swap_question_flag = 'swap_same' in f or 'swap_diff' in f
        response_restriction = 100 if 'qr100' in f else (250 if 'qr250' in f else -1)

        # print(dynamic, f)

        q_type = 'static'
        if titan_hidden:
            q_type = 'titan_hidden'
        elif swap_same:
            q_type = 'swap_same'
        elif swap_diff:
            q_type = 'swap_diff'
        elif topk:
            q_type = 'topk'
        elif titan_qr100:
            q_type = 'titan_qr100'
        elif titan_swap:
            q_type = 'titan_swap'
        elif hide_q:
            q_type = 'hidden'
        elif titan:
            q_type = 'titan'
        elif swap_answer:
            q_type = 'swap_answer'
        elif random:
            q_type = 'random'
        elif dynamic:
            q_type = 'dynamic'
        else:
            pass
        df['question_type'] = q_type

        df['q_swap_type'] = None if 'swap' not in f else ('swap_same' if 'same' in f else 'swap_diff')

        df['dynamic'] = True if dynamic else False
        df['hidden'] = hide_question_flag
        df['swap_response'] = swap_response_flag
        df['response_restriction'] = response_restriction
        df['swap_question'] = swap_question_flag

        vdm = df['verdict_model_name'].iloc[0]
        df['model_size'] = self.model_type.get(vdm)
        df['model_params'] = self.model_params.get(vdm)

    def merge_files(self):
        stats_all_ext = 'stats_all.csv'
        stats_n_comp_ext = 'stats_by_n_comp.csv'
        stats_n_comp_q_ext = 'stats_by_n_comp_questions.csv'
        stat_files = [stats_all_ext, stats_n_comp_ext, stats_n_comp_q_ext]

        for sf in stat_files:
            folders = []
            for bf in self.base_folders:
                folders_i = find_folder_paths_with_target_ext(bf, sf, require_substring='statistics')
                folders = folders + folders_i

            dfs_yn = []
            dfs_c = []

            for f in [os.path.join(f, sf) for f in folders]:
                df = pd.read_csv(f)
                if len(df) < 1:
                    print(f'[warning!] empty file: {f}')
                    continue

                self._annotate(f=f, df=df)

                if 'certainty' not in f and not self.skip_yesno:
                    dfs_yn.append(df)

                if 'certainty' in f and not self.skip_certainty:
                    dfs_c.append(df)

            if dfs_yn is not None and len(dfs_yn) > 0:
                dfs_yn = save_df_concat(dfs_yn).reset_index(drop=True)
                dfs_yn = self._column_reorder(dfs_yn)
                self.stats_yesno[sf] = dfs_yn

            if dfs_c is not None and len(dfs_c) > 0:
                dfs_c = save_df_concat(dfs_c).reset_index(drop=True)
                dfs_c = self._column_reorder(dfs_c)
                self.stats_certainty[sf] = dfs_c

    def print_merge_summary(self):
        if self.stats_yesno is not None and any(self.stats_yesno):
            print('\nYes/No views:')
            for k, v in self.stats_yesno.items():
                print(f'  {k}:'.ljust(35) + f'{v.shape}')

            if self.stats_yesno.get('stats_by_n_comp.csv') is not None:
                print('\navailable question_types:')
                for qt in self.stats_yesno['stats_by_n_comp.csv']['question_type'].unique():
                    print(f'  > {qt}')

        if self.stats_certainty is not None and any(self.stats_certainty):
            print('\nCertainty views:')
            for k, v in self.stats_certainty.items():
                print(f'  {k}:'.ljust(35) + f'{v.shape}')

            if self.stats_certainty.get('stats_by_n_comp.csv') is not None:
                print('\navailable question_types:')
                for qt in self.stats_certainty['stats_by_n_comp.csv']['question_type'].unique():
                    print(f'  > {qt}')

    def get_stats(self, use_yesno: bool = True, stats_type: str = 'by_question',
                  question_type: Union[str, list] = None):

        stats_key = {'all': 'stats_all.csv',
                     'by_ncomp': 'stats_by_n_comp.csv',
                     'by_question': 'stats_by_n_comp_questions.csv'
                     }

        assert stats_type in stats_key.keys(), f'Invalid stats_type: {stats_type} - need one of: {stats_key.keys()}'

        stats = self.stats_yesno if use_yesno else self.stats_certainty
        y = stats[stats_key[stats_type]]
        y = self.filter_df(y, model_name=None, dynamic=None, n_comparisons=None, name_cats=None,
                           swap_type=None, question_type=question_type)
        return y

    def plot_histograms(self, df: pd.DataFrame = None, use_yesno: bool = True, model_names: list = None,
                        dynamic: bool = None, n_comparisons: int = 2,
                        name_cats: list = [['first_not_name_drop', 'not_first_not_name_drop']],
                        plot_labels: list = None,
                        swap_type: str = None, question_type: Union[str, list] = None, title: str = '-',
                        plot_key: str = 'correct_mean', save: bool = False):

        if df is None:
            y = self.get_stats(use_yesno=use_yesno)
        else:
            y = df.copy()

        avg = 1 / n_comparisons
        datasets = []
        if model_names is None:
            model_names = y['verdict_model_name'].unique()

        if isinstance(name_cats, str) or name_cats is None:
            name_cats = [[name_cats]]
        elif isinstance(name_cats, list) and isinstance(name_cats[0], str):
            name_cats = [name_cats]
        else:
            pass

        for m in model_names:
            slices = []
            for k, nc in enumerate(name_cats):
                data = {'data': self.filter_df(y, model_name=m, dynamic=dynamic,swap_type=swap_type,
                                               question_type=question_type,name_cats=nc, n_comparisons=n_comparisons
                                               ).groupby('question')[
                    [plot_key]].mean().reset_index()[plot_key],
                        'model_name': m,
                        'label': str(nc) if plot_labels is None or len(plot_labels) != len(name_cats)
                        else plot_labels[k]
                        }
                slices.append(data)
            datasets.append(slices)

        colors = {
            0: '#4575b4',
            1: '#fc8d59',
            2: '#fee090'
        }

        # Number of bins and bin edges
        bins = np.linspace(0, 1, 11)

        # Create a 5x3 grid for the subplots
        fig, axs = plt.subplots(3, 5, figsize=(15, 8))

        # Flatten the 5x3 array of axes for easy iteration
        axs = axs.flatten()

        handles_list, labels_list = [], []
        # Plot histograms
        for i, slices in enumerate(datasets):
            data_to_stack = []
            means = []
            cs = []
            ls = []
            max_height = 30
            model_name = '-'
            for j, sl in enumerate(slices):
                data, model_name, l = sl['data'], sl['model_name'], sl['label']
                ls.append(l)
                data_f = np.round(np.array([d for d in data if not np.isnan(d)]), 3)
                data_histo, _ = np.histogram(data_f)
                max_height = max(20, max(data_histo)) * 2.5
                data_to_stack.append(data_f)
                model_mean = -1 if len(data_f) < 1 else data_f.mean()
                means.append(model_mean)
                cs.append(colors.get(j))

            axs[i].hist(data_to_stack, stacked=True, bins=bins, color=cs, edgecolor='white', label=ls)
            axs[i].set_title(f'{pretty_print_model_names(model_name)}')
            axs[i].set_xlim(0, 1)
            axs[i].set_ylim(0, max_height)
            for ii, (c, mean) in enumerate(zip(cs, means)):
                if mean > 0:
                    axs[i].axvline(mean, color=c, linestyle='dashed', linewidth=1, label=f'mean({ls[ii]})', alpha=0.8)
            axs[i].axvline(avg, color='black', linestyle='dashed', linewidth=1, label='random', alpha=0.8)

            handles, labels = axs[i].get_legend_handles_labels()
            handles = [h for (h, l_i) in zip(handles, labels) if l_i not in labels_list]
            labels = [l_i for l_i in labels if l_i not in labels_list]
            handles_list.extend(handles)
            labels_list.extend(labels)

        K = len(datasets)
        # Remove empty subplots
        for j in range(K, len(axs)):
            fig.delaxes(axs[j])

        # Adjust layout
        plt.suptitle(title)

        # order the legend
        zipped = [(l, h) for (l, h) in zip(labels_list, handles_list) if 'mean' not in l and l != 'random']
        mean_labels = [(f'mean({l})', handles_list[labels_list.index(f'mean({l})')]) for (l, h) in zipped
                       if f'mean({l})' in labels_list]
        random_handle = handles_list[labels_list.index('random')]
        zipped = zipped + mean_labels + [('random', random_handle)]
        # Create a single legend for all subplots
        unique_labels = dict(zipped)  # Remove duplicates while preserving order
        fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.25))

        # Adjust layout
        # plt.tight_layout()
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        if save:
            plt.savefig('model_question_histograms.pdf', bbox_inches='tight', dpi=300, format='pdf')
        plt.show()

    def plot_effects(self, df=None, models=None, use_yesno=True, question_types=['static', 'dynamic'], n_comparisons=2,
                     name_cats=['first', 'not_first'], swap_types=None, labels=None, colors=None, markers=None,
                     title='-', add_model_size=True, plot_key='correct_mean'):

        if models is None:
            models = self.get_model_names()

        if df is None:
            y = self.get_stats(use_yesno=use_yesno)
        else:
            y = df.copy()
        avg = 1 / n_comparisons

        stats = {}
        for qt in question_types:
            x = self.filter_df(y, model_name=models, dynamic=None, question_type=qt,
                               n_comparisons=n_comparisons, swap_type=swap_types, name_cats=name_cats)
            res1 = x.groupby('verdict_model_name')[plot_key].mean().to_dict()
            res2 = x.groupby('verdict_model_name')[plot_key].std().to_dict()
            res3 = x.groupby('verdict_model_name')[plot_key].count().to_dict()
            for (k1, v1), (k2, v2), (k3, v3) in zip(res1.items(), res2.items(), res3.items()):
                arr = stats.get(k1)
                s = (round(v1, 3), round(v2 / np.sqrt(v3), 3))
                if arr is None:
                    stats[k1] = [s]
                else:
                    stats[k1] = arr + [s]

        if colors is None:
            colors = plot_colors

        if labels is None:
            labels = {
                0: 'static.',
                1: 'dynamic',
            }
        if markers is None:
            markers = {
                0: 'o',
                1: 'o',
                2: 'x',
                3: 'x'
            }

        fig, ax1 = plt.subplots(figsize=(12, 4))
        ax1.axhline(y=avg, xmin=0., xmax=len(stats) + 1, label='random', c='gray', linestyle='--', linewidth=0.95)
        sort_keys = sorted([(k, v) for k, v in self.model_params.items() if k in models],
                           key=lambda x_: x_[1], reverse=False)

        labels_added = False
        for i, (k, k_params) in enumerate(sort_keys):
            v = stats.get(k)
            if v is None:
                continue
            for j, (p, se) in enumerate(v):
                d = {'y': p, 'x': i + 1 - (j / 8), 'c': colors.get(j), 'alpha': 0.75, 'marker': markers.get(j)}
                if not labels_added:
                    try:
                        d['label'] = labels.get(j)
                    except:
                        pass

                ax1.axvline(ymin=p - se, ymax=p + se, x=i + 1 - (j / 8), c=colors.get(j), alpha=0.3)
                ax1.scatter(**d)
            labels_added = True

        ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0),
                   ncol=1, fancybox=False, shadow=False, fontsize=7, alignment='right')

        models = [m for m, _ in sort_keys]
        ax1.axvline(ymin=0, ymax=1., x=3.5, c='gray', linewidth=1)
        ax1.axvline(ymin=0, ymax=1., x=7.5, c='gray', linewidth=1)
        ax1.set_xticks(ticks=np.arange(1, len(models) + 1), labels=[pretty_print_model_names(m) for m in models],
                       rotation=35, fontsize=8)
        ax1.spines[['right', 'top']].set_visible(False)

        plt.ylim(0, 1.)
        plt.ylabel('Avg. Discrimination Accuracy\n', fontsize=10)
        plt.xlabel('\nVerdict Model', fontsize=10)

        # Plot data on the left x-axis
        # Create a second x-axis
        if add_model_size:
            ax2 = ax1.twiny()
            # Set positions and labels for the new x-ticks
            ax2.set_xticks([1, 3, 5, 6])  # Set positions for the ticks
            ax2.set_xticklabels(['small', 'medium', 'large', ''], rotation=0, fontsize=10)  # Set the tick labels

            # Move the x-axis ticks and labels to the top
            ax2.xaxis.set_ticks_position('top')
            ax2.xaxis.set_label_position('top')

            # Hide the secondary y-axis to avoid clutter
            ax2.yaxis.set_visible(False)
            ax2.spines[['right', 'top']].set_visible(False)

            # Adjust the tick parameters to remove the small ticks under the labels
            ax2.tick_params(axis='x', which='both', bottom=False, top=False)

        plt.suptitle('The effect of LMs designing their own questions\n' if len(title) < 2 else title, y=1.02)
        plt.show()

    @staticmethod
    def len_stat(x, bins=[0, 1000, 5000, 10000, 25000, np.inf], words=True, len_sum=False):
        rps = eval(x)
        lens = [(len(r.split()) if words else len(r)) for r in rps if isinstance(r, str)]
        if len_sum:
            y = np.sum(lens)
        else:
            y = np.mean(lens)

        if bins is not None:
            binx, _ = np.histogram(y, bins=bins)
            bin_num = np.argmax(binx)
            y = bin_num

        return y

    @staticmethod
    def v_len(v, ms_, rps_, bins=[0, 1000, 5000, 10000, 25000, np.inf], other=False, words=True):
        rps = eval(rps_)
        ms = eval(ms_)
        idx = 0
        if len(ms) > 1:
            idx = ms.index(v)
        if other and len(ms) > 1:
            idx = 1 - idx

        r = rps[idx]
        r_len = len(r.split()) if words else len(r)

        binx, _ = np.histogram(r_len, bins=bins)
        bin_num = np.argmax(binx)

        return bin_num

    @staticmethod
    def get_stat_c(s, r=3, c_stde=True, c_N=True):
        N = len(s)
        mu = np.round(np.mean(s), r)
        std = np.std(s, axis=0)
        std_e = np.round(std / np.sqrt(N), r)

        c = f"{mu: .3f},".ljust(5)
        if c_stde:
            c += f" ({std_e: .3f})".ljust(6)
        if c_N:
            c += f" {N: .0f}".ljust(10)

        return c

    @staticmethod
    def get_position_stat_series(df, bin_key='len_stat_bin', cor_key='correct'):
        bias = {}

        keys = []
        weights = []
        means = []
        std_errors = []
        for bin, df_ in df.groupby([bin_key]):
            keys.append(bin[0])

            if cor_key == 'correct' or cor_key == 'best':
                A_key = True
                B_key = False
            elif cor_key == 'verdict_extract':
                A_key = 'A'
                B_key = 'B'
            else:
                raise NotImplemented

            x = df_[cor_key].value_counts()
            A = x.get(A_key, 0)
            B = x.get(B_key, 0)

            ts = np.concatenate([np.ones(A), np.zeros(B)])
            std_e = np.inf
            if len(ts) > 1:
                std_e = np.std(ts) / np.sqrt(len(ts))
            means.append(np.mean(ts))
            std_errors.append(std_e)
            weights.append(len(ts))

        weights = np.array(weights) / np.max(weights)

        for k, w, mu, std_e in zip(keys, weights, means, std_errors):
            bias[k] = {'weight': w, 'mean': mu, 'std_e': std_e}

        return bias

    def compute_biases(self, fname, biases=None, bins=None):
        vs = pd.read_csv(fname)
        vm = vs['verdict_model_name'].iloc[0]
        vs = vs[vs['n_comparisons'] == 2].copy()
        vs = vs[~vs['name_drop_any_specific_flag']].copy()

        if bins is None:
            bins = list(np.linspace(0, 2000, 30).astype(int)) + [np.inf]
        words = True

        vs['len_stat'] = vs['contestant_responses'].apply(lambda x: self.len_stat(x, bins=None, len_sum=True))
        vs['len_stat_bin'] = vs['contestant_responses'].apply(
            lambda x: self.len_stat(x, bins=bins, words=True, len_sum=True))
        vs['own_resp_len'] = vs.apply(lambda x: self.v_len(x['verdict_model_name'],
                                                           x['contestants'],
                                                           x['contestant_responses'],
                                                           other=False,
                                                           bins=bins,
                                                           words=words
                                                           ),
                                      axis=1)
        vs['other_resp_len'] = vs.apply(lambda x: self.v_len(x['verdict_model_name'],
                                                             x['contestants'],
                                                             x['contestant_responses'],
                                                             other=True,
                                                             bins=bins,
                                                             words=words
                                                             ),
                                        axis=1)
        if biases is None:
            biases = {'position_bias': {}, 'position_accuracy': {}}

        pos_b = self.get_position_stat_series(vs, cor_key='verdict_extract')
        pos_acc = self.get_position_stat_series(vs, cor_key='correct')
        biases['position_bias'][vm] = pos_b
        biases['position_accuracy'][vm] = pos_acc

        return biases

    @staticmethod
    def _plot_biases(biases, bins, title=None):

        num_biases = len(biases.keys())
        num_models = max([len(v.values()) for v in biases.values()])

        fig, axs = plt.subplots(ncols=num_biases, figsize=(14, 5))

        for i, (ax, bias) in enumerate(zip(axs, list(biases.keys()))):

            idx = max([list(v.keys())[-1] for v in biases[bias].values()])
            # max_bin= bins[idx+1]

            for j, (m, x) in enumerate(biases[bias].items()):
                color = plot_colors.get(j)
                means = np.array([x_['mean'] for x_ in x.values()])
                std_errors = np.array([x_['std_e'] for x_ in x.values()])
                weights = np.array([x_['weight'] for x_ in x.values()])

                assert len(means) == len(std_errors) == len(weights) == len(x.keys())
                ax.scatter(x=list(x.keys()), y=means, label=pretty_print_model_names(m),
                           color=color, s=np.exp(weights * 4))

                ax.plot(list(x.keys()), means, linewidth=0.7, color=color)
                ax.fill_between(list(x.keys()), means - std_errors, means + std_errors, color=color, alpha=0.2)

            ax.hlines(y=0.5, xmin=0, xmax=idx, color='gray', linestyles='--', linewidth=0.9)
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_ylim(0, 1.05)

            ax.set_xticks(ticks=[k for k in range(len(bins[:idx + 1]))])
            ax.set_xticklabels(labels=[f'[{bins[k - 1]}, {bins[k]})' for k in range(1, len(bins[:idx + 2]))],
                               rotation=-30, fontsize=8)
            ax.set_ylabel(bias)

        if title is not None:
            fig.suptitle(title)

        plt.tight_layout()
        plt.legend(loc='lower center', bbox_to_anchor=(0.0, -0.25),
                   ncol=num_models, fancybox=False, shadow=False, fontsize=10, alignment='right')

        plt.show()

    def plot_biases(self, files, bins=None, title=None):
        if bins is None:
            bins = list(np.linspace(0, 2000, 30).astype(int)) + [np.inf]

        biases = {'position_bias': {}, 'position_accuracy': {}}
        for f in files:
            biases = self.compute_biases(f, biases=biases, bins=bins)
        print(biases)

        self._plot_biases(biases=biases, bins=bins, title=title)

    def get_top_k(self, k, per_model=False, per_model_max=-1, use_yesno=True, n_comparisons=2,
                  dynamic=True, name_cats=['first', 'not_first'], swap_type=None, question_type=None,
                  return_all_merged=False, verbosity=0):
        stats = self.stats_yesno if use_yesno else self.stats_certainty
        y = stats['stats_by_n_comp_questions.csv']
        m_qs = self.filter_df(y, model_name=None, dynamic=dynamic, n_comparisons=n_comparisons,
                              name_cats=name_cats, swap_type=swap_type, question_type=question_type)
        m_qs = m_qs.groupby(['verdict_model_name', 'question'])[['correct_mean']].mean().sort_values(
            by=['verdict_model_name', 'correct_mean'], ascending=False).reset_index()

        model_qs = {}
        for m, df in m_qs.groupby('verdict_model_name'):
            qs = df.to_dict(orient='records')
            model_qs[m] = qs[:k] if per_model else qs

        if per_model:
            return model_qs

        q_merged = []
        for v in model_qs.values():
            if verbosity > 0:
                for j, i in enumerate(v):
                    c, q = i['correct_mean'], i['question']
                    print(f'{j + 1} [{c: .4f}]: {q}')
                print('\n')
            q_merged.extend(v)

        q_merged_s = sorted(q_merged, key=lambda x: x['correct_mean'], reverse=True)

        top_k = {}
        count = 0
        for q in q_merged_s:
            vm, q, c = q['verdict_model_name'], q['question'], q['correct_mean']
            res = top_k.get(vm)
            if res is None:
                top_k[vm] = [q]
            elif per_model_max < 1 or len(res) < per_model_max:
                top_k[vm] = res + [q]
            else:
                continue

            count += 1
            if count == k and k > 0:
                break

        if return_all_merged:
            return top_k, q_merged_s

        return top_k

    def get_random_sample(self, k, per_model_max=-1, use_yesno=True, n_comparisons=2,
                          dynamic=False, name_cats=['first', 'not_first'], swap_type=None, question_type=None,
                          verbosity=0):

        _, q_merged = self.get_top_k(k=-1, per_model=False, per_model_max=-1, use_yesno=use_yesno, dynamic=dynamic,
                                     n_comparisons=n_comparisons, name_cats=name_cats,
                                     swap_type=swap_type, question_type=question_type,
                                     return_all_merged=True, verbosity=verbosity)
        np.random.shuffle(q_merged)

        random_k = {}
        count = 0
        for i, q in enumerate(q_merged):
            vm, q, c = q['verdict_model_name'], q['question'], q['correct_mean']
            res = random_k.get(vm)
            if res is None:
                random_k[vm] = [q]
            elif per_model_max < 1 or len(res) < per_model_max:
                random_k[vm] = res + [q]
            else:
                continue

            count += 1
            if count == k:
                break

        return random_k

    @staticmethod
    def get_responses_for_questions(questions: dict, base_folder: str, verbosity=0) -> pd.DataFrame:
        """
        Expects a dictionary with keys as model names and values as lists of questions
        :param questions: (dict) questions per model
        :param base_folder: (str) base folder path to responses
        :param verbosity: (int) verbosity level
        :return: pd.Dataframe of responses for each question
        """
        fs = {'gpt-3.5-turbo': 'csv_gpt-3.5-turbo.csv',
              'meta-llama/Llama-3-8b-chat-hf': 'csv_meta-llama_Llama-3-8b-chat-hf.csv',
              'claude-3-opus-20240229': 'csv_claude-3-opus-20240229.csv',
              'gemini-1.0-pro': 'csv_gemini-1.0-pro.csv',
              'claude-3-sonnet-20240229': 'csv_claude-3-sonnet-20240229.csv',
              'mistralai/Mixtral-8x22B-Instruct-v0.1': 'csv_mistralai_Mixtral-8x22B-Instruct-v0.1.csv',
              'claude-3-haiku-20240307': 'csv_claude-3-haiku-20240307.csv',
              'command-r-plus': 'csv_command-r-plus.csv',
              'meta-llama/Llama-3-70b-chat-hf': 'csv_meta-llama_Llama-3-70b-chat-hf.csv',
              'gpt-4-turbo': 'csv_gpt-4-turbo.csv'}

        q_responses = []
        for k, v in questions.items():
            k_rps = pd.read_csv(os.path.join(base_folder, fs[k], 'responses.csv'))

            for q in v:
                q_rps = k_rps[k_rps['question'] == q].copy()
                if len(q_rps) < 1:
                    print('ERROR!!!', q)
                q_responses.append(q_rps)

        df = save_df_concat(q_responses).reset_index(drop=True)
        printv(f'shape of responses: {df.shape}', verbosity)

        return df

    def save_stats(self, path):
        os.makedirs(path, exist_ok=True)

        if self.stats_yesno is not None:
            for k, v in self.stats_yesno.items():
                v.to_csv(os.path.join(path, k), index=False)

        if self.stats_certainty is not None:
            for k, v in self.stats_certainty.items():
                v.to_csv(os.path.join(path, k), index=False)
