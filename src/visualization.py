import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import LatentModel, model_order, model_mapping, plot_colors


def collect_data(root_fld: str):
    dfs = []
    for root, dirs, files in os.walk(root_fld):
        for file in files:
            if file.endswith('verdicts_with_stats.csv') and ('u_yesno' in root or 'u_pref' in root):
                df = pd.read_csv(os.path.join(root, file))
                dfs.append(df)
    return pd.concat(dfs)


def preprocess_data(df: pd.DataFrame):
    df = df[~df['name_drop_any_specific_flag']]
    df.loc[:, 'contestant_responses'] = df.loc[:, 'contestant_responses'].apply(eval)
    df.loc[:, 'contestants'] = df.loc[:, 'contestants'].apply(eval)
    df.loc[:, 'verdict_model_name'] = df.loc[:, 'verdict_model_name'].apply(lambda x: model_mapping.get(x, x))
    df.loc[:, 'question_model_name'] = df.loc[:, 'question_model_name'].apply(lambda x: model_mapping.get(x, x))
    df.loc[:, 'contestants'] = df.loc[:, 'contestants'].apply(lambda c: [model_mapping.get(x, x) for x in c])
    return df


def plot_data(accuracies, nrows: int, ncols: int, idx: int, title: str = None, annot: bool = True):
    plt.subplot(nrows, ncols, idx)
    ax = plt.gca()
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    sns.heatmap(accuracies, annot=annot, vmin=0, vmax=1,
                cmap='coolwarm', yticklabels=((idx - 1) % ncols == 0),
                xticklabels=((idx - 1) // ncols == nrows - 1),
                square=True, linecolor='white', linewidths=1, cbar=False)
    ax.tick_params(axis='x', which='major', pad=10)
    ax.tick_params(axis='y', which='major', pad=10)
    ax.xaxis.labelpad = 10

    if title is not None:
        plt.title(title)


def get_conf_comp_matrix(verdict_model_name: str, df_: pd.DataFrame):
    df = df_.copy()
    n_comparisons = df.iloc[0]['n_comparisons']
    df['correct_position'] = df.loc[:, 'contestants'].apply(lambda x: x.index(verdict_model_name))

    def filter_verd(verd):
        if not isinstance(verd, str) or len(verd) != 1:
            return None
        if ord(verd) - ord('A') >= n_comparisons:
            return None
        return ord(verd) - ord('A')

    df['verdict_position'] = df.loc[:, 'verdict_extract'].apply(filter_verd)
    df = df[['correct_position', 'verdict_position']]
    df['count'] = 1
    df['correct_position'] = df['correct_position'].astype(pd.Int64Dtype())
    df['verdict_position'] = df['verdict_position'].astype(pd.Int64Dtype())
    matrix = df.pivot_table(columns='correct_position', index='verdict_position', values='count', aggfunc='sum').fillna(
        0)
    matrix /= matrix.to_numpy().sum()
    return matrix


class Visualization:
    def __init__(self, path_to_random: str, path_to_dynamic: str):

        self.df_full = collect_data(path_to_random)
        self.df_dynamic = collect_data(path_to_dynamic)

        self.df_full = preprocess_data(self.df_full)
        self.df_dynamic = preprocess_data(self.df_dynamic)

        self.cure_data()

        lm2 = LatentModel(n_comparisons=2, n_samples=3 * 10 ** 6)
        lm3 = LatentModel(n_comparisons=3, n_samples=3 * 10 ** 6)
        lm5 = LatentModel(n_comparisons=5, n_samples=3 * 10 ** 6)

        self.lms = {
            2: lm2,
            3: lm3,
            5: lm5
        }

        self.order = [model_mapping[model] for model in model_order]

    def redistribute_data(self, n_comparisons: int, num_questions: int):
        sub_dynamic = self.df_dynamic[self.df_dynamic['n_comparisons'] == n_comparisons]

        for verdict_model_name, group in sub_dynamic.groupby('verdict_model_name'):
            questions = group['question'].unique()[:num_questions]

            to_add = sub_dynamic[
                (sub_dynamic['verdict_model_name'] == verdict_model_name) & (sub_dynamic['question'].isin(questions))]
            self.df_full = pd.concat([self.df_full, to_add])

    def cure_data(self):
        if len(self.df_dynamic[self.df_dynamic['n_comparisons'] == 2]) == 0:
            # PUT from random
            dynamic_in_random = self.df_full[(self.df_full['n_comparisons'] == 2) & (
                        self.df_full['verdict_model_name'] == self.df_full['question_model_name'])]
            self.df_dynamic = pd.concat([self.df_dynamic, dynamic_in_random])
        else:
            # PUT from dynamic only verdicts with questions that are present in random for n_comparisons=2
            self.redistribute_data(2, 25)

        # PUT from dynamic only verdicts with questions that are present in random for n_comparisons=3,5
        self.redistribute_data(3, 5)
        self.redistribute_data(5, 5)

    def plot_confusion_matrix_verdict_contestant(self, n_comparisons: int, full_or_dynamic: str = 'full',
                                                 nrows: int = 1, ncols: int = 1, idx: int = 1, title: str = None,
                                                 do_plot: bool = True):

        assert n_comparisons == 2, 'Only 2 comparisons are supported for this plot'
        if full_or_dynamic == 'full':
            df = self.df_full.copy()
        elif full_or_dynamic == 'dynamic':
            df = self.df_dynamic.copy()
        else:
            raise ValueError('full_or_dynamic must be either "full" or "dynamic"')

        def get_other_contestant(entry):
            if entry['contestants'][0] == entry['verdict_model_name']:
                return entry['contestants'][1]
            else:
                return entry['contestants'][0]

        df = df[df['n_comparisons'] == n_comparisons]
        assert len(df) > 0, f'No data for n_comparisons={n_comparisons}'

        needed_columns = ['correct', 'verdict_model_name', 'contestants', 'question', 'question_model_name',
                          'correct_position']
        df = df[needed_columns]
        df['other_contestant'] = df.apply(get_other_contestant, axis=1)

        df = df.groupby(['verdict_model_name', 'other_contestant', 'correct_position'])['correct'].mean().reset_index()
        accuracies = df.groupby(['verdict_model_name', 'other_contestant'])['correct'].mean().reset_index()

        order_ = [model for model in self.order if model in accuracies['verdict_model_name'].unique()]
        result = pd.pivot_table(accuracies, index='verdict_model_name', columns='other_contestant', values='correct')
        result = result.loc[order_, order_]

        if do_plot:
            plot_data(result, title=title, nrows=nrows, ncols=ncols, idx=idx)

        plt.ylabel('Verdict model')
        plt.xlabel('Contestant model')

        return result

    def plot_confusion_matrix_verdict_question(self, n_comparisons: int, full_or_dynamic: str = 'full', nrows: int = 1,
                                               ncols: int = 1, idx: int = 1, title: str = None):
        if full_or_dynamic == 'full':
            df = self.df_full.copy()
        elif full_or_dynamic == 'dynamic':
            df = self.df_dynamic.copy()
        else:
            raise ValueError('full_or_dynamic must be either "full" or "dynamic"')

        df = df[df['n_comparisons'] == n_comparisons]
        assert len(df) > 0, f'No data for n_comparisons={n_comparisons}'

        accuracies = df.groupby(['verdict_model_name', 'question_model_name'])['correct'].mean().reset_index()

        order_ = [model for model in self.order if model in accuracies['verdict_model_name'].unique()]
        result = pd.pivot_table(accuracies, index='verdict_model_name', columns='question_model_name', values='correct')
        result = result.loc[order_, order_]

        plot_data(result, title=title, nrows=nrows, ncols=ncols, idx=idx)

        plt.ylabel('Verdict model')
        plt.xlabel('Question model')

        return result

    def plot_confusion_matrix_bias(self, full_or_dynamic: str = 'full', verdict_model_names: list = None,
                                   title: str = None):
        if full_or_dynamic == 'full':
            df = self.df_full.copy()
        elif full_or_dynamic == 'dynamic':
            df = self.df_dynamic.copy()
        else:
            raise ValueError('full_or_dynamic must be either "full" or "dynamic"')

        if not verdict_model_names:
            verdict_model_names = df['verdict_model_name'].unique().tolist()

        for i, verdict_model_name in enumerate(verdict_model_names):
            group = df[df['verdict_model_name'] == verdict_model_name]
            for j, n_comparisons in enumerate([2, 3, 5]):
                plt.subplot(len(verdict_model_names), 3, i * 3 + j + 1)
                matrix = get_conf_comp_matrix(verdict_model_name, group[group['n_comparisons'] == n_comparisons].copy())
                sns.heatmap(matrix, annot=n_comparisons != 5, cbar=False, fmt='.2f', center=matrix.to_numpy().mean(),
                            cmap='coolwarm', linewidths=1, linecolor='white')
                plt.text(
                    s=f'{verdict_model_name}' if j == 1 else None,
                    x=0.5, y=-0.1, ha='center', va='top', transform=plt.gca().transAxes
                )
                plt.gca().xaxis.set_label_position('top')
                if i == 0:
                    plt.xlabel('Correct position')
                else:
                    plt.xlabel('')
                if j == 0:
                    plt.ylabel('Verdict position')
                else:
                    plt.ylabel('')

                plt.gca().tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        plt.suptitle(title)
        plt.tight_layout()

    def plot_accuracies_among_n_comparisons(self, full_or_dynamic: str = 'full', title: str = None):
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if full_or_dynamic == 'full':
            df = self.df_full.copy()
        elif full_or_dynamic == 'dynamic':
            df = self.df_dynamic.copy()
        else:
            raise ValueError('full_or_dynamic must be either "full" or "dynamic"')

        dfs = {
            2: df[df['n_comparisons'] == 2],
            3: df[df['n_comparisons'] == 3],
            5: df[df['n_comparisons'] == 5]
        }

        for n_comp in [2, 3, 5]:
            dfi = dfs[n_comp]
            assert len(dfi) > 0, f'No data for n_comparisons={n_comp}'

        models = []
        plot_colors_ = [plot_colors[1], plot_colors[5], plot_colors[7]]
        markers = ['o', '*', 'x']

        for i, vname in enumerate(self.order[::-1]):
            group = df[df['verdict_model_name'] == vname]
            models.append(vname)
            for j, n_comp in enumerate([2, 3, 5]):
                lm = self.lms[n_comp]
                marker = markers[j]
                color = plot_colors_[j]
                subgroup = group[group['n_comparisons'] == n_comp]

                if len(subgroup) == 0:
                    print('Warning: no data for', vname, n_comp)
                    continue

                init_acc = subgroup['correct'].mean()
                std = subgroup['correct'].std() / np.sqrt(subgroup['correct'].count())
                init_left = init_acc - 1.96 * std
                init_right = init_acc + 1.96 * std

                map_to_2 = lambda a: self.lms[2].get_accuracy(lm.get_shift(a))
                # print(init_acc, init_left, init_right)

                acc = map_to_2(init_acc)
                left = map_to_2(init_left)
                right = map_to_2(init_right)
                plt.scatter([acc], [i + j / 10], color=color, marker=marker,
                            label=f'{n_comp} answers' if i == 0 else None)
                plt.hlines(i + j / 10, left, right, color=color)

        plt.title(title)
        plt.yticks(range(len(models)), models)
        plt.xlim(0.15, 0.85)
        # plt.legend(loc='lower right')
        plt.axvline(0.5, ls='--', c='gray', label='Random')
        plt.xlabel('Remapped $n=2$ Accuracy')
        plt.tight_layout()
