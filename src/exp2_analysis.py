import os
import json
import numpy as np
import scipy.stats
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.rcParams['font.family'] = 'arial'
from helper import *


def plot_pretense_preference_against_feature_judgment_full_panel(obj_names, item2options, pref_data_all, task_xs, task_y,
                                                      figsize=(9, 3), x_labels=None, y_label=None, savepath=None):
    color_dict = dict(zip(['pretense', 'shape', 'color', 'clip'], ['k', "#0072B2", "#D55E00", "#EFC000FF"]))
    axis_label_font_size = 12.5

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    fig.tight_layout()

    for ax_idx in range(len(axes)):
        ax = axes[ax_idx]
        task_x = task_xs[ax_idx]
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

        # Plot the preferred pretense options
        xs = []
        ys = []
        xerrs = [[], []] if task_x != 'clip' else None
        yerrs = [[], []]
        for n, obj_name in enumerate(obj_names):
            op1, op2 = item2options[obj_name]

            # Select the preferred pretense option for each item
            op = op1 if pref_data_all['pretense'][obj_name][op1] > pref_data_all['pretense'][obj_name][op2] else op2

            if task_x == 'clip':
                pref_x = pref_data_all[task_x][obj_name][op]
            else:
                pref_x = pref_data_all[task_x][obj_name][op]/(pref_data_all[task_x][obj_name][op1] + pref_data_all[task_x][obj_name][op2])
                pref_x_ci_low, pref_x_ci_upp = proportion_confint(
                    pref_data_all[task_x][obj_name][op], 
                    (pref_data_all[task_x][obj_name][op1] + pref_data_all[task_x][obj_name][op2])
                    )
            
            pref_y = pref_data_all[task_y][obj_name][op]/(pref_data_all[task_y][obj_name][op1] + pref_data_all[task_y][obj_name][op2])
            pref_y_ci_low, pref_y_ci_upp = proportion_confint(
                pref_data_all[task_y][obj_name][op], 
                (pref_data_all[task_y][obj_name][op1] + pref_data_all[task_y][obj_name][op2])
                )

            xs.append(pref_x)
            if task_x != 'clip':
                xerrs[0].append(pref_x - pref_x_ci_low)
                xerrs[1].append(pref_x_ci_upp - pref_x)

            ys.append(pref_y)
            yerrs[0].append(pref_y - pref_y_ci_low)
            yerrs[1].append(pref_y_ci_upp - pref_y)

        ax.errorbar(xs, ys, xerr=xerrs, yerr=yerrs, fmt='o', ecolor='k', mec='none', mfc='none', alpha=0.2, elinewidth=0.5)
        ax.plot(xs, ys, 'o', mec=color_dict[task_x], mfc='w', alpha=0.8)

        # Plot the dispreferred pretense options; the data is symmetrical with respect to the preferred options
        xs2 = []
        ys2 = []

        for n, obj_name in enumerate(obj_names):
            op1, op2 = item2options[obj_name]

            # Select the dispreferred pretense option for each item
            op = op2 if pref_data_all['pretense'][obj_name][op1] > pref_data_all['pretense'][obj_name][op2] else op1

            if task_x == 'clip':
                pref_x = pref_data_all[task_x][obj_name][op]
            else:
                pref_x = pref_data_all[task_x][obj_name][op]/(pref_data_all[task_x][obj_name][op1] + pref_data_all[task_x][obj_name][op2])
                pref_x_ci_low, pref_x_ci_upp = proportion_confint(
                    pref_data_all[task_x][obj_name][op], 
                    (pref_data_all[task_x][obj_name][op1] + pref_data_all[task_x][obj_name][op2])
                    )
            
            pref_y = pref_data_all[task_y][obj_name][op]/(pref_data_all[task_y][obj_name][op1] + pref_data_all[task_y][obj_name][op2])
            pref_y_ci_low, pref_y_ci_upp = proportion_confint(
                pref_data_all[task_y][obj_name][op], 
                (pref_data_all[task_y][obj_name][op1] + pref_data_all[task_y][obj_name][op2])
                )

            xs2.append(pref_x)
            ys2.append(pref_y)

        ax.plot(xs2, ys2, 'o', mec='gainsboro', mfc='#ebebeb', alpha=0.9, zorder=-1)

        if x_labels:
            x_label = x_labels[ax_idx]
            ax.set_xlabel(x_label, fontsize=axis_label_font_size)
        if y_label and ax_idx == 0:
            ax.set_ylabel(y_label, fontsize=axis_label_font_size)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Annotate the Mean Squared Error
        mse = mean_squared_error(ys, xs)
        mse_text_patch = ax.text(0.65, 0.03, r'$MSE=${:.3f}'.format(mse), transform=ax.transAxes)
        mse_text_patch.set_bbox(dict(facecolor='white', alpha=0.25, edgecolor='none'))

        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.show()


def plot_pretense_preference_against_feature_judgment_for_preferred_option_full_panel(obj_names, item2options, pref_data_all, task_xs, task_y,
                                                      figsize=(9, 3), x_labels=None, y_label=None, savepath=None):
    color_dict = dict(zip(['pretense', 'shape', 'color', 'clip'], ['k', "#0072B2", "#D55E00", "#EFC000FF"]))
    axis_label_font_size = 12.5

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.tight_layout()

    # Only plot the preferred option
    for ax_idx in range(len(axes)):
        ax = axes[ax_idx]
        task_x = task_xs[ax_idx]
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

        xs = []
        ys = []
        xerrs = [[], []] if task_x != 'clip' else None
        yerrs = [[], []]
        for n, obj_name in enumerate(obj_names):
            op1, op2 = item2options[obj_name]

            # Select the preferred pretense option for each item
            op = op1 if pref_data_all['pretense'][obj_name][op1] > pref_data_all['pretense'][obj_name][op2] else op2

            if task_x == 'clip':
                pref_x = pref_data_all[task_x][obj_name][op]
            else:
                pref_x = pref_data_all[task_x][obj_name][op]/(pref_data_all[task_x][obj_name][op1] + pref_data_all[task_x][obj_name][op2])
                pref_x_ci_low, pref_x_ci_upp = proportion_confint(
                    pref_data_all[task_x][obj_name][op], 
                    (pref_data_all[task_x][obj_name][op1] + pref_data_all[task_x][obj_name][op2])
                    )
            
            pref_y = pref_data_all[task_y][obj_name][op]/(pref_data_all[task_y][obj_name][op1] + pref_data_all[task_y][obj_name][op2])
            pref_y_ci_low, pref_y_ci_upp = proportion_confint(
                pref_data_all[task_y][obj_name][op], 
                (pref_data_all[task_y][obj_name][op1] + pref_data_all[task_y][obj_name][op2])
                )

            xs.append(pref_x)
            if task_x != 'clip':
                xerrs[0].append(pref_x - pref_x_ci_low)
                xerrs[1].append(pref_x_ci_upp - pref_x)

            ys.append(pref_y)
            yerrs[0].append(pref_y - pref_y_ci_low)
            yerrs[1].append(pref_y_ci_upp - pref_y)

        ax.errorbar(xs, ys, xerr=xerrs, yerr=yerrs, fmt='o', ecolor='k', mec='none', mfc='none', alpha=0.2, elinewidth=0.5)
        ax.plot(xs, ys, 'o', mec=color_dict[task_x], mfc='w', alpha=0.8)

        if x_labels:
            x_label = x_labels[ax_idx]
            ax.set_xlabel(x_label, fontsize=axis_label_font_size)
        if y_label and ax_idx == 0:
            ax.set_ylabel(y_label, fontsize=axis_label_font_size)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Annotate Spearman correlation
        spearmanr, spearmanr_p_value = scipy.stats.spearmanr(xs, ys)
        significance_anno = get_significance_level_stars(spearmanr_p_value)
        text_patch = ax.text(0.65, 0.1, r'$\rho=${:.3f}{}'.format(spearmanr, significance_anno), transform=ax.transAxes) 
        text_patch.set_bbox(dict(facecolor='white', alpha=0.25, edgecolor='none'))

        # Annotate Mean Squared Error
        mse = mean_squared_error(ys, xs)
        mse_text_patch = ax.text(0.65, 0.03, r'$MSE=${:.3f}'.format(mse), transform=ax.transAxes)
        mse_text_patch.set_bbox(dict(facecolor='white', alpha=0.25, edgecolor='none'))

        ax.set_xlim(0, 1.05)
        ax.set_ylim(0.45, 1.05)

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.show()


def plot_shape_color_comparison(obj_names, pref_data_all, item2options, item2op, figsize=(4,3), savepath=None, jitter_random_seed=1001):
    np.random.seed(jitter_random_seed)
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ratings = {'shape':[], 'color':[]}
    for obj_name in obj_names:
        op1, op2 = item2options[obj_name]
        op = item2op[obj_name]
        pref_color = pref_data_all['color'][obj_name][op]/(pref_data_all['color'][obj_name][op1]
            + pref_data_all['color'][obj_name][op2])
        pref_shape = pref_data_all['shape'][obj_name][op]/(pref_data_all['shape'][obj_name][op1]
            + pref_data_all['shape'][obj_name][op2])
        ratings['shape'].append(pref_shape)
        ratings['color'].append(pref_color)

    norming_conds = ['shape', 'color']
    cond_colors = [color_dict[cond] for cond in norming_conds]
    x_jitter_by_cond_lists = (np.random.random((len(norming_conds), len(obj_names)))*2-1)*0.05

    for j in range(len(obj_names)):
        plt.plot(np.arange(2)+x_jitter_by_cond_lists[:, j], [ratings[cond][j] for cond in norming_conds], 'k-', alpha=0.08)

    for k, cond in enumerate(norming_conds):
        plt.plot(k + x_jitter_by_cond_lists[k], ratings[cond], 'o', c=cond_colors[k], mfc='none', alpha=0.2) 

    for k, cond in enumerate(norming_conds):
        plt.errorbar(k, np.mean(ratings[cond]), yerr=scipy.stats.sem(ratings[cond])*1.96, color=cond_colors[k], marker='o')

    # plt.xlim(-0.5, 1.5)
    plt.xlim(-0.2, 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(['Shape', 'Color'], fontsize=12)
    plt.ylabel('Similarity preference', fontsize=12)
    plt.ylim(0, 1.05)

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')

    plt.show()


# Load stimuli information
df = pd.read_csv('data/stimuli_info/exp2_pretend-to-real.csv')

stimuli_all = []
item2options = {}
item2option_set = {}
obj2article = {}
obj_names = []

for row_idx, row in df.iterrows():
    op1 = row['real_object1']
    op2 = row['real_object2']
    article = row['article_of_pretend_object']
    pretend_obj = row['pretend_object']

    options = [op1, op2]

    stimulus = {}
    stimulus['pretend_object'] = pretend_obj
    stimulus['options'] = options
    stimuli_all.append(stimulus)

    obj_names.append(pretend_obj)
    item2options[pretend_obj] = options
    item2option_set[pretend_obj] = set(options)
    obj2article[pretend_obj] = article


print(len(obj_names), 'objects selected')
print(df)
print()


# Load human behavioral data from pretense, shape, and color tasks
pref_data_all = {}
data_all = {}

task_conditions = ['pretense', 'shape', 'color']
attention_check_words = ['imagine', 'shape', 'color']

for task_idx, cond in enumerate(task_conditions):
    data = load_human_data(path='data/exp2_{}_participant_data.json'.format(cond))
    attention_check_word = attention_check_words[task_idx]

    pref_data = {}
    for obj_name in obj_names:
        pref_data[obj_name] = {}
        for option in item2options[obj_name]:
            pref_data[obj_name][option] = 0

    n_included_participant = 0
    data_included = []

    for subject_data in data:
        if subject_data[3]['response']['Q0'].lower().strip() != attention_check_word:
            print(subject_data[2]['response']['Q0'], subject_data[3]['response']['Q0'].lower().strip())
            continue

        for trial_data in subject_data:
            if trial_data['trial_type'] != 'survey-multi-image-choice':
                continue
            response = trial_data['response']['choice']
            item = trial_data['item']
            pref_data[item][response] += 1

        data_included.append(subject_data)
        n_included_participant += 1

    print('Choice distribution in the {} task ({} participants recruited, {} included):'.format(cond.upper(), len(data), n_included_participant))
    print('-'*68)
    for obj_name in obj_names:
        op1, op2 = item2options[obj_name]
        print('{:<20} {:<15} ({})\t{:<15} ({})'.format(obj_name, op1, pref_data[obj_name][op1], op2, pref_data[obj_name][op2]))
        
    pref_data_all[cond] = pref_data
    data_all[cond] = data_included

    for subject_idx, subject_data in enumerate(data_all[cond]):
        assert(subject_data[3]['response']['Q0'].lower().strip() == attention_check_word)

    print()

# Load representational similarity judgment from CLIP
pref_data_all['clip'] = json.load(open('data/exp2_clip_pref_data.json'))

# Construct a dictionary that maps an item of pretend object to the preferred real object option
item2op = {}
for obj_name in obj_names:
    op1, op2 = item2options[obj_name]
    op = op1 if pref_data_all['pretense'][obj_name][op1] > pref_data_all['pretense'][obj_name][op2] else op2
    item2op[obj_name] = op

color_dict = dict(zip(['pretense', 'shape', 'color', 'clip'], ['k', "#0072B2", "#D55E00", "#EFC000FF"]))


plot_pretense_preference_against_feature_judgment_full_panel(obj_names, item2options, pref_data_all, 
                                                            task_xs=['shape', 'color', 'clip'], task_y='pretense',
                                                            x_labels=['Shape similarity', 'Color similarity', 'CLIP similarity'], 
                                                            y_label='Pretense preference',
                                                            figsize=(9, 3), savepath='fig/exp2_pretense_pref_feature_full_panel.pdf')

plot_pretense_preference_against_feature_judgment_for_preferred_option_full_panel(obj_names, item2options, pref_data_all, 
                                                            task_xs=['shape', 'color', 'clip'], task_y='pretense',
                                                            x_labels=['Shape similarity', 'Color similarity', 'CLIP similarity'], 
                                                            y_label='Pretense preference',
                                                            figsize=(9, 3), savepath='fig/exp2_pretense_pref_feature_for_preferred_pretend_object_full_panel.pdf')

plot_shape_color_comparison(obj_names, pref_data_all, item2options, item2op, figsize=(2.5,3), 
                                    savepath='fig/exp2_compare_feature_judgment_for_preferred_pretend_object.pdf')


# Compute summary statistics and run statistical tests
lhr_test_stats, lhr_p_value = likelihood_ratio_test_on_pretend_preference(obj_names, item2options, pref_data_all['pretense'])
print('Likelihood ratio test of pretense preference: lambda_LR={}, p={}\n'.format(lhr_test_stats, lhr_p_value))

pref_ps_all = {}
conds = ['pretense', 'shape', 'color', 'clip']
for cond in conds:
    pref_ps_all[cond] = []

for obj_name in obj_names:
    op1, op2 = item2options[obj_name]
    op = item2op[obj_name] # preferred option

    for cond in conds:
        if cond == 'clip':
            pref_ps_all[cond].append(pref_data_all['clip'][obj_name][op])
        else:
            pref_ps_all[cond].append(pref_data_all[cond][obj_name][op]/(pref_data_all[cond][obj_name][op1] + pref_data_all[cond][obj_name][op2]))

for cond in ['shape', 'color']:
    prop_mean = np.mean(pref_ps_all[cond])
    prop_sem = scipy.stats.sem(pref_ps_all[cond])
    print('Proportion of choosing the preferred pretend option in the {cond} task\nMean: {prop_mean:.3f}, 95% CI=[{ci_low:.3f}, {ci_upp:.3f}]\n'.format(
        cond=cond, prop_mean=prop_mean, ci_low=prop_mean-1.96*prop_sem, ci_upp=prop_mean+1.96*prop_sem))

paired_ttest_rs = scipy.stats.ttest_rel(pref_ps_all['shape'], pref_ps_all['color'])
print('Paired t-test between shape and color: t({})={:.3f}, p={}\n'.format(paired_ttest_rs.df, paired_ttest_rs.statistic, paired_ttest_rs.pvalue))

for cond in ['shape', 'color', 'clip']:
    rho, p_value = scipy.stats.spearmanr(pref_ps_all['pretense'], pref_ps_all[cond])
    print('Spearman rho(pretense, {}): {:.3f} (p={})'.format(cond, rho, p_value))

rho, p_value = scipy.stats.spearmanr(pref_ps_all['shape'], pref_ps_all['clip'])
print('Spearman rho(shape, clip): {:.3f} (p={})'.format(rho, p_value))
