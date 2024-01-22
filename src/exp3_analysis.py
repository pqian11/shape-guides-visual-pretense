import numpy as np
import json
import os
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
matplotlib.rcParams['font.family'] = 'arial'
from helper import load_human_data

def load_exp_data(dir_path):
    fnames = sorted(os.listdir(dir_path))
    data = []
    for fname in fnames:
        subject_data = json.load(open(os.path.join(dir_path, fname)))
        data.append(subject_data)
    return data 


# Load stimuli information for the feature evaluation experiment of Study 3
exp3_info = json.load(open('data/stimuli_info/exp3_similarity_multi_choice_stimuli.json'))
obj_names = exp3_info['obj_names']
item2options = exp3_info['item2options']   # A dictionary that maps an item to three subsampled pretend options generated for that item
item2baseline = exp3_info['item2baseline'] # A dictionary that maps an item to three randomly paired options that were generated for other items


# Load human behavioral data in the shape and color tasks
task_conditions = ['shape', 'color']
attention_check_words = ['shape', 'color']

data_all = {}
pref_data_all = {}

for task_idx, cond in enumerate(task_conditions):
    data = load_human_data('data/exp3_{}_participant_data.json'.format(cond))
    attention_check_word = attention_check_words[task_idx]
    for subject_idx, subject_data in enumerate(data):
        assert(subject_data[3]['response']['Q0'].strip().lower() == attention_check_word)

    data_all[cond] = data

    pref_data = {}
    for obj_name in obj_names:
        pref_data[obj_name] = {}
        for op in item2options[obj_name]:
            pref_data[obj_name][op] = 0

        for op in item2baseline[obj_name]:
            pref_data[obj_name][op] = 0

    for subject_data in data:
        for trial_data in subject_data:
            if trial_data['trial_type'] != 'image-multi-choice':
                continue
            item = trial_data['item']
            response = trial_data['response']['choice']
            selected_op = ' '.join(response.split()[1:])
            pref_data[item][selected_op] += 1

    pref_data_all[cond] = pref_data

color_dict = dict(zip(['pretense', 'shape', 'color', 'clip'], ['k', "#0072B2", "#D55E00", "#EFC000FF"]))


# Plot the distribution of shape and color judgment data for the pretend options that were generated for the corresponding items
plt.figure(figsize=(5,3))
ax = plt.gca()
ratings = {}
norming_tasks = ['shape', 'color']
for task in norming_tasks:
    ratings[task] = []
    for obj_name in obj_names:
        for op_pair_idx, pretense_op in enumerate(item2options[obj_name]):
            random_pair_op = item2baseline[obj_name][op_pair_idx]
            preference = pref_data_all[task][obj_name][pretense_op]/(pref_data_all[task][obj_name][pretense_op] + pref_data_all[task][obj_name][random_pair_op])
            ratings[task].append(preference)

for k, cond in enumerate(norming_tasks):
    density = scipy.stats.gaussian_kde(ratings[cond])
    xs = np.arange(0., 1.0, 0.01)
    ys = density(xs)
    plt.plot(xs, ys, c=color_dict[cond])
    plt.fill_between(xs, ys, alpha=0.2, color=color_dict[cond])
    plt.axvline(np.mean(ratings[cond]), color=color_dict[cond], ls='-')

condition_legend_elements = []
pretty_cond_names = ['Shape', 'Color']

for cond_idx, cond in enumerate(norming_tasks):
    patch = mpatches.Patch(ec=color_dict[cond], fc=matplotlib.colors.to_rgba(color_dict[cond], 0.2), label=pretty_cond_names[cond_idx])
    condition_legend_elements.append(patch)

legend = plt.legend(handles=condition_legend_elements, ncol=1, loc='upper left', bbox_to_anchor=(0.05, 1))
ax.add_artist(legend)

plt.ylim(ymin=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Similarity preference', fontsize=12.5)
plt.ylabel('Density', fontsize=12.5)

plt.xlim(0, 1)

plt.savefig('fig/exp3_feature_evaluation_for_subsampled_pretense_options_relative_to_random_pairs_density_plot.pdf', 
             bbox_extra_artists=[legend],bbox_inches='tight')
plt.show()

# Run statistical tests
pref_ps_all = {}
conds = ['shape', 'color']
for cond in conds:
    pref_ps_all[cond] = []

for obj_name in obj_names:
    for op_idx, op in enumerate(item2options[obj_name]):
        op1, op2 = item2options[obj_name][op_idx], item2baseline[obj_name][op_idx]
        for cond in conds:
            pref_p = pref_data_all[cond][obj_name][op]/(pref_data_all[cond][obj_name][op1] + pref_data_all[cond][obj_name][op2])
            pref_ps_all[cond].append(pref_p)

for cond in ['shape', 'color']:
    prop_mean = np.mean(pref_ps_all[cond])
    prop_sem = scipy.stats.sem(pref_ps_all[cond])
    print('Proportion of choosing the geenrated pretend option in the {cond} task\nMean: {prop_mean:.3f}, 95% CI=[{ci_low:.3f}, {ci_upp:.3f}]\n'.format(
        cond=cond, prop_mean=prop_mean, ci_low=prop_mean-1.96*prop_sem, ci_upp=prop_mean+1.96*prop_sem))

paired_ttest_rs = scipy.stats.ttest_rel(pref_ps_all['shape'], pref_ps_all['color'])
print('Paired t-test between shape and color: t({})={:.3f}, p={}\n'.format(paired_ttest_rs.df, paired_ttest_rs.statistic, paired_ttest_rs.pvalue))
