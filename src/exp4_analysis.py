import numpy as np
import json
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'arial'
import scipy.stats


def get_subpart_alignment_stimuli_info(path):
    df = pd.read_csv(path)
    obj_names = df['real_object'].to_list()

    item2pretense = {}
    pretense2article = {}
    item2subparts = {}

    for row_idx, row in df.iterrows():
        item = row['real_object']
        pretense_article = row['pretense_article']
        pretense = row['pretense']
        subparts = [row['subpart_1'], row['subpart_2']]
        item2pretense[item] = pretense
        pretense2article[pretense] = pretense_article
        item2subparts[item] = subparts

    return {'obj_names':obj_names, 'item2pretense':item2pretense, 'pretense2article':pretense2article, 'item2subparts':item2subparts}


def load_human_click_data(data_path, obj_names, item2subparts, verbose=False):
    data = json.load(open(data_path))

    if verbose:
        for subject_idx, subject_data in enumerate(data):
            print('subject', subject_idx, end=' ')
            print(subject_data[2]['response']['Q0'])
            assert subject_data[3]['response']['Q0'].lower().strip() == 'imagine'
            print(subject_data[3]['response']['Q0'])
            print(subject_data[-1]['response']['Q0'] + ' Feedback: ' + subject_data[-1]['response']['Q1'])

    response_dict = {}

    for obj_name in obj_names:
        response_dict[obj_name] = {}
        for instance_idx in range(3):
            response_dict[obj_name][instance_idx] = {}
            for subpart in item2subparts[obj_name]:
                response_dict[obj_name][instance_idx][subpart] = []

    n_included_participant = 0

    for i, subject_data in enumerate(data):

        if subject_data[3]['response']['Q0'].lower().strip() != 'imagine':
            continue

        for trial_data in subject_data:
            if trial_data['trial_type'] != 'image-single-click':
                continue
            assert trial_data['trial_index'] >= 4 and trial_data['trial_index'] < len(subject_data) - 1

            click = [trial_data['response']['x'], trial_data['response']['y']]
            item = trial_data['trial_info']['object']
            subpart =  trial_data['trial_info']['subpart']
            instance_idx = trial_data['trial_info']['instance_idx']
            response_dict[item][instance_idx][subpart].append(click)

        n_included_participant += 1

    print('{} participants are included from {}'.format(n_included_participant, data_path))

    return response_dict


def load_annotation_data_for_model_inpainting_samples(data_path, obj_names, item2subparts, verbose=False):
    data = json.load(open(data_path))

    if verbose:
        for subject_idx, subject_data in enumerate(data):
            print('subject', subject_idx, end=' ')
            print(subject_data[2]['response']['Q0'])
            print(subject_data[3]['response']['Q0'])
            print(subject_data[-1]['response']['Q0'] + ' Feedback: ' + subject_data[-1]['response']['Q1'])

    response_dict = {}

    for obj_name in obj_names:
        response_dict[obj_name] = {}
        for instance_idx in range(3):
            response_dict[obj_name][instance_idx] = {}
            for set_idx in range(10):
                response_dict[obj_name][instance_idx][set_idx] = {}
                for subpart in item2subparts[obj_name]:
                    response_dict[obj_name][instance_idx][set_idx][subpart] = []

    n_included_participant = 0

    for i, subject_data in enumerate(data):

        if subject_data[3]['response']['Q0'].lower().strip() != 'object':
            continue

        for trial_data in subject_data:
            if trial_data['trial_type'] != 'image-single-click':
                continue
            assert trial_data['trial_index'] >= 4 and trial_data['trial_index'] < len(subject_data) - 1

            click = [trial_data['response']['x'], trial_data['response']['y']]
            item = trial_data['trial_info']['object']
            subpart =  trial_data['trial_info']['subpart']
            instance_idx = trial_data['trial_info']['instance_idx']
            set_idx = trial_data['trial_info']['set_idx']
            pretense = trial_data['trial_info']['pretense']

            response_dict[item][instance_idx][set_idx][subpart].append(click)

        n_included_participant += 1

    print('{} participants are included from {}'.format(n_included_participant, data_path))

    return response_dict


def calc_dist_all(points1, points2):
    dists = []
    for point1, point2 in zip(points1, points2):
        dists.append(np.sqrt(np.sum(np.square((point1 - point2)))))
    return dists


def calc_deviation_scale(points):
    points = np.array(points)
    cov_matrix = np.cov(points.T)
    scale = np.sqrt(np.linalg.det(cov_matrix))
    return scale


def plot_dispersion_comparison(dispersion_list_all, savepath=None, jitter_random_seed=100):
    np.random.seed(jitter_random_seed) # Set random seed for jitter
    conds = ['pretense', 'prior']
    plt.figure(figsize=(2, 3.5))
    ax = plt.gca()

    for pretense_scale, prior_scale in zip(dispersion_list_all['pretense'], dispersion_list_all['prior']):
        plt.plot((np.random.random(2)*2-1)*0.05 + np.array([0, 1]), [pretense_scale, prior_scale], 'ko-', alpha=0.08, mfc='none', markersize=4)

    plt.plot(np.arange(2), [np.mean(dispersion_list_all[cond]) for cond in conds], color='k')

    dispersion_data_all = [dispersion_list_all[cond] for cond in conds]
    cond_colors = ['firebrick', 'gold']
    for idx in range(2):
        plt.errorbar(idx, np.mean(dispersion_data_all[idx]), yerr=scipy.stats.sem(dispersion_data_all[idx])*1.96, fmt='o',
            markersize=7, color=cond_colors[idx])

    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Pretense', 'Prior'])
    ax.set_ylabel('Dispersion '+r'($|\Sigma|^{\frac{1}{2}}$)')
    # ax.set_ylabel('Dispersion '+r'(pixel$^2$)')
    ax.set_ylim(ymin=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def plot_comparison_of_center_of_clicks(distances_for_cond_pairs, cond_comparison_pairs, savepath=None, jitter_random_seed=100):
    np.random.seed(jitter_random_seed) # Set random seed for jitter

    bar_colors = ['firebrick', 'gold', 'skyblue', 'royalblue']

    plt.figure(figsize=(5,3))
    ax = plt.gca()

    for idx, (cond1, cond2) in enumerate(cond_comparison_pairs):
        distances = distances_for_cond_pairs[cond1+' '+cond2]
        plt.plot((np.random.random(len(distances))*2-1)*0.05 + idx, distances, 'ko', alpha=0.15, mfc='none')

    for bar_idx, (cond1, cond2) in enumerate(cond_comparison_pairs):
        distances = distances_for_cond_pairs[cond1+' '+cond2]
        plt.bar(bar_idx, np.mean(distances),
                yerr=scipy.stats.sem(distances)*1.96,
                width=0.4, color=bar_colors[bar_idx])

    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(['Pretense\nvs. Pretense', 'Prior\n vs.Pretense', 'DALLE-2\nvs. Pretense', 'Stable Diffusion\nvs. Pretense'])
    ax.set_ylabel('Distance (pixels)', fontsize=12.5)
    ax.set_ylim(ymin=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


# Load stimuli information for Study 4
stimuli_info = get_subpart_alignment_stimuli_info('data/stimuli_info/exp4_subpart_alignment.csv')

obj_names = stimuli_info['obj_names']
item2pretense = stimuli_info['item2pretense']
pretense2article = stimuli_info['pretense2article']
item2subparts = stimuli_info['item2subparts']
n_instance = 3  # Number of image instances per category

response_dict_all = {}

task_conds = ['pretense', 'prior']
for task_cond in task_conds:
    data_path = 'data/exp4_{}_participant_data.json'.format(task_cond)
    response_dict_all[task_cond] = load_human_click_data(data_path, obj_names, item2subparts)

model_tags = ['dalle', 'stable_diffusion']
for model_tag in model_tags:
    data_path = 'data/exp4_subpart_anno_{}_participant_data.json'.format(model_tag)
    response_dict_all[model_tag] = load_annotation_data_for_model_inpainting_samples(data_path, obj_names, item2subparts)

print()

# Analyze click dispersion across Pretense and Prior conditions
dispersion_list_all = {}

conds = ['pretense', 'prior']

for cond in conds:

    dispersion_list = []

    for obj_name in obj_names:
        for instance_idx in range(n_instance):
            for subpart in item2subparts[obj_name]:
                dispersion = calc_deviation_scale(np.array(response_dict_all[cond][obj_name][instance_idx][subpart]))
                dispersion_list.append(dispersion)

    dispersion_mean = np.mean(dispersion_list)
    dispersion_sem = scipy.stats.sem(dispersion_list)

    print('Dispersion in {}\nMean: {}, SEM: {}\n'.format(cond, dispersion_mean, dispersion_sem))

    dispersion_list_all[cond] = dispersion_list

plot_dispersion_comparison(dispersion_list_all, savepath='fig/exp4/exp4_dispersion_plot_colored.pdf')


# Analyze distance between the center of clicks for each subpart across pairs of conditions
center_of_clicks_list_across_conds = {}

pretense_mean_clicks_half1 = []
pretense_mean_clicks_half2 = []
prior_mean_clicks = []
dalle_mean_clicks = []
stable_diffusion_mean_clicks = []

for obj_name in obj_names:
    for instance_idx in range(3):
        for subpart in item2subparts[obj_name]:
            # Split the click data in pretense condition into halves
            # As participants were recruited randomly, simply cut the pretense data list through the middle point.
            points = np.array(response_dict_all['pretense'][obj_name][instance_idx][subpart])
            n_points = len(points)

            mean_click1 = [np.mean(points[:(n_points//2), 0]), np.mean(points[:(n_points//2), 1])]
            pretense_mean_clicks_half1.append(mean_click1)

            mean_click2 = [np.mean(points[(n_points//2):, 0]), np.mean(points[(n_points//2):, 1])]
            pretense_mean_clicks_half2.append(mean_click2)

            # Compute the center of clicks for prior condition
            points = np.array(response_dict_all['prior'][obj_name][instance_idx][subpart])
            n_points = len(points)
            mean_click = [np.mean(points[:, 0]), np.mean(points[:, 1])]
            prior_mean_clicks.append(mean_click)

            # Compute the center of click annotations for DALLE-2 inpainting samples
            click_center_for_inpainting_instances = []
            for set_idx in range(10):
                points = np.array(response_dict_all['dalle'][obj_name][instance_idx][set_idx][subpart])
                click_center_for_inpainting_instances.append([np.mean(points[:, 0]), np.mean(points[:, 1])])

            click_center_for_inpainting_instances = np.array(click_center_for_inpainting_instances)
            
            mean_click = [np.mean(click_center_for_inpainting_instances[:, 0]), np.mean(click_center_for_inpainting_instances[:, 1])]
            dalle_mean_clicks.append(mean_click)

            # Compute the center of click annotations for Stable Diffusion inpainting samples
            click_center_for_inpainting_instances = []
            for set_idx in range(10):
                points = np.array(response_dict_all['stable_diffusion'][obj_name][instance_idx][set_idx][subpart])
                click_center_for_inpainting_instances.append([np.mean(points[:, 0]), np.mean(points[:, 1])])

            click_center_for_inpainting_instances = np.array(click_center_for_inpainting_instances)
            
            mean_click = [np.mean(click_center_for_inpainting_instances[:, 0]), np.mean(click_center_for_inpainting_instances[:, 1])]
            stable_diffusion_mean_clicks.append(mean_click)

center_of_clicks_list_across_conds['pretense_random_split_1'] = pretense_mean_clicks_half1
center_of_clicks_list_across_conds['pretense_random_split_2'] = pretense_mean_clicks_half2
center_of_clicks_list_across_conds['prior'] = prior_mean_clicks
center_of_clicks_list_across_conds['dalle'] = dalle_mean_clicks
center_of_clicks_list_across_conds['stable_diffusion'] = stable_diffusion_mean_clicks


cond_comparison_pairs = [
    ['pretense_random_split_1', 'pretense_random_split_2'],
    ['pretense_random_split_1', 'prior'],
    ['pretense_random_split_1', 'dalle'],
    ['pretense_random_split_1', 'stable_diffusion']
]

distances_for_cond_pairs = {}

for cond1, cond2 in cond_comparison_pairs:
    distances_for_cond_pairs[cond1+' '+cond2] = calc_dist_all(
        np.array(center_of_clicks_list_across_conds[cond1]), np.array(center_of_clicks_list_across_conds[cond2]))

plot_comparison_of_center_of_clicks(distances_for_cond_pairs, cond_comparison_pairs, savepath='fig/exp4/compare_where_people_click_across_conditions_colored.pdf')
