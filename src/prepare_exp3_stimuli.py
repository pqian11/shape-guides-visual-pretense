import numpy as np
import pandas as pd
import json
import os


def normalize(scores):
    total = np.sum(scores)
    return list(np.array(scores)/total)


def normalize_response(response):
    tokens = response.lower().split()

    if tokens[0] == 'a' or tokens[0] == 'an':
        return ' '.join(tokens[1:])
    else:
        return ' '.join(tokens)
    

def include_option(op):
    if op == 'sex toy' or op == 'penis' or op == 'sperm':
        return False
    else:
        tokens = op.split()
        if len(tokens) > 2:
            return False
        else:
            return True
        

def get_article(w):
    vowels = set(['a', 'e', 'i', 'o', 'u'])
    if w[0] in vowels:
        return 'an'
    else:
        return 'a'
    

# Load object names
df = pd.read_csv('data/stimuli_info/exp3_freeform_pretend.csv')
obj_names = df['label'].to_list()

# Load experiment data
data = json.load(open('data/exp3_pretense_participant_data.json'))

# Calculate option frequency
option_freq_dict = {}

for obj_name in obj_names:
    option_freq_dict[obj_name] = {}

for subject_idx, subject_data in enumerate(data):
    for trial_data in subject_data:
        if trial_data['trial_type'] != 'cloze-with-preamble':
            continue
        response = trial_data['response'][0]
        response = normalize_response(response)
        item = trial_data['item']
        if response in option_freq_dict[item]:
            option_freq_dict[item][response] += 1
        else:
            option_freq_dict[item][response] = 1


# Subsample the space of produced free-form pretense options
n_option = 3
np.random.seed(29)

item2options = {}
option2article = {}

for obj_name in obj_names:
    options = list(option_freq_dict[obj_name].keys())
    options = [op for op in options if include_option(op) and op != obj_name]
    probs = normalize([option_freq_dict[obj_name][op] for op in options])
    sampled_options = np.random.choice(options, n_option, p=probs, replace=False)
    print('{}: {}'.format(obj_name, ', '.join(sampled_options)))
    item2options[obj_name] = list(sampled_options)
    
    for option in sampled_options:
        option2article[option] = get_article(option)

# Generate random pairing for triplet judgment task
np.random.seed(11)

item2baseline = {}
for obj_name in obj_names:
    item2baseline[obj_name] = []

original_item_of_baseline_dict = dict(zip(obj_names, [{} for _ in range(len(obj_names))]))

for j in range(3):

    random_pairs = obj_names[:]
    np.random.shuffle(random_pairs)

    random_pair_dict = {}
    for i in range(int(len(random_pairs)/2)):
        o1 = random_pairs[i*2]
        o2 = random_pairs[i*2+1]
        random_pair_dict[o1] = o2
        random_pair_dict[o2] = o1

    for obj_name in obj_names:
        baseline_option = item2options[random_pair_dict[obj_name]][j]
        item2baseline[obj_name].append(baseline_option)
        assert(baseline_option not in item2options[obj_name])

        original_item_of_baseline_dict[obj_name][j] = random_pair_dict[obj_name]


print('\n150 triads for Study 3 Feature Evaluation:')
for obj_name in obj_names:
    for j in range(3):
        print('{:<20} {:<20} \t{} [from {}]'.format(obj_name, item2options[obj_name][j], item2baseline[obj_name][j], original_item_of_baseline_dict[obj_name][j]))

# # Export sampled options as stimuli set
# with open('data/stimuli_info/exp3_similarity_multi_choice_stimuli.js', 'w') as f:
#     f.write('var obj_names = ' + json.dumps(obj_names) + '\n')
#     f.write('var item2options =' + json.dumps(item2options, indent=4) + '\n')
#     f.write('var item2baseline =' + json.dumps(item2baseline, indent=4) + '\n')
#     f.write('var option2article =' + json.dumps(option2article, indent=4) + '\n')

# # Export sampled options as stimuli set
# with open('data/stimuli_info/exp3_similarity_multi_choice_stimuli.json', 'w') as f:
#     json.dump({'obj_names':obj_names, 'item2options':item2options, 
#                'item2baseline':item2baseline, 'option2article':option2article}, f, indent=6)