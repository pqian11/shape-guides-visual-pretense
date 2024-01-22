import os
import json
import numpy as np
from scipy.stats.distributions import chi2


def get_significance_level_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

def load_human_data(path):
    if os.path.isdir(path):
        fnames = sorted(os.listdir(path))
        data = []
        for fname in fnames:
            subject_data = json.load(open(os.path.join(ppath, fname)))
            data.append(subject_data)
    else:
        data = json.load(open(path))
    return data 


def likelihood_ratio_test_on_pretend_preference(obj_names, item2options, pref_data):
    # Test whether a list of binomial distributions consists of at least some non-fair coins
    # H0: every coin is a fair unbiased coin (p=0.5)
    # H1: the weights of the coins can be biased; H0 is nested in H1.
    H1_ps = []
    loglh_H0 = 0
    loglh_H1 = 0
    for n, obj_name in enumerate(obj_names):
        op1, op2 = item2options[obj_name]

        op = op1 if pref_data[obj_name][op1] > pref_data[obj_name][op2] else op2
        p =  pref_data[obj_name][op]/(pref_data[obj_name][op1]
                + pref_data[obj_name][op2])
        total_count = pref_data[obj_name][op1] + pref_data[obj_name][op2]
        H1_ps.append(p)
        if total_count - pref_data[obj_name][op] != 0:
            loglh_H1 += np.log(p)*pref_data[obj_name][op] + np.log(1-p)*(total_count - pref_data[obj_name][op])
        else:
            loglh_H1 += np.log(p)*pref_data[obj_name][op]
        loglh_H0 += np.log(0.5)*total_count

    test_statistic = -2*(loglh_H0 - loglh_H1)
    p_value = chi2.sf(test_statistic, len(obj_names))
    return test_statistic, p_value