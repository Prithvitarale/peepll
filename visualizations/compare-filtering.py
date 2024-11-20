import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib as mpl
sty ='seaborn-v0_8'
mpl.style.use(sty)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

###### START: Experiment Setup #####

# Set FILENAME
filenames2 = [f'./_{i}thresh_exper_data_2.0_1.pkl' for i in range(49, 54)]
# filenames2 = [f'./finalruns_imgnet_expre{i}_data_2.0_1.pkl' for i in [1, 2, 3, 4]]

# Set Experiment type -
# 1 for Total RA responses on x-axis
# 2 for Threshold on x-axis
experiment_type = 1

# Find Figure saved as 'vis-compare-filter.png'

###### END: Experiment Setup #####


all_data_ = [[load_data(f) for f in filenames2]]


import pdb
dataset = 0
import matplotlib as mpl
sty ='seaborn-v0_8'
mpl.style.use(sty)
fig, ax = plt.subplots()
ax.set_title('Comparison of Communication Protocols', color='C0')

if experiment_type == 1:
    threshold = 40000
elif experiment_type == 2:
    threshold = 100000000
sm_thresh = 800
for all_data in all_data_:

    pick_all_corr_experiment_agg = []
    pick_all_total_experiment_agg = []
    pick_all_ratio_experiment_agg = []
    pick_all_clash_corr_experiment_agg = []
    pick_all_clash_total_experiment_agg = []
    pick_all_clash_ratio_experiment_agg = []
    majority_corr_experiment_agg = []
    majority_tot_experiment_agg = []
    majority_ratio_experiment_agg = []
    majority_clash_corr_experiment_agg = []
    majority_clash_tot_experiment_agg = []
    majority_clash_ratio_experiment_agg = []
    maj_orig_corr_experiment_agg = []
    maj_orig_tot_experiment_agg = [] 
    maj_orig_ratio_experiment_agg = [] 
    maj_orig_clash_corr_experiment_agg = [] 
    maj_orig_clash_tot_experiment_agg = []
    maj_orig_clash_ratio_experiment_agg = []
    entropy_corr_experiment_agg = []
    entropy_tot_experiment_agg = []
    re = []


    pac_agg = []
    eac_agg = []
    min_corr = np.inf
    for data in all_data:
        (pick_all_corr_experiment,
        pick_all_total_experiment,

        pick_all_clash_corr_experiment,
        pick_all_clash_tot_experiment,

        majority_corr_experiment,
        majority_tot_experiment,

        majority_clash_corr_experiment,
        majority_clash_tot_experiment,

        maj_orig_corr_experiment,
        maj_orig_tot_experiment, 

        maj_orig_clash_corr_experiment, 
        maj_orig_clash_tot_experiment, 

        entropy_corr_experiment, entropy_tot_experiment, _,) = [np.cumsum(np.array(i), axis=0) for i in data]

        print(1)
        pick_all_corr_experiment_agg.append(pick_all_corr_experiment[-1])
        pick_all_total_experiment_agg.append(pick_all_total_experiment[-1])
        pick_all_ratio_experiment_agg.append(pick_all_corr_experiment[-1]/pick_all_total_experiment[-1])
        
        entropy_corr_experiment_agg.append(entropy_corr_experiment[-1])
        entropy_tot_experiment_agg.append(entropy_tot_experiment[-1])
        re.append(entropy_corr_experiment[-1]/entropy_tot_experiment[-1])


        pick_all_clash_corr_experiment_agg.append(pick_all_clash_corr_experiment[-1])
        pick_all_clash_total_experiment_agg.append(pick_all_clash_tot_experiment[-1])
        pick_all_clash_ratio_experiment_agg.append(pick_all_clash_corr_experiment[-1]/pick_all_clash_tot_experiment[-1])

        majority_corr_experiment_agg.append(majority_corr_experiment[-1])
        majority_tot_experiment_agg.append(majority_tot_experiment[-1])
        majority_ratio_experiment_agg.append(majority_corr_experiment[-1]/majority_tot_experiment[-1])


        majority_clash_corr_experiment_agg.append(majority_clash_corr_experiment[-1])
        majority_clash_tot_experiment_agg.append(majority_clash_tot_experiment[-1])
        majority_clash_ratio_experiment_agg.append(majority_clash_corr_experiment[-1]/majority_clash_tot_experiment[-1])
        
        maj_orig_corr_experiment_agg.append(maj_orig_corr_experiment[-1])
        maj_orig_tot_experiment_agg.append(maj_orig_tot_experiment[-1])
        maj_orig_ratio_experiment_agg.append(maj_orig_corr_experiment[-1]/maj_orig_tot_experiment[-1])

        maj_orig_clash_corr_experiment_agg.append(maj_orig_clash_corr_experiment[-1])
        maj_orig_clash_tot_experiment_agg.append(maj_orig_clash_tot_experiment[-1])
        maj_orig_clash_ratio_experiment_agg.append(maj_orig_clash_corr_experiment[-1]/maj_orig_clash_tot_experiment[-1])
        
    def mean_confidence_interval(data, confidence=0.95):
        n = len(data)
        mean = np.mean(data, axis=0)
        se = scipy.stats.sem(data, axis=0)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return mean, mean - h, mean + h  # return mean, lower bound, upper bound

    pick_all_corr_mean, pick_all_corr_low, pick_all_corr_high = mean_confidence_interval(pick_all_corr_experiment_agg)
    pick_all_total_mean, pick_all_total_low, pick_all_total_high = mean_confidence_interval(pick_all_total_experiment_agg)
    rpick_all_total_mean, rpick_all_total_low, rpick_all_total_high = mean_confidence_interval(pick_all_ratio_experiment_agg)

    e_all_corr_mean, e_all_corr_low, e_all_corr_high = mean_confidence_interval(entropy_corr_experiment_agg)
    e_all_total_mean, e_all_total_low, e_all_total_high = mean_confidence_interval(entropy_tot_experiment_agg)
    re_all_total_mean, re_all_total_low, re_all_total_high = mean_confidence_interval(re)


    pccm, pccl, pcch = mean_confidence_interval(pick_all_clash_corr_experiment_agg)
    pctm, pctl, pcth = mean_confidence_interval(pick_all_clash_total_experiment_agg)
    rpctm, rpctl, rpcth = mean_confidence_interval(pick_all_clash_ratio_experiment_agg)

    mcm, mcl, mch = mean_confidence_interval(majority_corr_experiment_agg)
    mtm, mtl, mth = mean_confidence_interval(majority_tot_experiment_agg)
    rmtm, rmtl, rmth = mean_confidence_interval(majority_ratio_experiment_agg)

    mccm, mccl, mcch = mean_confidence_interval(majority_clash_corr_experiment_agg)
    mctm, mctl, mcth = mean_confidence_interval(majority_clash_tot_experiment_agg)
    rmctm, rmctl, rmcth = mean_confidence_interval(majority_clash_ratio_experiment_agg)

    mocm, mocl, moch = mean_confidence_interval(maj_orig_corr_experiment_agg)
    motm, motl, moth = mean_confidence_interval(maj_orig_tot_experiment_agg)
    rmotm, rmotl, rmoth = mean_confidence_interval(maj_orig_ratio_experiment_agg)

    moccm, moccl, mocch = mean_confidence_interval(maj_orig_clash_corr_experiment_agg)
    moctm, moctl, mocth = mean_confidence_interval(maj_orig_clash_tot_experiment_agg)
    rmoctm, rmoctl, rmocth = mean_confidence_interval(maj_orig_clash_ratio_experiment_agg)

    aa_thresholds = np.arange(0, 1.0, 0.01)
    mask = np.logical_and((pick_all_total_mean < threshold), (pick_all_total_mean > sm_thresh))
    if experiment_type == 1:
        ax.plot(pick_all_total_mean[mask], 100*rpick_all_total_mean[mask], label='TRUE (MiniImageNet)', color='royalblue', linestyle="--")
        ax.fill_between(pick_all_total_mean[mask], 100*(rpick_all_total_low)[mask], 100*(rpick_all_total_high)[mask], color='royalblue', alpha=0.3)
    elif experiment_type == 2:
        ax.plot(aa_thresholds[mask], 100*rpick_all_total_mean[mask], label='TRUE', color='royalblue', linestyle="-")
        ax.fill_between(aa_thresholds[mask], 100*(rpick_all_total_low)[mask], 100*(rpick_all_total_high)[mask], color='royalblue', alpha=0.3)   

    mask = np.logical_and((pctm < threshold), pctm > sm_thresh)
    if experiment_type == 1:
        ax.plot(pctm[mask] , 100*(rpctm)[mask], label='TRUE+ICF', color='royalblue', linestyle='dashdot')
        ax.fill_between(pctm[mask], 100*rpctl[mask], 100*rpcth[mask], color='royalblue', alpha=0.3)
    elif experiment_type == 2:
        ax.plot(aa_thresholds[mask] , 100*(rpctm)[mask], label='TRUE+ICF', color='royalblue', linestyle='dashdot')
        ax.fill_between(aa_thresholds[mask], 100*rpctl[mask], 100*rpcth[mask], color='royalblue', alpha=0.3)

    mask = np.logical_and((mtm < threshold), mtm > sm_thresh)
    if experiment_type == 1:
        ax.plot(mtm[mask] , 100*(rmtm)[mask], label='Majority', color='dodgerblue', alpha=0.6)
        ax.fill_between(mtm[mask],100*rmtl[mask], 100*rmth[mask], color='dodgerblue', alpha=0.3)
    elif experiment_type == 2:
        ax.plot(aa_thresholds[mask] , 100*(rmtm)[mask], label='Majority', color='dodgerblue', alpha=0.6, linestyle='-')
        ax.fill_between(aa_thresholds[mask],100*rmtl[mask], 100*rmth[mask], color='dodgerblue', alpha=0.3)

    mask = np.logical_and((mctm < threshold), mctm > sm_thresh)
    if experiment_type == 1:
        ax.plot(mctm[mask] , 100*(rmctm)[mask], label='Majority+ICF', color='dodgerblue', linestyle='dashdot', alpha=0.55)
        ax.fill_between(mctm[mask], 100*rmctl[mask], 100*rmcth[mask], color='dodgerblue', alpha=0.25)
    elif experiment_type == 2:
        ax.plot(aa_thresholds[mask] , 100*(rmctm)[mask], label='Majority+ICF', color='dodgerblue', linestyle='dashdot', alpha=0.55)
        ax.fill_between(aa_thresholds[mask], 100*rmctl[mask], 100*rmcth[mask], color='dodgerblue', alpha=0.25)

    mask = np.logical_and((motm < threshold), motm > sm_thresh)
    if experiment_type == 1:
        ax.plot(motm[mask] , 100*(rmotm)[mask], label='MCG', color='darkgreen', alpha=0.7)
        ax.fill_between(motm[mask], 100*rmotl[mask], 100*rmoth[mask], color='darkgreen', alpha=0.3)
    elif experiment_type == 2:
        ax.plot(aa_thresholds[mask] , 100*(rmotm)[mask], label='MCG', color='darkgreen', alpha=0.7)
        ax.fill_between(aa_thresholds[mask], 100*rmotl[mask], 100*rmoth[mask], color='darkgreen', alpha=0.3)

    mask = np.logical_and((moctm < threshold), moctm > sm_thresh)
    if experiment_type == 1:
        ax.plot(moctm[mask] , 100*(moccm/moctm)[mask], label='MCG+ICF (REFINE)', color='darkgreen', linestyle='dashdot', alpha=0.7)
        ax.fill_between(moctm[mask], 100*rmoctl[mask], 100*rmocth[mask], color='darkgreen', alpha=0.3)
    elif experiment_type == 2:
        ax.plot(aa_thresholds[mask] , 100*(rmoctm)[mask], label='TRUE + REFINE', color='darkgreen', linestyle='dashdot', alpha=0.7)
        ax.fill_between(aa_thresholds[mask], 100*rmoctl[mask], 100*rmocth[mask], color='darkgreen', alpha=0.3)

plt.legend()
if experiment_type == 1:
    ax.set_xlabel('Total RA Responses Received at the QA')
elif experiment_type == 2:
    ax.set_xlabel('TRUE Threshold for RAs to send responses to QA')
ax.set_ylabel(r'% Correct in $\mathbf{Accepted}$ RA Responses')
plt.savefig('vis-compare-filter.png', bbox_inches='tight')