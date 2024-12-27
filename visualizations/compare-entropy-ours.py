import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pdb
import seaborn as sns

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

##############################################################################
###### START: Experiment Setup #####

# A. Set FILENAME - This is the 'experiments_data' (variable)
# saved in the very end of "peell_cifar.py" or "peell_miniImageNet.py"
# filenames2 = [f'./_{i}thresh_exper_data_2.0_1.pkl' for i in range(49, 54)]
# filenames2 = [f'./_{i}thresh_exper_data_2.0_1.pkl' for i in range(49, 54)]
filenames2 = [f'./accept_modular_exp5_test_c_data_2.0_1.pkl']

# B. Set Experiment type -
# 1 -> Total RA responses on x-axis
# 2 -> Threshold on x-axis
experiment_type = 1

# C. Set Dataset - 
# 1 -> CIFAR100
# 2 -> MiniImageNet
dataset = 2

# D. Find Figures saved as 'vis-entropy-oursc.png' and 'vis-comm-io-better-confidencec.png'

###### END: Experiment Setup #####
##############################################################################








##############################################################################
#                                   Code
##############################################################################

all_data_ = [[load_data(f) for f in filenames2]]
import matplotlib as mpl
sty ='seaborn-v0_8'
mpl.style.use(sty)

# TRUE vs Entropy Sharing Accuracy
fig, ax = plt.subplots()

# Percent fewer total RA responses required to get the same number of correct RA responses at the QA, while using TRUE vs Entropy
fig1, ax1 = plt.subplots() 


for all_data in all_data_:
    pick_all_corr_experiment_agg = []
    pick_all_total_experiment_agg = []
    entropy_corr_experiment_agg = []
    entropy_tot_experiment_agg = []

    pac_agg = []
    eac_agg = []
    min_corr = np.inf

    for data in all_data:
        if dataset == 2:
            # pdb.set_trace()
            (pick_all_corr_experiment, pick_all_total_experiment, _, _, _, _, _, _, _, _, a, b, entropy_corr_experiment, entropy_tot_experiment, _, _, _) = [np.cumsum(np.array(i), axis=0) for i in data]
            # (pick_all_corr_experiment, pick_all_total_experiment, _, _, _, _, _, _, _, _, a, b, entropy_corr_experiment, entropy_tot_experiment, _) = [np.cumsum(np.array(i), axis=0) for i in data]
        else:
            (pick_all_corr_experiment, pick_all_total_experiment, _, _, _, _, _, _, _, _, a, b, entropy_corr_experiment, entropy_tot_experiment, _) = [np.cumsum(np.array(i), axis=0) for i in data]
        pick_all_corr_experiment_agg.append(pick_all_corr_experiment[-1])
        pick_all_total_experiment_agg.append(pick_all_total_experiment[-1])
        entropy_corr_experiment_agg.append(entropy_corr_experiment[-1])
        entropy_tot_experiment_agg.append(entropy_tot_experiment[-1])

        pac = np.interp(np.arange(1, np.max(entropy_corr_experiment[-1]))[::-1], pick_all_corr_experiment[-1][::-1], pick_all_total_experiment[-1][::-1], left=0)
        eac = np.interp(np.arange(1, np.max(entropy_corr_experiment[-1]))[::-1], entropy_corr_experiment[-1][::-1], entropy_tot_experiment[-1][::-1], left=0)
        min_corr = min(min_corr, np.max(entropy_corr_experiment[-1]), np.max(pick_all_corr_experiment[-1]))
        pac_agg.append(pac)
        eac_agg.append(eac)

    i = 0
    e_pac_agg = []
    e_eac_agg = []
    min_corr-=10
    for pac, eac in (zip(pac_agg, eac_agg)):
        pac_agg[i] = pac[-min_corr:]
        eac_agg[i] = eac[-min_corr:]
        e_pac_agg.append(pac[-min_corr:])
        e_eac_agg.append(eac[-min_corr:])
        i+=1

    pac_agg = e_pac_agg
    eac_agg = e_eac_agg

    def mean_confidence_interval(data, confidence=0.95):
        n = len(data)
        mean = np.mean(data, axis=0)
        se = scipy.stats.sem(data, axis=0)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return mean, mean - h, mean + h  # return mean, lower bound, upper bound

    pick_all_corr_mean, pick_all_corr_low, pick_all_corr_high = mean_confidence_interval(pick_all_corr_experiment_agg)
    pick_all_total_mean, pick_all_total_low, pick_all_total_high = mean_confidence_interval(pick_all_total_experiment_agg)
    entropy_corr_mean, entropy_corr_low, entropy_corr_high = mean_confidence_interval(entropy_corr_experiment_agg)
    entropy_tot_mean, entropy_tot_low, entropy_tot_high = mean_confidence_interval(entropy_tot_experiment_agg)

    ax.set_title('Sharing Accuracy vs Total RA Responses', color='C0')
    threshold = 40000

    # For TRUE
    mask = np.logical_and((pick_all_total_mean < threshold), (pick_all_total_mean > 800))
    if dataset == 1:
        ax.plot(pick_all_total_mean[mask], 100*pick_all_corr_mean[mask]/pick_all_total_mean[mask], label='TRUE-CIFAR100',  color='royalblue')
    else:
        ax.plot(pick_all_total_mean[mask], 100*pick_all_corr_mean[mask]/pick_all_total_mean[mask], label='TRUE-MiniImageNet',  color='royalblue', linestyle="-.")
    ax.fill_between(pick_all_total_mean[mask], 100*(pick_all_corr_low/pick_all_total_mean)[mask], 100*(pick_all_corr_high/pick_all_total_mean)[mask], color='royalblue', alpha=0.3)
    
    # For entropy
    mask = np.logical_and((entropy_tot_mean < threshold), (entropy_tot_mean > 800))
    if dataset == 1:
        ax.plot(entropy_tot_mean[mask], 100*entropy_corr_mean[mask]/entropy_tot_mean[mask], label='Entropy-CIFAR100', color='palevioletred')
    else:
        ax.plot(entropy_tot_mean[mask], 100*entropy_corr_mean[mask]/entropy_tot_mean[mask], label='Entropy-MiniImageNet', color='palevioletred', linestyle="-.")
    ax.fill_between(entropy_tot_mean[mask], 100*(entropy_corr_low/entropy_tot_mean)[mask], 100*(entropy_corr_high/entropy_tot_mean)[mask], color='palevioletred', alpha=0.3)
    ax.legend()
    ax.set_xlabel('Total RA responses received at the QA')
    ax.set_ylabel('% Correct in RA responses')


    pac_mean, pac_low, pac_high = mean_confidence_interval(pac_agg)
    eac_mean, eac_low, eac_high = mean_confidence_interval(eac_agg)
    mask = np.array(eac_mean) > 6000
    ax1.set_title('% Reduction in Communication Overhead while using TRUE vs Entropy', color='C0')
    if dataset == 1:
        ax1.plot(np.arange(1, min_corr+1)[::-1][mask][100:], 100*((eac_mean[mask] - pac_mean[mask]) / eac_mean[mask])[100:], color='darkgreen', alpha=0.75, label='CIFAR100')
    else:
        ax1.plot(np.arange(1, min_corr+1)[::-1][mask][100:], 100*((eac_mean[mask] - pac_mean[mask]) / eac_mean[mask])[100:], color='darkgreen', alpha=0.75, linestyle="-.", label="MiniImageNet")
        
    ax1.fill_between(np.arange(1, min_corr+1)[::-1][mask][100:], 100*((eac_low - pac_high) / eac_low)[mask][100:], 100*((eac_high - pac_low) / eac_high)[mask][100:], color='darkgreen', alpha=0.3)
    ax1.set_xlabel('Correct RA Responses received at the QA')
    ax1.set_ylabel(f'% Fewer RA responses sent to the QA')
    ax1.legend()

fig.savefig('vis-entropy-oursc.png')
fig1.savefig('vis-comm-io-better-confidencec.png')