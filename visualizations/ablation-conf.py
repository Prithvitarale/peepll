import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pdb

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# filenames_orig = [f'./_{i}thresh_exper_data_2.0_1.pkl' for i in range(49, 54)]
filenames_orig = [f'./finalruns_cifar100_exper_data_2.0_1.pkl']
filenames_sem = [f'./finalruns_cifar100_expersemonly_data_2.0_1.pkl']
filenames_unc = [f'./finalruns_cifar100_experunconly_data_2.0_1.pkl']
# filenames = [f'./finalruns_cifar100_expms_data_2.0_1.pkl']
# filenames = [f'./finalruns_imgnet_exp2_data_2.0_1.pkl']
# filenames = [f'./finalruns_cifar100_expms3_data_2.0_1.pkl']

# filenames = [f'./_45imgnet_exp_data_2.0_1.pkl']
# filenames = [f'./_imgnet_1reexp_data_2.0_1.pkl']
# filenames = [f'./finalruns_imgnet_expre{i}_data_2.0_1.pkl' for i in [1, 2, 3, 4]]
# filenames.append('./_imgnet_1reexp_data_2.0_1.pkl')
# filenames = [f'./_2re_data_2.1_1.pkl']
# filenames = [f'./try_2.0_1.pkl']


all_data = [load_data(f) for f in filenames_orig]

pick_all_corr_experiment_agg = []
pick_all_total_experiment_agg = []
entropy_corr_experiment_agg = []
entropy_tot_experiment_agg = []

pac_agg = []
eac_agg = []
min_corr = np.inf

for data in all_data:
    # (pick_all_corr_experiment, pick_all_total_experiment, _, _, _, _, _, _, _, _, a, b, entropy_corr_experiment, entropy_tot_experiment, _, _, _) = [np.cumsum(np.array(i), axis=0) for i in data]
    (pick_all_corr_experiment, pick_all_total_experiment, _, _, _, _, _, _, _, _, a, b, entropy_corr_experiment, entropy_tot_experiment, _) = [np.cumsum(np.array(i), axis=0) for i in data]
    # pdb.set_trace()
    pick_all_corr_experiment_agg.append(pick_all_corr_experiment[-1])
    pick_all_total_experiment_agg.append(pick_all_total_experiment[-1])
    entropy_corr_experiment_agg.append(entropy_corr_experiment[-1])
    entropy_tot_experiment_agg.append(entropy_tot_experiment[-1])

    pac = np.interp(np.arange(1, np.max(entropy_corr_experiment[-1]))[::-1], pick_all_corr_experiment[-1][::-1], pick_all_total_experiment[-1][::-1], left=0)
    eac = np.interp(np.arange(1, np.max(entropy_corr_experiment[-1]))[::-1], entropy_corr_experiment[-1][::-1], entropy_tot_experiment[-1][::-1], left=0)
    min_corr = min(min_corr, np.max(entropy_corr_experiment[-1]), np.max(pick_all_corr_experiment[-1]))
    pac_agg.append(pac)
    eac_agg.append(eac)
# pdb.set_trace()

i = 0
e_pac_agg = []
e_eac_agg = []
min_corr-=10
for pac, eac in (zip(pac_agg, eac_agg)):
    pac_agg[i] = pac[-min_corr:]
    eac_agg[i] = eac[-min_corr:]
    print(pac[-min_corr:].shape)
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
# Plotting
import matplotlib as mpl
sty ='seaborn-v0_8'
mpl.style.use(sty)
fig, ax = plt.subplots()
ax.set_title('Sharing Accuracy vs Total RA Responses', color='C0')
threshold = 500000
# threshold = 40000


# For pick_all
mask = np.logical_and((pick_all_total_mean < threshold), (pick_all_total_mean > 800))
ax.plot(pick_all_total_mean[mask], pick_all_corr_mean[mask]/pick_all_total_mean[mask], label='TRUE',  color='royalblue')
ax.fill_between(pick_all_total_mean[mask], (pick_all_corr_low/pick_all_total_mean)[mask], (pick_all_corr_high/pick_all_total_mean)[mask], color='royalblue', alpha=0.3)
mask = np.logical_and((entropy_tot_mean < threshold), (entropy_tot_mean > 800))
ax.plot(entropy_tot_mean[mask], entropy_corr_mean[mask]/entropy_tot_mean[mask], label='Entropy', color='palevioletred')
ax.fill_between(entropy_tot_mean[mask], (entropy_corr_low/entropy_tot_mean)[mask], (entropy_corr_high/entropy_tot_mean)[mask], color='palevioletred', alpha=0.3)



all_data = [load_data(f) for f in filenames_sem]

pick_all_corr_experiment_agg = []
pick_all_total_experiment_agg = []
entropy_corr_experiment_agg = []
entropy_tot_experiment_agg = []

pac_agg = []
eac_agg = []
min_corr = np.inf

for data in all_data:
    # (pick_all_corr_experiment, pick_all_total_experiment, _, _, _, _, _, _, _, _, a, b, entropy_corr_experiment, entropy_tot_experiment, _, _, _) = [np.cumsum(np.array(i), axis=0) for i in data]
    (pick_all_corr_experiment, pick_all_total_experiment, _, _, _, _, _, _, _, _, a, b, entropy_corr_experiment, entropy_tot_experiment, _) = [np.cumsum(np.array(i), axis=0) for i in data]
    # pdb.set_trace()
    pick_all_corr_experiment_agg.append(pick_all_corr_experiment[-1])
    pick_all_total_experiment_agg.append(pick_all_total_experiment[-1])
    entropy_corr_experiment_agg.append(entropy_corr_experiment[-1])
    entropy_tot_experiment_agg.append(entropy_tot_experiment[-1])

    pac = np.interp(np.arange(1, np.max(entropy_corr_experiment[-1]))[::-1], pick_all_corr_experiment[-1][::-1], pick_all_total_experiment[-1][::-1], left=0)
    eac = np.interp(np.arange(1, np.max(entropy_corr_experiment[-1]))[::-1], entropy_corr_experiment[-1][::-1], entropy_tot_experiment[-1][::-1], left=0)
    min_corr = min(min_corr, np.max(entropy_corr_experiment[-1]), np.max(pick_all_corr_experiment[-1]))
    pac_agg.append(pac)
    eac_agg.append(eac)
# pdb.set_trace()

i = 0
e_pac_agg = []
e_eac_agg = []
min_corr-=10
for pac, eac in (zip(pac_agg, eac_agg)):
    pac_agg[i] = pac[-min_corr:]
    eac_agg[i] = eac[-min_corr:]
    print(pac[-min_corr:].shape)
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
# Plotting
# import matplotlib as mpl
# sty ='seaborn-v0_8'
# mpl.style.use(sty)
# fig, ax = plt.subplots()
# ax.set_title('Sharing Accuracy vs Total RA Responses', color='C0')
threshold = 500000
# threshold = 40000


# For pick_all
mask = np.logical_and((pick_all_total_mean < threshold), (pick_all_total_mean > 800))
ax.plot(pick_all_total_mean[mask], pick_all_corr_mean[mask]/pick_all_total_mean[mask], label='TRUE - Dispersion Distance',  color='bisque')
ax.fill_between(pick_all_total_mean[mask], (pick_all_corr_low/pick_all_total_mean)[mask], (pick_all_corr_high/pick_all_total_mean)[mask], color='bisque', alpha=0.3)
mask = np.logical_and((entropy_tot_mean < threshold), (entropy_tot_mean > 800))
ax.plot(entropy_tot_mean[mask], entropy_corr_mean[mask]/entropy_tot_mean[mask], label='Entropy', color='palevioletred')
ax.fill_between(entropy_tot_mean[mask], (entropy_corr_low/entropy_tot_mean)[mask], (entropy_corr_high/entropy_tot_mean)[mask], color='palevioletred', alpha=0.3)



all_data = [load_data(f) for f in filenames_unc]

pick_all_corr_experiment_agg = []
pick_all_total_experiment_agg = []
entropy_corr_experiment_agg = []
entropy_tot_experiment_agg = []

pac_agg = []
eac_agg = []
min_corr = np.inf

for data in all_data:
    # (pick_all_corr_experiment, pick_all_total_experiment, _, _, _, _, _, _, _, _, a, b, entropy_corr_experiment, entropy_tot_experiment, _, _, _) = [np.cumsum(np.array(i), axis=0) for i in data]
    (pick_all_corr_experiment, pick_all_total_experiment, _, _, _, _, _, _, _, _, a, b, entropy_corr_experiment, entropy_tot_experiment, _) = [np.cumsum(np.array(i), axis=0) for i in data]
    # pdb.set_trace()
    pick_all_corr_experiment_agg.append(pick_all_corr_experiment[-1])
    pick_all_total_experiment_agg.append(pick_all_total_experiment[-1])
    entropy_corr_experiment_agg.append(entropy_corr_experiment[-1])
    entropy_tot_experiment_agg.append(entropy_tot_experiment[-1])

    pac = np.interp(np.arange(1, np.max(entropy_corr_experiment[-1]))[::-1], pick_all_corr_experiment[-1][::-1], pick_all_total_experiment[-1][::-1], left=0)
    eac = np.interp(np.arange(1, np.max(entropy_corr_experiment[-1]))[::-1], entropy_corr_experiment[-1][::-1], entropy_tot_experiment[-1][::-1], left=0)
    min_corr = min(min_corr, np.max(entropy_corr_experiment[-1]), np.max(pick_all_corr_experiment[-1]))
    pac_agg.append(pac)
    eac_agg.append(eac)
# pdb.set_trace()

i = 0
e_pac_agg = []
e_eac_agg = []
min_corr-=10
for pac, eac in (zip(pac_agg, eac_agg)):
    pac_agg[i] = pac[-min_corr:]
    eac_agg[i] = eac[-min_corr:]
    print(pac[-min_corr:].shape)
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
# Plotting
# import matplotlib as mpl
# sty ='seaborn-v0_8'
# mpl.style.use(sty)
# fig, ax = plt.subplots()
# ax.set_title('Sharing Accuracy vs Total RA Responses', color='C0')
threshold = 500000
# threshold = 40000

# For pick_all
mask = np.logical_and((pick_all_total_mean < threshold), (pick_all_total_mean > 800))
ax.plot(pick_all_total_mean[mask], pick_all_corr_mean[mask]/pick_all_total_mean[mask], label='TRUE - Semantic Distance',  color='olive')
ax.fill_between(pick_all_total_mean[mask], (pick_all_corr_low/pick_all_total_mean)[mask], (pick_all_corr_high/pick_all_total_mean)[mask], color='royalblue', alpha=0.3)
# mask = np.logical_and((entropy_tot_mean < threshold), (entropy_tot_mean > 800))
# ax.plot(entropy_tot_mean[mask], entropy_corr_mean[mask]/entropy_tot_mean[mask], label='Entropy', color='palevioletred')
# ax.fill_between(entropy_tot_mean[mask], (entropy_corr_low/entropy_tot_mean)[mask], (entropy_corr_high/entropy_tot_mean)[mask], color='palevioletred', alpha=0.3)




ax.legend()
ax.set_xlabel('Total RA Responses')
ax.set_ylabel('Sharing Accuracy: Correct RA Responses / Total RA Responses')



plt.savefig('vis-entropy-ours-abla.png')


# pac_mean, pac_low, pac_high = mean_confidence_interval(pac_agg)
# eac_mean, eac_low, eac_high = mean_confidence_interval(eac_agg)
# mask = np.array(eac_mean) > 1000
# print(pac_mean)
# print(eac_mean)
# print(eac_mean-pac_mean)
# # Plotting
# fig, ax = plt.subplots()
# ax.set_title('Percentage Reduction in Communication Overhead using TRUE, vs Entropy', color='C0')
# ax.plot(np.arange(1, min_corr+1)[::-1][mask], 100*((eac_mean[mask] - pac_mean[mask]) / eac_mean[mask]), color='royalblue')
# ax.fill_between(np.arange(1, min_corr+1)[::-1][mask], 100*((eac_low - pac_high) / eac_low)[mask], 100*((eac_high - pac_low) / eac_high)[mask], color='royalblue', alpha=0.3)
# ax.set_xlabel('Correct RA Responses received at the QA')
# ax.set_ylabel(f'% Lower Communication (Total RA Responses) Required')
# ax.legend()
# plt.savefig('vis-comm-io-better-confidencec.png')


# # print(a[-1][-39])
# # print(b[-1][-39])
# # print((a[-1]/b[-1])[-39])
# # print(np.arange(0.1, 1.0, 0.01)[-39])

# # print(entropy_corr_experiment[-1][-9])
# # print(entropy_tot_experiment[-1][-9])
# # print((entropy_corr_experiment[-1]/entropy_tot_experiment[-1])[-9])
# # print(np.arange(0.1, 1.0, 0.01)[-9])


# print(entropy_corr_mean[-9])
# print(entropy_tot_mean[-9])
# # print(entropy_tot_mean)
# print(entropy_corr_mean[-9]/entropy_tot_mean[-9])
# # print(np.arange(0, 1.0, 0.01)[-9])

# print(pick_all_corr_mean[-26])
# print(pick_all_total_mean[-26])
# print(pick_all_corr_mean[-26]/pick_all_total_mean[-26])
# print(((pick_all_corr_mean[-26]/pick_all_total_mean[-26]) - (entropy_corr_mean[-9]/entropy_tot_mean[-9]))/(entropy_corr_mean[-9]/entropy_tot_mean[-9]))
# # print(np.arange(0, 1.0, 0.01)[-26])

# print(((eac_mean - pac_mean) / eac_mean)[20000])