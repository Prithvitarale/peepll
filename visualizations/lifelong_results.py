import pickle
import numpy as np
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


###### START: Experiment Setup #####

# Set FILENAMES for each experiment type on lines

# Set Experiment Number
# 1 -> QA Accuracy on Pretrained data
# 2 -> QA Accuracy on Untrained data (Future Tasks)
# 3 -> QA Accuracy on Past data (tasks learned until that timestep)
# 4 -> QA Accuracy on Complete Test Set
experiment_number = 1

# Find Figure saved as 'performances.png'

###### END: Experiment Setup #####


# TRUE
with open('./finalruns_cifar100_reducing_meorig_2.0_1.pkl', 'rb') as f:
    info = pickle.load(f)

# TRUE + ICF
with open('./finalruns_imgnet_vanclash_s1_2.0_1.pkl', 'rb') as f:
    info_e = pickle.load(f)

# MCG
with open('./finalruns_imgnet_conf_s1_2.0_1.pkl', 'rb') as f:
    info_mo = pickle.load(f)

# MCG+ICF
with open('./finalruns_imgnet_confclash_s1_2.0_1.pkl', 'rb') as f:
    info_mo_e = pickle.load(f)

# Majority
with open('./finalruns_imgnet_maj_s1_2.0_1.pkl', 'rb') as f:
    info_m = pickle.load(f)

# Majority + ICF
with open('./finalruns_imgnet_majclash_s1_2.0_1.pkl', 'rb') as f:
    info_m_e = pickle.load(f)

# Entropy
with open('./finalruns_imgnet_ent_s1_3_2.0_1.pkl', 'rb') as f:
    info_ent = pickle.load(f)

# Supervised + Replay (ER)
with open('./finalruns_imgnet_single_s1_2.0_1.pkl', 'rb') as f:
    info_single = pickle.load(f)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

maj_intervals = []
sty ='seaborn-v0_8'
sns.set_style("whitegrid", {
    'axes.facecolor': 'whitesmoke',  # Lighter background color for axes
    'figure.facecolor': 'white'      # Lighter background color for figure
})

fig, ax = plt.subplots()

for x in range(0, len(info_single[experiment_number][0]), 50):
    plt.axvline(x=x, color='cornflowerblue', linestyle='--', alpha=1)
    if x+3 < len(info_single[0]):  # To ensure the label is within the plot range
        plt.text(x + 25, plt.ylim()[0], str(x // 49 + 1), ha='center', va='bottom', color='royalblue', alpha=1)


if experiment_number == 1:
    ax.set_title(f'QA\'s Local Performance on its Pretrained classes', color='C0')
elif experiment_number == 2:
    ax.set_title(f'QA\'s Local Performance on Future Tasks', color='C0')
elif experiment_number == 3:
    ax.set_title(f'QA\'s Local Performance on Tasks introduced so far', color='C0')
elif experiment_number == 4:
    ax.set_title(f'QA\'s Local Performance on Complete Test Set', color='C0')

ax.set_xlabel('Task ID / Time')
ax.set_ylabel('Accuracy (%)')


window_size = 30
ax.plot(range(len(info[experiment_number][0]) - window_size + 1), moving_average(100*np.array(info[experiment_number][0]), window_size), color='y', label='TRUE', linestyle='dashdot')
ax.plot(range(len(info_e[experiment_number][0]) - window_size + 1), moving_average(100*np.array(info_e[experiment_number][0]), window_size), color='y', label='TRUE+ICF', linestyle='-')
ax.plot(range(len(info_m[experiment_number][0]) - window_size  + 1), moving_average(100*np.array(info_m[experiment_number][0]), window_size), color='darkgreen', linestyle="dashdot", label='Majority')
ax.plot(range(len(info_m_e[experiment_number][0]) - window_size  + 1), moving_average(100*np.array(info_m_e[experiment_number][0]), window_size), color='darkgreen', label='Majority+ICF')
ax.plot(range(len(info_mo[experiment_number][0]) - window_size + 1), moving_average(100*np.array(info_mo[experiment_number][0]), window_size), color='cornflowerblue', label='MCG', linestyle="dashdot")
ax.plot(range(len(info_mo_e[experiment_number][0]) - window_size + 1), moving_average(100*np.array(info_mo_e[experiment_number][0]), window_size), color='cornflowerblue', label='MCG+ICF')
ax.plot(range(len(info_ent[experiment_number][0]) - window_size + 1), moving_average(100*np.array(info_ent[experiment_number][0]), window_size), color='dimgray', label='Entropy')
ax.plot(range(len(info_single[experiment_number][0]) - window_size + 1), moving_average(100*np.array(info_single[experiment_number][0]), window_size), color='tomato', label='Single-Agent')

ax.set_ybound(0)
ax.legend(loc=2)
plt.savefig('performances.png')