import numpy as np
import pickle
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {
    'axes.facecolor': 'whitesmoke',  # Lighter background color for axes
    'figure.facecolor': 'white'      # Lighter background color for figure
})

##############################################################################
###### START: Experiment Setup #####

# A. Set FILENAME - This is the 'experiments_data' (variable)
# saved in the very end of "peell_cifar.py" or "peell_miniImageNet.py"
with open('./finalruns_imgnet_reducingqc3_data_2.0_1.pkl', 'rb') as f:
    info_p = pickle.load(f)

# B. Find 'confidence_vs_time.png', 'commfreq_vs_time.png', 'conf_comm_bar.png'

###### END: Experiment Setup #####    
##############################################################################


def moving_average(data, window_size):
    """Compute the moving average using a sliding window."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

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

entropy_corr_experiment, 
entropy_tot_experiment, 

total_correct_experiment,
final_total_calls_made,
avg_qa_conf,) = info_p

################################################################################################################################
################################################################################################################################

################################################################################################################################
#                            Plots QA's TRUE Confidence in Queries of Each Task over Time
################################################################################################################################
plt.figure()
fig = plt.figure()
import matplotlib as mpl
sty ='seaborn-v0_8'
mpl.style.use(sty)
sns.set_style("whitegrid", {
    'axes.facecolor': 'whitesmoke',  # Lighter background color for axes
    'figure.facecolor': 'white'      # Lighter background color for figure
})
fig, ax = plt.subplots()
avg_vals = []
episodes = 25
small_episodes = 20
diff_conf_vals = []
only_diff = []
window_size = 3

for task_id in range(19):
    avg_val = 0
    if task_id < 18:
        for episode in range(episodes):
            avg_val += avg_qa_conf[task_id*episodes + episode]
            avg_vals.append(avg_val/(episode+1))
        aqc = moving_average(avg_qa_conf[episodes*task_id: episodes*task_id+episodes], window_size)
        ax.plot(range(episodes*task_id, episodes*task_id+episodes-window_size+1), aqc, color='darkgreen', alpha=0.7, linewidth=1.75)
        diff_conf_vals.append(100*(aqc[-1] - aqc[0])/aqc[0])
        only_diff.append(aqc[-1] - aqc[0])
    else:
        for episode in range(small_episodes):
            avg_val += avg_qa_conf[task_id*episodes + episode]
            avg_vals.append(avg_val/(episode+1))
        aqc = moving_average(avg_qa_conf[episodes*task_id: episodes*task_id+small_episodes], window_size)
        ax.plot(range(episodes*task_id, episodes*task_id+small_episodes-window_size+1), aqc, color='darkgreen', alpha=0.7, linewidth=1.75, label="QA Confidence")
        diff_conf_vals.append(100*(aqc[-1] - aqc[0])/aqc[0])
        only_diff.append(aqc[-1] - aqc[0])

for x in range(0, len(avg_vals)+6, episodes):
    if x == len(avg_vals)+5:
        ax.axvline(x=x-5, color='cornflowerblue', linestyle='--', alpha=0.8)
    else:
        ax.axvline(x=x, color='cornflowerblue', linestyle='--', alpha=0.8)
    if x+20 <= len(avg_vals):  # To ensure the label is within the plot range
        ax.text(x + 12, plt.ylim()[0], str(x // 25 + 1), ha='center', va='bottom', color='royalblue', alpha=0.8)
ax.set_facecolor('white')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')
ax.legend()
ax.set_title("QA Gains Confidence Over Time in Answering Queries of Each Task")
ax.set_xlabel("Task ID/Time")
ax.set_ylabel("QA's Confidence at Task")
plt.savefig('confidence_vs_time.png')

################################################################################################################################
################################################################################################################################

################################################################################################################################
#                            Plots QA's Communication Frequency for Queries of Each Task over Time
################################################################################################################################

fig = plt.figure()
import matplotlib as mpl
sty ='seaborn-v0_8'
mpl.style.use(sty)
sns.set_style("whitegrid", {
    'axes.facecolor': 'whitesmoke',  # Lighter background color for axes
    'figure.facecolor': 'white'      # Lighter background color for figure
})
fig, ax = plt.subplots()
avg_vals = []
diff_comm_vals = []
for task_id in range(19):
    avg_val = 0
    if task_id < 18:
        for episode in range(episodes):
            avg_val += final_total_calls_made[task_id*episodes + episode]*20
            avg_vals.append(avg_val/(episode+1))
        avvals = moving_average(final_total_calls_made[episodes*task_id: episodes*task_id+episodes], window_size)/40
        ax.plot(range(episodes*task_id, episodes*task_id+episodes-window_size+1), avvals, color='brown', alpha=0.7, linewidth=1.75)
        diff_comm_vals.append((avvals[0] - avvals[-1]))
    else:
        for episode in range(small_episodes):
            avg_val += final_total_calls_made[task_id*episodes + episode]*20
            avg_vals.append(avg_val/(episode+1))
        avvals = moving_average(final_total_calls_made[episodes*task_id: episodes*task_id+small_episodes], window_size)/40
        ax.plot(range(episodes*task_id, episodes*task_id+small_episodes-window_size+1), avvals, color='brown', alpha=0.7, linewidth=1.75, label="Communication Likelihood per Query")
        diff_comm_vals.append((avvals[0] - avvals[-1]))
      
for x in range(0, len(avg_vals)+6, episodes):
    if x == len(avg_vals)+5:
        ax.axvline(x=x-5, color='cornflowerblue', linestyle='--', alpha=0.8)
    else:
        ax.axvline(x=x, color='cornflowerblue', linestyle='--', alpha=0.8)
    if x+20 <= len(avg_vals):  # To ensure the label is within the plot range
        ax.text(x + 12, plt.ylim()[0], str(x // 25 + 1), ha='center', va='bottom', color='royalblue', alpha=0.8)
ax.set_facecolor('white')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)
ax.xaxis.grid(True, which='major', color='blue', linestyle='--', linewidth=0.5)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

ax.set_title("Fall in Likelihood of QA Initiating Communication after Sufficiently Learning Task")
ax.set_xlabel("Task ID/Time")
ax.set_ylabel("Likelihood of QA Initiating Communication")
plt.legend()
plt.savefig('commfreq_vs_time.png')

################################################################################################################################
################################################################################################################################

################################################################################################################################
#         Plots QA's Increase in TRUE Confidence and Decrease in Communication Likelihood at the end of Each Task
################################################################################################################################

import scipy
from scipy.stats import pearsonr
diff_comm_vals = [-x for x in diff_comm_vals]
only_diff = [x for x in only_diff]
diff_conf_vals = [x for x in diff_conf_vals]
correlation, p_value = pearsonr(diff_comm_vals, only_diff)

print("Correlation coefficient:", correlation)
print("P-value:", p_value)
tasks = np.arange(1, 20)

fig, ax1 = plt.subplots(figsize=(10, 7))
ax1.set_xlabel('Task ID')
ax1.set_xticks(np.arange(1, 20, 1))
ax1.bar(tasks, diff_comm_vals, color='brown', alpha=0.75)
ax1.set_ylabel('Change in Likelihood of Communication for Task', fontsize=13, color='firebrick')
ax1.tick_params(axis='y', labelcolor='brown')
ax1.set_ylim([-0.4, 0.4])
# ax1.set_ylim([-0.15, 0.15])

ax2 = ax1.twinx()
ax2.bar(tasks, only_diff, color='darkgreen', alpha=0.7)
ax2.set_ylabel('Change in Confidence at Task', fontsize=13, color='darkgreen')
ax2.set_ylim([-0.4, 0.4])
# ax2.set_ylim([-0.15, 0.15])

ax2.tick_params(axis='y', labelcolor='darkgreen')
correlation_shortened = round(correlation, 2)
ax1.text(0.85, 1.2, r'$\mathbf{Correlation: }$' + f'{-correlation_shortened}', transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=dict(facecolor='plum', alpha=0.35))

ax1.grid(True)
ax2.grid(False)
ax1.set_axisbelow(True)
ax1.xaxis.grid(True, which='major', color='blue', linestyle='--', linewidth=0.3)
for spine in ax1.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')
plt.title('QA Gains Confidence and Reduces Communication After Sufficiently Learning Tasks')
plt.savefig('conf_comm_bar.png')
################################################################################################################################
################################################################################################################################