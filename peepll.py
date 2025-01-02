import argparse
import pdb
import random
random.seed(1)
from pprint import pprint
import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from tensorflow.keras.layers import (
    Dense, Conv2D, BatchNormalization,
    AveragePooling2D, Input, Flatten
)
np.seterr(all='raise')
tf.print(tf. __version__)

from data import load_dataset
from tensorflow.keras.callbacks import Callback, EarlyStopping
from utils.data_utils import datafilter, datafilter_perclass
from utils.peepll_utils import update_memory, identify_from_memory_scalable_forkl, call_response_agent
from utils.model_utils import build_model_data, pretrain_model, init_memories
from utils.communication import communication
from utils.shared_variables_class import SharedVariables


####### NOTE: GPU usage currently not supported. ####### 

parser = argparse.ArgumentParser(description="Run experiments with given dataset and settings.")
    
# Define the arguments
parser.add_argument('--dataset', type=str, required=True, help="Dataset (M for MiniImageNet, C for CIFAR100)")
parser.add_argument('--experiment_id', type=float, required=True, help="ID of the experiment (2.2 for Communication and LL, 5 for TRUE and Filter results)")
parser.add_argument('--learning_type', type=int, required=True, help="Type of learning (2 for Supervised, 3 for Communicative)")

# Parse the arguments
args = parser.parse_args()

# Access the arguments
print(f"Dataset: {args.dataset}")
print(f"Experiment ID: {args.experiment_id}")
print(f"Learning Type: {args.learning_type}")

if args.dataset == "M":
    dataset = "MiniImgNet"
else:
    dataset = "cifar100"

#####################
#   Experiment IDs
# Use 2.2 or 5.0 for TMLR paper results
# 2.2 -> PEEPLL, to choose communication protocol see line x [todo]
# 5.0 -> TRUE and Selective Response Filter results
# learning_type = 2 --> Supervised with replay
# learning_type = 3 --> Communicative (1 model communicates with 19 others)
#####################

experiment_version = args.experiment_id
input_learning_type = args.learning_type

pdb.set_trace()

#####################
#   Main
#####################

print("\n\n\n")
print("="*100)
print("Starting Up")
print("="*100)



#####################
#   default configs




config = argparse.Namespace()

config.num_classes = 100

if dataset == "MiniImgNet":
    config.input_shape = (84, 84, 3)
else:
    config.input_shape = (32, 32, 3)
config.batch_size = 256
config.epochs = 50

# config.epochs_inner_loop = 10 # for offline experiments
config.epochs_inner_loop = 1

# Use batch_size_inner_loop=1000 for TRUE experiments
# for faster execution, as batch size does not matter
# config.batch_size_inner_loop = 1000
config.batch_size_inner_loop = 20

config.knowledge_sharing_len = -1
config.confidence_threshold = .5
config.task_learning_order = "random"
config.num_memorize = 50
config.only_communicate_below_confidence = 0.7
config.detached_confidence_evaluator = True
config.compare_communication = False
config.model0_subset_train = 0
config.load_i = None
config.graph_itr = 50
config.num_agents = 20
config.class_wise = True
config.dataset = 'keras.datasets.cifar100.load_data()'
config.all_classes = list(range(100))
config.learning_type = None
config.model_classes = []

if experiment_version == 2.2 or experiment_version == 5.0:
    config.learning_type = input_learning_type
    config.figure_i = "2.0_1"
    config.task_learning_order = "ordered"
    config.num_classes = 100

    # IMPORTANT: [todo] Set Confidence Threshold separately at line x, do not use below
    config.only_communicate_below_confidence = 0.67
    config.model_classes = [\
    [3, 5, 44, 23, 68, 79, 92, 86],\

    [1, 11, 21, 31, 41, 10],\
    [2, 12, 22, 32, 42, 33, 65, 30, 31, 49, 50, 51, 52, 53, 61, 62, 63, 64, 65],\
    [3, 13, 23, 33, 43, 9, 18, 25, 49, 50, 51, 52, 53, 61, 62, 63, 64, 65],\
    [4, 14, 24, 34, 44, 10, 19, 26, 30, 37, 38, 39, 40, 71, 72, 73, 74, 75, 76],\
    [5, 15, 25, 35, 45, 73, 18, 30, 37, 38, 39, 40, 71, 72, 73, 74, 75, 76],\
    [6, 16, 26, 36, 46, 1, 5, 19, 24, 37, 38, 39, 40, 71, 72, 73, 74, 75, 76],\
    [7, 17, 27, 37, 47, 30, 4, 24, 29, 35, 54, 55, 56, 57, 58, 59, 60],\
    [8, 18, 28, 38, 48, 30, 83, 9, 31, 35, 54, 55, 56, 57, 58, 59, 60, 77, 78, 79, 80, 81, 82],\
    [9, 19, 29, 39, 49, 1, 14, 23, 31, 34, 54, 55, 56, 57, 58, 59, 60, 77, 78, 79, 80, 81, 82],\
    [10, 20, 30, 40, 0, 92, 8, 16, 29, 36, 66, 67, 68, 69, 70, 77, 78, 79, 80, 81, 82],\
    [51, 61, 71, 81, 91, 2, 13, 23, 26, 34, 66, 67, 68, 69, 70, 97, 98, 99],\
    [52, 62, 72, 82, 92, 5, 16, 22, 32, 36, 66, 67, 68, 69, 70, 97, 98, 99],\
    [53, 63, 73, 83, 93, 3, 6, 12, 15, 34, 87, 88, 89, 90, 91, 92, 50, 51, 52],\
    [54, 64, 74, 84, 94, 90, 8, 20, 27, 34, 87, 88, 89, 90, 91, 92, 53, 61, 62, 63,],\
    [55, 65, 75, 85, 95, 3, 13, 21, 29, 32, 41, 42, 43, 44, 87, 88, 89, 90, 91, 92],\
    [56, 66, 76, 86, 96, 4, 12, 15, 32, 33, 41, 42, 43, 44, 97, 98, 99, 64, 65],\
    [57, 67, 77, 87, 97, 6, 15, 20, 27, 28, 41, 42, 43, 44, 83, 84, 85, 86, 93, 94, 95, 96],\
    [58, 68, 78, 88, 98, 2, 50, 11, 14, 33, 45, 46, 47, 48, 83, 84, 85, 86, 93, 94, 95, 96],\
    [59, 69, 79, 89, 99, 7, 11, 28, 32, 36, 45, 46, 47, 48, 83, 84, 85, 86],\
    [60, 70, 80, 90, 50, 39, 7, 27, 33, 45, 46, 47, 48, 93, 94, 95, 96, 17, 25, 49]]
    config.num_agents = 20
    config.all_classes = list(range(100))
    config.true_results = 0
    if experiment_version == 5.0:
        config.true_results = 1
        config.batch_size_inner_loop = 1000

if config.load_i is None:
    config.load_i = config.figure_i

#####################
#   info
#####################

pprint(vars(config), depth=2)

print("\n\n\n")
print("="*100)
print("Building Models and Dataset")
print("="*100)

if dataset == "MiniImgNet":
    (x_train, y_train), (x_test, y_test) = load_dataset.load_data()
else:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()


print("x_train shape", x_train.shape)
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train_categorical = keras.utils.to_categorical(y_train, config.num_classes)
y_test_categorical = keras.utils.to_categorical(y_test, config.num_classes)


x_train_list, y_train_list, x_trainqa_list, y_trainqa_list = [], [], [], []
for i in range(100):
    x_train_c, y_train_c, _, _ = datafilter(x_train, y_train, x_test, y_test, [i])
    x_train_list.extend(x_train_c[:300])
    y_train_list.extend(y_train_c[:300])
    x_trainqa_list.extend(x_train_c[300:])
    y_trainqa_list.extend(y_train_c[300:])

# Data For pretraining
(x_train, y_train) = (np.array(x_train_list), np.array(y_train_list))
# Data For lifelong 
(x_train_forqa, y_train_forqa) = (np.array(x_trainqa_list), np.array(y_trainqa_list))

permutation = np.random.permutation(x_train.shape[0])
x_train = x_train[permutation]
y_train = y_train[permutation]

if config.class_wise:
    if config.model0_subset_train > 0:
        x_train_baseline,y_train_baseline = datafilter_perclass(
            x_train, y_train,
            config.model0_subset_train,
            config.model_classes[0], False
        )
        x_test_baseline,y_test_baseline = datafilter_perclass(
            x_test, y_test,
            config.model0_subset_train,
            config.model_classes[0], False
        )
    else:
        x_train_baseline,y_train_baseline,x_test_baseline,y_test_baseline = datafilter(x_train, y_train, x_test, y_test, config.model_classes[0])
else:
    print("Wrong Experiment type (check config.class_wise)")

x_trainN, y_trainN, x_testN, y_testN, modelN = [], [], [], [], []
y_train_precateg_N, y_test_precateg_N = [], []

for i in range(config.num_agents):
    x_tr, y_tr, x_te, y_te, m = build_model_data(
    f"{i+1}", x_train, y_train, x_test, y_test,
    config.model_classes[i+1], config.num_classes,
    config.input_shape, config.confidence_threshold
    )

    x_trainN.append(x_tr)
    y_trainN.append(y_tr)
    x_testN.append(x_te)
    y_testN.append(y_te)
    modelN.append(m)


print("\n\n\n")
print("="*100)
print("Loading Pre-trained or Pre-training Models")
print("="*100)

loaded_weights = False

input_shape_b = (None,) + config.input_shape

try:
    loaded = -1
    for mi in modelN:
        mi.build(input_shape_b)
    # -models
    for i, mi in enumerate(modelN):
        # mi.load_weights(f'/work/pi_hava_umass_edu/prithvi/save_cifar100/save_cifar100/{i+1}_newdatakl5_{config.figure_i}.h5')
        mi.load_weights(f'/work/pi_hava_umass_edu/prithvi/save_mimgnet/{i+1}_imgnetpre6_{config.figure_i}.h5')
        print(f'loaded model {i}')
        loaded = i
    loaded_weights = True
    print('loaded weights')
    # print('Agent performances on their training set')
    # for i, mi in enumerate(modelN):
    #     print(f'Agent {i}: {mi.evaluate(x_trainN[i], y_trainN[i], verbose=0)}')

except Exception as e:
    print("Failed to load Model due to exception, ", e)

if not loaded_weights:
    print("\n")
    for i, mi in enumerate(modelN):
        if i <= loaded:
            print('not training the trained')
            continue
        print(f'For model {i+1}: {x_trainN[i].shape, y_trainN[i].shape, np.unique(y_trainN[i])}')
        pretrain_model(
        f"{i+1}", mi,
        config.batch_size, config.epochs,
        x_trainN[i], y_trainN[i],
        x_testN[i], y_testN[i],
        weights = None
        )
        mi.save_weights(f'/work/pi_hava_umass_edu/prithvi/save_cifar100/{i+1}_40ssagentserr_err100a_{config.figure_i}.h5')
model_list = modelN
total_num_models = len(model_list)

print("\n\n\n")
print("="*100)
print("Add Memories")
print("="*100)

if config.num_memorize is None:
    print("memories disabled")
else:
    for i, mi in enumerate(modelN):
        init_memories(mi, x_trainN[i], y_trainN[i], config.num_memorize)

(x_train, y_train) = (x_train_forqa, y_train_forqa)
print("x_train shape", x_train.shape)
x_train_ = x_train
y_train_ = y_train
y_train_categorical = keras.utils.to_categorical(y_train, config.num_classes)
y_test_categorical = keras.utils.to_categorical(y_test, config.num_classes)
print("x_train shape", x_train.shape)


print("\n\n\n")
print("="*100)
print("Building Secondary Models for Communication")
print("="*100)


task_wise_x, task_wise_y = [], []
task_wise_x_test, task_wise_y_test = [], []

print("\n\n\n")
print("="*100)
print("Building untrained datasets")
print("="*100)
if config.class_wise:
    modelN_untrained_classes = []
    for mi_classes in config.model_classes:
        modelN_untrained_classes.append([c for c in config.all_classes if c not in mi_classes])
    
    tasks = len(modelN_untrained_classes[1])//5
    total_samples = 0
    print(f'tasks: {tasks}')
    for i in range(tasks):
        x_train_untrained1, y_train_untrained1, x_test_untrained, y_test_untrained = datafilter(x_train, y_train, x_test, y_test, modelN_untrained_classes[1][5*i:5*(i+1)])
        task_wise_x.append(x_train_untrained1)
        task_wise_y.append(y_train_untrained1)
        task_wise_x_test.append(x_test_untrained)
        task_wise_y_test.append(y_test_untrained)
        total_samples += x_train_untrained1.shape[0]
    x_train_untrained1, y_train_untrained1, x_test_untrained, y_test_untrained = datafilter(x_train, y_train, x_test, y_test, modelN_untrained_classes[1][5*tasks:])
    total_samples += x_train_untrained1.shape[0]
    task_wise_x.append(x_train_untrained1)
    task_wise_y.append(y_train_untrained1)
    task_wise_x_test.append(x_test_untrained)
    task_wise_y_test.append(y_test_untrained)


    _, _, x_test_untrained1, y_test_untrained1 = datafilter(x_train, y_train, x_test, y_test, modelN_untrained_classes[1])
    y_test_untrained1_ = y_test_untrained1
    y_test_untrained1 = keras.utils.to_categorical(y_test_untrained1, config.num_classes)

else:
    x_train_untrained1, y_train_untrained1, x_test_untrained1, y_test_untrained1 =\
    np.array(tf.concat(x_trainN[1:], 0)),\
    np.array(tf.concat(y_train_precateg_N[1:], 0)),\
    np.array(tf.concat(x_testN[1:],0)),\
    np.array(tf.concat(y_test_precateg_N[1:], 0))
    y_train_untrained1_ = y_train_untrained1
    y_test_untrained1_ = y_test_untrained1
    y_train_untrained1 = keras.utils.to_categorical(y_train_untrained1, config.num_classes)
    y_test_untrained1 = keras.utils.to_categorical(y_test_untrained1, config.num_classes)

print("\n\n\n")
print("="*100)
print("Preparing for Online Learning and Inference")
print("="*100)


shared_variables = SharedVariables(total_num_models)


print("\n\n\n")
print("="*100)
print("Training")
print("="*100)

prev_classes = None
for task_id, x_train in enumerate(task_wise_x):
    training_len = len(x_train)
    y_train = task_wise_y[task_id]
    ind_list = list(range(training_len))
    np.random.shuffle(ind_list)
    x_train = x_train[ind_list]
    y_train = y_train[ind_list]
    training_len//=config.batch_size_inner_loop
    new_classes = np.unique(y_train[0:1000])
    print("\n\n")
    print(f"New Task information: Task ID {task_id}, Introducing classes {new_classes}")

    # epoch here
    for task_epoch in range(1):
        print(f"Epoch {task_epoch}")
        for i in range(training_len):
            print("\n\n")
            print("[Sharing] {}/{}".format(i,training_len))
            
            i0 = i * config.batch_size_inner_loop
            i1 = (i+1) * config.batch_size_inner_loop

            if task_id != 18:
                print(
                        "train", x_train[i0:i1].shape, y_train[i0:i1].shape, np.unique(y_train[i0:i1]),
                "\n", "pretrained", x_testN[0].shape, y_testN[0].shape, np.unique(y_testN[0].argmax(1)),
                "\n", "untrained", np.concatenate(task_wise_x_test[task_id+1:]).shape, keras.utils.to_categorical(np.concatenate(task_wise_y_test[task_id+1:]), config.num_classes).shape, np.unique(np.concatenate(task_wise_y_test[task_id+1:])),
                "\n", "past", np.concatenate(task_wise_x_test[:task_id+1]).shape, np.concatenate(task_wise_y_test[:task_id+1]).shape, np.unique(np.concatenate(task_wise_y_test[:task_id+1])),
                "\n", "generative", x_trainN[0].shape, y_trainN[0].shape, np.unique(y_trainN[0].argmax(1)),
                "\n", "all", x_test.shape, y_test_categorical.shape, np.unique(y_test)
                )
                if i == training_len-1:
                    print_stats = 1
                else:
                    # change to 1 for always recording and printing,
                    # use 0 when trying to reduce runtime
                    print_stats = 0  

                communication(
                    0, x_train[i0:i1], y_train[i0:i1],
                    modelN,
                    x_testN[0], y_testN[0],
                    np.concatenate(task_wise_x_test[task_id+1:]), keras.utils.to_categorical(np.concatenate(task_wise_y_test[task_id+1:]), config.num_classes),
                    np.concatenate(task_wise_x_test[:task_id+1]), np.concatenate(task_wise_y_test[:task_id+1]), x_trainN[0], y_trainN[0], x_test, y_test_categorical,
                    print_stats, config, shared_variables, config.batch_size_inner_loop, config.epochs_inner_loop)
            else:
                print(
                        "train", x_train[i0:i1].shape, y_train[i0:i1].shape, np.unique(y_train[i0:i1]),
                "\n", "pretrained", x_testN[0].shape, y_testN[0].shape, np.unique(y_testN[0].argmax(1)),
                "\n", "past", np.concatenate(task_wise_x_test[:task_id+1]).shape, np.concatenate(task_wise_y_test[:task_id+1]).shape, np.unique(np.concatenate(task_wise_y_test[:task_id+1])),
                "\n", "generative", x_trainN[0].shape, y_trainN[0].shape, np.unique(y_trainN[0].argmax(1)),
                "\n", "all", x_test.shape, y_test_categorical.shape, np.unique(y_test)
                )
                communication(
                    0, x_train[i0:i1], y_train[i0:i1],
                    modelN,
                    x_testN[0], y_testN[0],
                    None, None,
                    np.concatenate(task_wise_x_test[:task_id+1]), np.concatenate(task_wise_y_test[:task_id+1]), x_trainN[0], y_trainN[0], x_test, y_test_categorical,
                    print_stats, config, shared_variables, config.batch_size_inner_loop, config.epochs_inner_loop)


            shared_variables.communication_calls.append(i)
            print(f"error_wrong_shared: {shared_variables.error_wrong_shared}")
            print(f"total shared: {shared_variables.count_correctly_shared_possible}")
            graphing_data = (
                shared_variables.communication_calls,
                shared_variables.acc_pretrained, shared_variables.acc_untrained,
                shared_variables.acc_past, shared_variables.acc_complete,
                shared_variables.count_shared, shared_variables.count_shared_possible,
                shared_variables.error_correctly_shared,
                shared_variables.false_negatives,
                shared_variables.count_correctly_shared_possible,
                shared_variables.error_wrong_shared,
            )
            with open(f"./accept_modular_cc_m2_test_{config.figure_i}.pkl", 'wb') as f:
                pickle.dump(graphing_data, f)

            experiments_data = (
                shared_variables.pick_all_corr_experiment,
                shared_variables.pick_all_total_experiment,

                shared_variables.pick_all_clash_corr_experiment,
                shared_variables.pick_all_clash_tot_experiment,

                shared_variables.majority_corr_experiment,
                shared_variables.majority_tot_experiment,

                shared_variables.majority_clash_corr_experiment,
                shared_variables.majority_clash_tot_experiment,

                shared_variables.maj_orig_corr_experiment,
                shared_variables.maj_orig_tot_experiment, 

                shared_variables.maj_orig_clash_corr_experiment, 
                shared_variables.maj_orig_clash_tot_experiment, 

                shared_variables.entropy_corr_experiment, 
                shared_variables.entropy_tot_experiment, 

                shared_variables.total_correct_experiment,
                shared_variables.final_total_calls_made,
                shared_variables.avg_qa_conf,
            )
            print("seed=1, cc")
            with open(f"./accept_modular_cc_test_m2_data_{config.figure_i}.pkl", 'wb') as f:
                pickle.dump(experiments_data, f)