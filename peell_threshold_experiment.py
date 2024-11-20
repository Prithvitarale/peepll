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
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import tensorflow_addons as tfa

from tensorflow.keras.layers import (
    Dense, Conv2D, BatchNormalization,
    AveragePooling2D, Input, Flatten
)
from tensorflow.keras.callbacks import Callback, EarlyStopping
np.seterr(all='raise')
tf.print(tf. __version__)

##########################################
#########  START: Functions ##############
##########################################

# Function Description: Early Stopping Mechanism
# for Pre-training models. Not used in Lifelong Learning.
class TargetValAccuracyCallback(Callback):
    def __init__(self, min_val_accuracy, max_val_accuracy):
        super(TargetValAccuracyCallback, self).__init__()
        self.min_val_accuracy = 0.75
        self.max_val_accuracy = 0.85

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        train_accuracy = logs.get('accuracy')
        okay = False
        if train_accuracy is not None:
            if train_accuracy > 0.85:
                okay = True
                print("enough training")
                if val_accuracy is not None:
                    if val_accuracy >= self.max_val_accuracy and okay:
                        print(f"\nReached {self.max_val_accuracy*100}% val accuracy, stopping training ({train_accuracy}).")
                        self.model.stop_training = True
                    elif val_accuracy >= self.min_val_accuracy:
                        print(f"\nReached {self.min_val_accuracy*100}% val accuracy, considering stopping training ({train_accuracy}).")
                        # If within range, consider stopping if further improvement is minimal
                        self.model.stop_training = True

# Function Description: Samples a latent code, given z_mean and z_variance
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Function Description: Defines our VAE model without reconstruction.
class VAE(keras.Model):
    def __init__(self, input_shape, latent_dim=128, encoder=None, decoder=None, confidence_threshold=0.5, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.confidence_threshold = confidence_threshold

        if encoder is None:
            inputs = keras.Input(shape=(input_shape))
            vgg = VGG16(weights='imagenet', include_top=False)
            inputs = preprocess_input(inputs)
            x = vgg(inputs)
            x = tf.reshape(x, [-1, x.shape[1]*x.shape[2]*x.shape[3]])

            z_mean = keras.layers.Dense(latent_dim)(x)
            z_log_var = keras.layers.Dense(latent_dim)(x)
            z = Sampling()([z_mean, z_log_var])
            self.encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
        else:
            self.encoder = encoder

        if decoder is None:
            inputs = keras.Input(shape=(latent_dim,))
            x = keras.layers.Dropout(0)(inputs)

            o = x
            o = keras.layers.Dense(256, activation='relu')(o)
            o = keras.layers.Dense(128)(o)
            o = keras.layers.Dense(config.num_classes, activation="softmax", name = "classifier")(o)
            self.decoder = keras.Model(inputs, [o], name="decoder")
        else:
            self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.train_mode = 0
        self.memorized_data = []
        self.memorized_targets = []
        self.memorized_prediction = []

    @property
    def metrics(self):
        m = super().metrics
        return list(m) + [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, data, training=True):
        z_mean, z_log_var, z = self.encoder(data, training=training)
        classification = self.decoder(z, training=training)
        return classification, z, z_mean, z_log_var
    
    def train_step(self, data):
        data, confidences, targets = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)
            a = keras.losses.categorical_crossentropy(targets, reconstruction)
            reconstruction_loss = tf.reduce_mean(a)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            if self.train_mode == 0:
                total_loss = reconstruction_loss + tf.reduce_mean(0.007*(kl_loss))
            else:
                raise RuntimeError("invalid train_mode", self.train_mode)
    

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        # self.generative_loss_tracker.update_state(generative_loss)
        self.compiled_metrics.update_state(targets, reconstruction)

        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "loss": self.total_loss_tracker.result(),
            "prediction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            # "generative_loss": self.generative_loss_tracker.result(),
        })
        return results

    def test_step(self, data):

        data, targets = data

        _, _, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)

        reconstruction_loss = tf.reduce_mean(
            keras.losses.categorical_crossentropy(targets, reconstruction)
        )

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.compiled_metrics.update_state(targets, reconstruction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        })
        return results

# ================================#
####  Util Method ####
# ================================#

# Function Description:
# Takes data and classes, and returns data of those classes alone.
def datafilter(X_train, Y_train, X_test, Y_test, classes):
    if np.shape(Y_train)[1] == 1:
        train_mask = np.isin(Y_train[:, 0], classes)
        test_mask = np.isin(Y_test[:, 0], classes)
    elif np.shape(Y_train)[1] > 1:
        train_mask = np.isin(Y_train.argmax(1), classes)
        test_mask = np.isin(Y_test.argmax(1), classes)
    else:
        train_mask = np.isin(Y_train, classes)
        test_mask = np.isin(Y_test, classes)
    print("datafilter classes", classes)
    X_train, Y_train = X_train[train_mask], Y_train[train_mask]
    X_test, Y_test = X_test[test_mask], Y_test[test_mask]
    return (X_train, Y_train, X_test, Y_test)


# Function Description:
# Takes data and classes, and returns data of those classes in blocks of maximum length num_per_class.
def datafilter_perclass(X_train, Y_train, num_per_class, classes, negate=False):

    if np.shape(Y_train)[1] == 1:
        Y_train_1d = Y_train[:, 0]
    elif np.shape(Y_train)[1] > 1:
        Y_train_1d = Y_train.argmax(1)
    else:
        Y_train_1d = Y_train
    train_mask = np.isin(Y_train_1d, classes)

    if negate:
        train_mask = ~train_mask

    Y_train_1d = Y_train_1d[train_mask]
    X_train, Y_train = X_train[train_mask], Y_train[train_mask]

    X_new = []
    Y_new = []
    avail_classes, avail_idxs = np.unique(Y_train_1d, return_inverse=True)
    print("Adding samples for classes", avail_classes)
    for idx, cls in enumerate(avail_classes):
        cls_mask = avail_idxs == idx
        X_new.extend(X_train[cls_mask][:num_per_class])
        Y_new.extend(Y_train[cls_mask][:num_per_class])

    return np.concatenate([X_new], 0), np.concatenate([Y_new], 0)

# ================================================================================================#
                            ### DYNAMIC MEMORY UPDATE METHOD ###
# Function Description:
# Adds (x, y, confidence) to model's memory.
# There is a limit of num_per_class for how many unique x's cant be stored for each y.
# If the limit is reached, it replaces (x, y, confidence) with least confidence with new
# (x', y', confidence'), if confidence' > confidence.
# ================================================================================================#
def update_memory(model, x, y, confidences, num_per_class):
    print(f'Maximum memory per class: {num_per_class}')
    if np.shape(y)[1] == 1:
        y = y[:, 0]
    elif np.shape(y)[1] > 1:
        y = y.argmax(1)
    else:
        print('No need for this')
    task_classes, incoming_counts = np.unique(y, return_counts=True)
    experienced_classes, old_counts = np.unique(model.memorized_targets, return_counts=True)
    for i, c in enumerate(task_classes):
        index = np.where(experienced_classes == c)[0]
        if len(index) == 0:
            c_count = 0
        else:
            c_count = int(old_counts[index[0]])
        if c_count < num_per_class:
            take = int(incoming_counts[i])
            if c_count + take > num_per_class:
                take = num_per_class - c_count
            take_indices = np.where(y == c)[0]
            x_take = x[take_indices][:take]
            y_take = y[take_indices][:take]
            confidences_take = confidences[take_indices][:take]

            model.memorized_data.extend(x_take)
            y_take = np.array([np.array([x]) for x in y_take])
            model.memorized_targets.extend(y_take)
            model.memorized_prediction = np.concatenate([model.memorized_prediction, confidences_take])

            print(f'Class {c}: {c_count + take}')
        else:
            # To remove Dynamic Memory Update, continue
            # continue
            print(f'Class {c}: {c_count}')
            take_indices = np.where(y == c)[0]
            x_take = x[take_indices]
            y_take = y[take_indices]
            confidences_take = confidences[take_indices]
            mtargets = np.concatenate([model.memorized_targets], 0)
            if np.shape(mtargets)[1] == 1:
                mt = mtargets[:, 0]
            elif np.shape(mtargets)[1] > 1:
                mt = mtargets.argmax(1)

            imt = np.where(mt == c)[0]
            y_take_ = np.array([np.array([x]) for x in y_take])

            for i_c, conf in enumerate(confidences_take):
                c_in_memory = np.where(model.memorized_prediction[imt] < conf)[0]
                if len(c_in_memory) > 0:
                    min_c_in_memory = np.argmin(model.memorized_prediction[imt][c_in_memory])
                    model.memorized_data[imt[c_in_memory[min_c_in_memory]]] = x_take[i_c]
                    model.memorized_targets[imt[c_in_memory[min_c_in_memory]]]= y_take_[i_c]
                    old_conf = model.memorized_prediction[imt[c_in_memory[min_c_in_memory]]]
                    model.memorized_prediction[imt[c_in_memory[min_c_in_memory]]]= conf
                    print(f"Replaced memory of class {c} of confidence old:{old_conf} with new:{conf}")

# ================================================================================================#
                            ####  TRUE: Confidence Method ####
# Funtion Description:
# Evaluates the TRUE confidence metric, given the model, the queries, and its Entropy in prediction.
# ConfN -> Entropy of its prediction per query
# ================================================================================================#
def identify_from_memory_scalable_forkl(model, queries, p_batch, confN, ms_scores):
    score_for_input = []
    confidences = []
    pred = np.argmax(p_batch, axis=1)

    memory = np.concatenate([model.memorized_data], 0)
    memory = tf.squeeze(memory, axis=4)
    mem_preds, memory_z, z_means, z_logs = model(memory) # 900, 128
    q_pred, queries_z, q_mean, q_log = model(queries) # 128,128

    all_scores = []
    for q, query in enumerate(queries):
        matching_indices = np.isin(model.memorized_targets, pred[q])
        matching_indices = np.where(matching_indices)[0].flatten()
        if len(matching_indices) == 0:
            all_scores.append(0)
        else:
            matching_z_means = np.array(z_means)[matching_indices, :] # 50, 128
            matching_z_logs = np.array(z_logs)[matching_indices, :] # 50, 128
            matching_z = np.array(memory_z)[matching_indices, :] # 50, 128
            query_var = np.exp(q_log[q, :]) # 128
            memory_var = np.exp(matching_z_logs) # 50, 128
            mean_memory_var = np.mean(memory_var, axis=0)
            mean_memory_mean = np.mean(matching_z_means, axis=0)

            entropy = confN[q][0]
            ms = ms_scores[q][0]

            uncertainty_score = np.exp(-0.01*np.sum(np.abs(mean_memory_var-query_var)))
            uncertainty_score = (uncertainty_score - 0.4)/(0.5)
            uncertainty_score = max(uncertainty_score, 0)
            uncertainty_score = min(uncertainty_score, 1)


            mean_distance = np.exp(-0.1*tf.norm(mean_memory_mean-q_mean[q, :]))
            mean_distance = (mean_distance - 0.4)/(0.4)
            mean_distance = max(mean_distance, 0)
            mean_distance = min(mean_distance, 1)
            all_scores.append((mean_distance + uncertainty_score + entropy)/3)
    return tf.convert_to_tensor(all_scores, dtype=tf.float32)

# Function Description:
# Calls each Response Agent, records their Predictions, Confidences, and Classes they think they know.
def call_response_agent(model, query_packet):
    xt, q_c = query_packet
    p, z2, _, _ = model.predict(xt)
    predN_append = np.expand_dims(p, 1)
    conf_recon_p_append = tf.argmax(p, 1)
    ms_append = np.expand_dims(tf.math.reduce_max(p, 1), 1)
    entropy = -np.sum(p * np.log2(p + 1e-10), axis=1)
    entropy_mapped = np.exp(-entropy)
    confN_append = np.expand_dims(entropy_mapped, 1)
    unique_elements, counts = np.unique(np.concatenate([model.memorized_targets], 0), return_counts=True)
    filtered_elements = unique_elements[counts > 40]
    classes_they_think_they_know_append = filtered_elements
    print(f'Model {i} thinks they know {filtered_elements}')
    confidences = identify_from_memory_scalable_forkl(model, xt, p, confN_append, ms_append)
    # Need to fix - can be reduced to 3
    return predN_append, conf_recon_p_append, ms_append, confN_append, classes_they_think_they_know_append, confidences

# ================================#
####  Main Communication Method ####
# ================================#
def communication(
        model_number, xtrain, ytrain,
        models,
        x_pretrained, y_pretrained,
        x_untrained, y_untrained,
        x_past, y_past, x_generative, y_generative, x_complete, y_complete,
        print_stats, batch_size=1, epochs=1
    ):

    print(f"Epochs running for: {epochs}")

    ###### SETUP ######
    # Freeze the RA models to ensure that weights stay the same
    for modeli in models:
        modeli.trainable = False
    orig_model = models[0]
    orig_model.trainable = True


    if config.num_memorize is not None:
        generative_X = np.concatenate([orig_model.memorized_data], 0)
        y_generative = np.concatenate([orig_model.memorized_targets], 0)
        y_generative = keras.utils.to_categorical(y_generative, config.num_classes)
        print("Memory Information")
        print("num memorized before", len(orig_model.memorized_data), np.unique(orig_model.memorized_targets))

    # Supervised with Replay
    if learning_config.learning_setup == 2:
        print("[Model] Supervised/Single-Agent")
        pred_orig = np.squeeze(np.expand_dims(keras.utils.to_categorical(ytrain,config.num_classes),1),1)
        new_train_set_X = np.append(generative_X, xtrain, axis = 0)
        new_train_set_Y = np.append(y_generative, pred_orig, axis = 0)

        confidences_supervised = np.ones(len(new_train_set_X), dtype=np.float32) # supervised: we have complete confidence in these samples
        dataset = tf.data.Dataset.from_tensor_slices((new_train_set_X, confidences_supervised, new_train_set_Y))
        dataset = dataset.shuffle(len(new_train_set_X))
        dataset = dataset.batch(20)

        orig_model.fit(dataset, batch_size=20, epochs=epochs, validation_split=0, verbose=2)
        update_memory(orig_model, xtrain, pred_orig, confidences_supervised, config.num_memorize)

    # ================================================================#
    # ================================================================#
    #                   PEEPLL with communication
    # If you need to change communication protocol, change (a, b, c) on line 857:
    # learn_x, learn_y = a, b
    # learn_confidences = c
    # 
    # (a, b, c) Dictionary:
    # Entropy                   -> shared_x_c, shared_y_c, confidences_x_c
    # TRUE                      -> shared_x, shared_y, confidences_x
    # TRUE + ICF                -> shared_x_e, shared_y_e, confidences_x_e
    # TRUE + Majority           -> shared_x_m, shared_y_m, confidences_x_m
    # TRUE + Majority + ICF     -> shared_x_m_e, shared_y_m_e, confidences_x_m_e
    # TRUE + MCG                -> shared_x_m_orig, shared_y_m_orig, confidences_x_m_orig
    # TRUE + MCG + ICF (REFINE) -> shared_x_m_e_orig, shared_y_m_e_orig, confidences_x_m_e_orig
    # 
    # Set the threshold on line 493, aa_threshold:
    # Entropy                   -> 
    # TRUE                      -> 
    # TRUE + ICF                -> 
    # TRUE + Majority           -> 
    # TRUE + Majority + ICF     -> 
    # TRUE + MCG                -> 
    # TRUE + MCG + ICF (REFINE) ->
    # ================================================================#
    # ================================================================#

    elif learning_config.learning_setup == 3:
        print("[Model] 1 agent communicating with others:")
        xt = tf.squeeze(xtrain, axis=4)
        pred_orig, _, _, _ = orig_model.predict(xtrain)
        predN = [np.expand_dims(keras.utils.to_categorical(pred_orig.argmax(1),config.num_classes),1)]
        entropy = -np.sum(pred_orig * np.log2(pred_orig + 1e-10), axis=1)
        entropy_mapped = np.exp(-entropy)
        confN = [np.expand_dims(entropy_mapped, 1)]
        ms = [np.expand_dims(tf.math.reduce_max(pred_orig, 1), 1)]
        q_c = identify_from_memory_scalable_forkl(models[0], xt, np.squeeze(predN[0], axis=1), confN[0], ms[0])

        conf_recon_p = [tf.argmax(pred_orig, 1)]
        classes_they_think_they_know = []
        all_confidences = []
        
        ######## Call Response Agents ########
        query_packet = (xt, q_c)
        print(f'xt.shape: {xt.shape}')
        for i, model in enumerate(models[1:]):

            # call and get responses
            print(f'Model {i}')
            predN_append, conf_recon_p_append, ms_append, confN_append, classes_they_think_they_know_append, confidences = call_response_agent(model, query_packet)
            # collect information
            predN.append(predN_append)
            conf_recon_p.append(conf_recon_p_append)
            ms.append(ms_append)
            confN.append(confN_append)
            classes_they_think_they_know.append(classes_they_think_they_know_append)
            all_confidences.append(confidences)


        ############################## Prepare information for ICF ########################################
        flattened_p = np.array(conf_recon_p).flatten() # RA predictions
        # For each prediction, make list of agents that know this prediction
        all_knowing_agent_ids = [[] for _ in range(len(flattened_p))]
        all_knowing_agent_agree_info = [[] for _ in range(len(flattened_p))]
        for idx, array in enumerate(classes_they_think_they_know):  # Go through all agents
            know_agent_preds = np.array(conf_recon_p)[idx+1]

            # curr_mask is a boolean array of the same length as flattened_p, where:
            # True (or 1) means the element in flattened_p is found in array (classes known by `idx' agent).
            curr_mask = np.isin(flattened_p, array).astype(int)
            curr_mask *= (idx+1) # agent id
            for i in range(len(flattened_p)):
                if curr_mask[i] > 0:
                    # If idx agent knows this class, add this agent ID (idx) to all_knowing_agent_ids
                    all_knowing_agent_ids[i].append(idx+1)
                    # Does this agent agree with this response?
                    all_knowing_agent_agree_info[i].append(flattened_p[i] == know_agent_preds[i%config.batch_size_inner_loop])


        pick_model_only_c = np.squeeze(np.array(confN).T, 0)
        # pick_model_only_c = np.squeeze(np.array(ms).T, 0)
        print('no reducing with conf')
        preds = np.squeeze(np.expand_dims(keras.utils.to_categorical(np.array(conf_recon_p).T, config.num_classes),2), 2)
        pick_model = np.vstack((q_c, np.array(all_confidences))).T
        pick_model_all = pick_model
        pick_model = pick_model.argmax(1)
        
        query_agent_confidences = q_c
        print(f"query_agent_confidences shape: {query_agent_confidences.shape}")
        avg_qa_conf.append(np.mean(query_agent_confidences))
        aa_thresholds, entropy_threshold = [0.585], [1]
        if config.true_results == 1:
            aa_thresholds, entropy_threshold = np.arange(0, 1.0, 0.01).tolist(), np.arange(0, 1.0, 0.01).tolist()
        
        print(f"All Confidence thresholds: {aa_thresholds}")
        print(f"All Entropy thresholds: {entropy_threshold}")
        pick_all_corr = []
        pick_all_tot = []
        pick_all_clash_corr = []
        pick_all_clash_tot = []
        maj_corr = []
        maj_tot = []
        maj_corr_clash = []
        maj_tot_clash = []
        maj_corr_orig = []
        maj_tot_orig = []
        maj_corr_orig_clash = []
        maj_tot_orig_clash = []
        entropy_corr = []
        entropy_tot = []
        for threshold_aa, threshold_entropy in zip(aa_thresholds, entropy_threshold):
            print(f'Confidence Threshold: {threshold_aa}')
            print(f'Entropy Threshold: {threshold_entropy}')
            democratic_mask = np.logical_and(pick_model_all > threshold_aa, pick_model_only_c > 0)
            
            democratic_mask_only_c = pick_model_only_c > threshold_entropy

            # ============================================================ #
            ########################### ICF ###########################
            # ============================================================ #
            all_knowing_agent_confs = [[] for _ in range(len(flattened_p))]
            # For each response, go through agents that claim to know that predicted label
            for i, agent_ids in enumerate(all_knowing_agent_ids):
                total = True
                for j, agent_id in enumerate(agent_ids):
                    # Do all agents that claim to know this predicted label agree on the prediction?
                    total  = total and (pick_model_all.T[agent_id][i%config.batch_size_inner_loop] > threshold_aa and all_knowing_agent_agree_info[i][j])
                # If yes, total = True
                all_knowing_agent_confs[i] = total
            # ICF MASK = For each response, do all agents that know the predicted label, agree on the prediction?
            post_clash_pick_mask_all = np.array(all_knowing_agent_confs)
            post_clash_pick_mask_all = post_clash_pick_mask_all.reshape(np.array(conf_recon_p).shape)
            # Confident Responses + ICF Mask
            democratic_mask_experimental = democratic_mask & post_clash_pick_mask_all.T

            # ============================================================= #
            ######################### END: ICF #########################
            # ============================================================= #



            # ============================================================= #
            ################### Filter out Learnable Data ###################
            # ============================================================= #
            shared_x = []
            shared_x_e = []
            shared_x_m = []
            shared_x_m_e = []
            shared_x_m_orig = []
            shared_x_m_e_orig = []
            shared_x_c = []

            confidences_x = []
            confidences_x_e = []
            confidences_x_m = []
            confidences_x_m_e = []
            confidences_x_m_orig = []
            confidences_x_m_e_orig = []
            confidences_x_c = []

            
            real_y = []
            real_y_e = []
            real_y_m = []
            real_y_m_e = []
            real_y_m_orig = []
            real_y_m_e_orig = []
            real_y_c = []

            shared_y = []
            shared_y_e = []
            shared_y_m = []
            shared_y_m_e = []
            shared_y_m_orig = []
            shared_y_m_e_orig = []
            shared_y_c = []

            #### False Negatives masks ####
            fn_pick_all = np.ones(democratic_mask.shape)
            fn_pick_all_clash = np.ones(democratic_mask.shape)
            fn_maj = np.ones(democratic_mask.shape)
            fn_maj_clash = np.ones(democratic_mask.shape)
            fn_maj_orig = np.ones(democratic_mask.shape)
            fn_maj_orig_clash = np.ones(democratic_mask.shape)
            fn_entropy = np.ones(democratic_mask.shape)



            #### Communication Stats ####
            rejected_comm_because_qa_confident = 0
            responses_not_received_because_qa_confident = 0
            total_calls_made = len(democratic_mask)
            for i, row in enumerate(democratic_mask):
                #####################################################################################
                # REDUCING COMMUNICATION: In order to reduce communication,
                # as the agent's confidence increases, uncomment the following. This also serves as 
                # a natural Early-Stopping Mechanism, which helps performance.

                # if query_agent_confidences[i] > learning_config.only_communicate_below_confidence:
                #     rejected_comm_because_qa_confident += 1
                #     responses_not_received_because_qa_confident += len(np.where(row == True))
                #     total_calls_made-=1
                #     continue
                #####################################################################################
                
                # Collect Responses Based on Entropy Scores
                shared_y_c.append(preds[i, democratic_mask_only_c[i]])
                confidences_x_c.append(pick_model_only_c[i, democratic_mask_only_c[i]])
                fn_entropy[i, democratic_mask_only_c[i]] = 0
                repeats_c = shared_y_c[-1].shape[0]
                repeated_y_c = np.repeat(ytrain[i], repeats=repeats_c, axis=0)
                real_y_c.append(repeated_y_c)
                x = xtrain[i]
                repeated_x_c = np.repeat(x[np.newaxis, :, :, :, :], repeats=repeats_c, axis=0)
                shared_x_c.append(repeated_x_c)

                # Collect Responses Based on TRUE Scores
                responses = np.argmax(preds[i, row], axis=-1)
                confidences = pick_model_all[i, row]
                fn_pick_all[i, row] = 0

                # Collect Responses Based on TRUE + ICF
                responses_e = np.argmax(preds[i, democratic_mask_experimental[i]], axis=-1)
                confidences_e = pick_model_all[i, democratic_mask_experimental[i]]
                fn_pick_all_clash[i, democratic_mask_experimental[i]] = 0

                if len(confidences) > 0:
                    confidence_dict = {}
                    for response, confidence in zip(responses, confidences):
                        if response not in confidence_dict:
                            confidence_dict[response] = [confidence]
                        else:
                            confidence_dict[response].append(confidence)

                    # Mean-Democratic
                    # Group agents by response, take most confident group
                    average_confidences_orig = {response: np.mean(confidences) for response, confidences in confidence_dict.items()}
                    max_confidence_response_orig = max(average_confidences_orig, key=average_confidences_orig.get)
                    
                    # Majority-Democratic
                    # Group agents by response, take most populated group
                    average_confidences = {response: len(confidences) for response, confidences in confidence_dict.items()}
                    max_confidence_response = max(average_confidences, key=average_confidences.get)

                    indices = np.where(responses == max_confidence_response)
                    indices_orig = np.where(responses == max_confidence_response_orig)

                    fn_maj[i, row][indices] = 0
                    fn_maj_orig[i, row][indices_orig] = 0


                    if len(confidences_e) > 0:
                        confidence_dict_e = {}
                        for response_e, confidence_e in zip(responses_e, confidences_e):
                            if response_e not in confidence_dict_e:
                                confidence_dict_e[response_e] = [confidence_e]
                            else:
                                confidence_dict_e[response_e].append(confidence_e)

                        
                        # Majority-Democratic
                        # Group agents by response, take most populated group
                        # Eliminate any eliminated by ICF
                        average_confidences_e = {response_e: len(confidences_e) for response_e, confidences_e in confidence_dict_e.items()}
                        max_confidence_response_e = max(average_confidences_e, key=average_confidences_e.get)
                        indices_e = np.where(responses_e == max_confidence_response_e)

                        # Mean-Democratic
                        # Group agents by response, take most confident group
                        # Eliminate any eliminated by ICF
                        average_confidences_e_orig = {response_e: np.mean(confidences_e) for response_e, confidences_e in confidence_dict_e.items()}
                        max_confidence_response_e_orig = max(average_confidences_e_orig, key=average_confidences_e_orig.get)
                        indices_e_orig = np.where(responses_e == max_confidence_response_e_orig)
                    
                        fn_maj_clash[i, democratic_mask_experimental[i]][indices_e] = 0
                        fn_maj_orig_clash[i, democratic_mask_experimental[i]][indices_e_orig] = 0


                    ### Setup ###
                    # Collect responses by each communication protocol
                    shared_y.append(preds[i, row])
                    confidences_x.append(pick_model_all[i, row])

                    shared_y_m.append(preds[i, row][indices])
                    confidences_x_m.append(pick_model_all[i, row][indices])

                    shared_y_m_orig.append(preds[i, row][indices_orig])
                    confidences_x_m_orig.append(pick_model_all[i, row][indices_orig])


                    if len(confidences_e) > 0:
                        shared_y_e.append(preds[i, democratic_mask_experimental[i]])
                        confidences_x_e.append(pick_model_all[i, democratic_mask_experimental[i]])

                        shared_y_m_e.append(preds[i, democratic_mask_experimental[i]][indices_e])
                        confidences_x_m_e.append(pick_model_all[i, democratic_mask_experimental[i]][indices_e])

                        shared_y_m_e_orig.append(preds[i, democratic_mask_experimental[i]][indices_e_orig])
                        confidences_x_m_e_orig.append(pick_model_all[i, democratic_mask_experimental[i]][indices_e_orig])
                    
                    repeats = shared_y[-1].shape[0]
                    repeats_m = shared_y_m[-1].shape[0]
                    repeats_m_orig = shared_y_m_orig[-1].shape[0]
                    if len(confidences_e) > 0:
                        repeats_e = shared_y_e[-1].shape[0]
                        repeats_m_e = shared_y_m_e[-1].shape[0]
                        repeats_m_e_orig = shared_y_m_e_orig[-1].shape[0]
                    repeated_y = np.repeat(ytrain[i], repeats=repeats, axis=0)
                    repeated_y_m = np.repeat(ytrain[i], repeats=repeats_m, axis=0)
                    repeated_y_m_orig = np.repeat(ytrain[i], repeats=repeats_m_orig, axis=0)
                    if len(confidences_e) > 0:
                        repeated_y_e = np.repeat(ytrain[i], repeats=repeats_e, axis=0)
                        repeated_y_m_e = np.repeat(ytrain[i], repeats=repeats_m_e, axis=0)
                        repeated_y_m_e_orig = np.repeat(ytrain[i], repeats=repeats_m_e_orig, axis=0)
                    real_y.append(repeated_y)
                    real_y_m.append(repeated_y_m)
                    real_y_m_orig.append(repeated_y_m_orig)
                    if len(confidences_e) > 0:
                        real_y_e.append(repeated_y_e)
                        real_y_m_e.append(repeated_y_m_e)
                        real_y_m_e_orig.append(repeated_y_m_e_orig)

                    x = xtrain[i]
                    repeated_x = np.repeat(x[np.newaxis, :, :, :, :], repeats=repeats, axis=0)
                    shared_x.append(repeated_x)

                    repeated_x_m = np.repeat(x[np.newaxis, :, :, :, :], repeats=repeats_m, axis=0)
                    shared_x_m.append(repeated_x_m)

                    repeated_x_m_orig = np.repeat(x[np.newaxis, :, :, :, :], repeats=repeats_m_orig, axis=0)
                    shared_x_m_orig.append(repeated_x_m_orig)

                    if len(confidences_e) > 0:
                        repeated_x_e = np.repeat(x[np.newaxis, :, :, :, :], repeats=repeats_e, axis=0)
                        shared_x_e.append(repeated_x_e)

                        repeated_x_m_e = np.repeat(x[np.newaxis, :, :, :, :], repeats=repeats_m_e, axis=0)
                        shared_x_m_e.append(repeated_x_m_e)

                        repeated_x_m_e_orig = np.repeat(x[np.newaxis, :, :, :, :], repeats=repeats_m_e_orig, axis=0)
                        shared_x_m_e_orig.append(repeated_x_m_e_orig)

            if len(shared_y) > 0:
                shared_x = np.concatenate(shared_x, axis=0)
                shared_y = np.concatenate(shared_y, axis=0)
                real_y = np.concatenate(real_y, axis=0)
                pick_all_corr.append((shared_y.argmax(1) == real_y).sum())
                pick_all_tot.append(shared_y.shape[0])
                confidences_x = np.concatenate(confidences_x)
            else:
                pick_all_corr.append(0)
                pick_all_tot.append(0)

            if len(shared_y_e) > 0:
                shared_x_e = np.concatenate(shared_x_e, axis=0)
                shared_y_e = np.concatenate(shared_y_e, axis=0)
                real_y_e = np.concatenate(real_y_e, axis=0)
                pick_all_clash_corr.append((shared_y_e.argmax(1) == real_y_e).sum())
                pick_all_clash_tot.append(shared_y_e.shape[0])
                confidences_x_e = np.concatenate(confidences_x_e)
            else:
                pick_all_clash_corr.append(0)
                pick_all_clash_tot.append(0)

            if len(shared_y_m) > 0:
                shared_x_m = np.concatenate(shared_x_m, axis=0)
                shared_y_m = np.concatenate(shared_y_m, axis=0)
                real_y_m = np.concatenate(real_y_m, axis=0)
                maj_corr.append((shared_y_m.argmax(1) == real_y_m).sum())
                maj_tot.append(shared_y_m.shape[0])
                confidences_x_m = np.concatenate(confidences_x_m)
            else:
                maj_corr.append(0)
                maj_tot.append(0)

            if len(shared_y_m_orig) > 0:
                shared_x_m_orig = np.concatenate(shared_x_m_orig, axis=0)
                shared_y_m_orig = np.concatenate(shared_y_m_orig, axis=0)
                real_y_m_orig = np.concatenate(real_y_m_orig, axis=0)
                maj_corr_orig.append((shared_y_m_orig.argmax(1) == real_y_m_orig).sum())
                maj_tot_orig.append(shared_y_m_orig.shape[0])
                confidences_x_m_orig = np.concatenate(confidences_x_m_orig)
            else:
                maj_corr_orig.append(0)
                maj_tot_orig.append(0)

            if len(shared_y_m_e) > 0:
                shared_x_m_e = np.concatenate(shared_x_m_e, axis=0)
                shared_y_m_e = np.concatenate(shared_y_m_e, axis=0)
                real_y_m_e = np.concatenate(real_y_m_e, axis=0)
                maj_corr_clash.append((shared_y_m_e.argmax(1) == real_y_m_e).sum())
                maj_tot_clash.append(shared_y_m_e.shape[0])
                confidences_x_m_e = np.concatenate(confidences_x_m_e)
            else:
                maj_corr_clash.append(0)
                maj_tot_clash.append(0)

            if len(shared_y_m_e_orig) > 0:
                shared_x_m_e_orig = np.concatenate(shared_x_m_e_orig, axis=0)
                shared_y_m_e_orig = np.concatenate(shared_y_m_e_orig, axis=0)
                real_y_m_e_orig = np.concatenate(real_y_m_e_orig, axis=0)
                maj_corr_orig_clash.append((shared_y_m_e_orig.argmax(1) == real_y_m_e_orig).sum())
                maj_tot_orig_clash.append(shared_y_m_e_orig.shape[0])
                confidences_x_m_e_orig = np.concatenate(confidences_x_m_e_orig)
            else:
                maj_corr_orig_clash.append(0)
                maj_tot_orig_clash.append(0)

            if len(shared_y_c) > 0:
                shared_x_c = np.concatenate(shared_x_c, axis=0)
                shared_y_c = np.concatenate(shared_y_c, axis=0)
                real_y_c = np.concatenate(real_y_c, axis=0)
                entropy_corr.append((shared_y_c.argmax(1) == real_y_c).sum())
                entropy_tot.append(shared_y_c.shape[0])
                confidences_x_c = np.concatenate(confidences_x_c)
            else:
                entropy_corr.append(0)
                entropy_tot.append(0)
        ### END: Setup ###
        # END: Collect responses by each communication protocol

        # Setup: To Record Quality of Responses for each communication protocol
        pick_all_corr_experiment.append(pick_all_corr)
        pick_all_total_experiment.append(pick_all_tot)

        pick_all_clash_corr_experiment.append(pick_all_clash_corr)
        pick_all_clash_tot_experiment.append(pick_all_clash_tot)

        majority_corr_experiment.append(maj_corr)
        majority_tot_experiment.append(maj_tot)

        majority_clash_corr_experiment.append(maj_corr_clash)
        majority_clash_tot_experiment.append(maj_tot_clash)

        maj_orig_corr_experiment.append(maj_corr_orig)
        maj_orig_tot_experiment.append(maj_tot_orig)

        maj_orig_clash_corr_experiment.append(maj_corr_orig_clash)
        maj_orig_clash_tot_experiment.append(maj_tot_orig_clash)

        entropy_corr_experiment.append(entropy_corr)
        entropy_tot_experiment.append(entropy_tot)

        ra_preds = np.array(conf_recon_p)
        ytrain_repeated = np.squeeze(np.repeat(ytrain[np.newaxis, :], ra_preds.shape[0], axis=0), axis=-1)
        fn_correct_mask = (ra_preds==ytrain_repeated).astype(int).T

        total_correct_experiment.append(np.sum(fn_correct_mask))
        # pdb.set_trace()
        # ``
        if config.true_results == 1:
            return
        # return


        # ============================================================ #
        ####################### False Negatives #######################
        # ============================================================ #

        fn_pick_all_tot = np.sum(fn_pick_all)
        fn_pick_all_clash_tot = np.sum(fn_pick_all_clash)
        fn_maj_tot = np.sum(fn_maj)
        fn_maj_clash_tot = np.sum(fn_maj_clash)
        fn_maj_orig_tot = np.sum(fn_maj_orig)
        fn_maj_orig_clash_tot = np.sum(fn_maj_orig_clash)
        fn_entropy_tot = np.sum(fn_entropy)


        fn_pick_all_cor = np.sum(np.bitwise_and(fn_correct_mask, fn_pick_all.astype(int)).astype(int))
        fn_pick_all_clash_cor = np.sum(np.bitwise_and(fn_correct_mask, fn_pick_all_clash.astype(int)).astype(int))
        fn_maj_cor = np.sum(np.bitwise_and(fn_correct_mask, fn_maj.astype(int)).astype(int))
        fn_maj_clash_cor = np.sum(np.bitwise_and(fn_correct_mask, fn_maj_clash.astype(int)).astype(int))
        fn_maj_orig_cor = np.sum(np.bitwise_and(fn_correct_mask, fn_maj_orig.astype(int)).astype(int))
        fn_maj_orig_clash_cor = np.sum(np.bitwise_and(fn_correct_mask, fn_maj_orig_clash.astype(int)).astype(int))
        fn_entropy_cor = np.sum(np.bitwise_and(fn_correct_mask, fn_entropy.astype(int)).astype(int))

        # ============================================================ #
        ####################  END: False Negatives  ####################
        # ============================================================ #

        # ============================================================ #
        ######################## START: Learn  #######################
        # ============================================================ #
        generative_X = np.concatenate([orig_model.memorized_data], 0)
        y_generative = np.concatenate([orig_model.memorized_targets], 0)
        y_generative = keras.utils.to_categorical(y_generative, config.num_classes)
        print("Memory Information")
        print("num memorized before", len(orig_model.memorized_data), np.unique(orig_model.memorized_targets))

        learn_x, learn_y = shared_x_m_e_orig, shared_y_m_e_orig
        learn_confidences = confidences_x_m_e_orig
        learnable_data = len(learn_x)
        learn_total_len = learnable_data

        if(learnable_data > 0):
            new_train_set_X = np.append(learn_x, generative_X, axis = 0)
            new_train_set_Y = np.append(learn_y, y_generative, axis = 0)
            memory_confidences = orig_model.memorized_prediction
            new_train_set_confidences = np.append(learn_confidences, memory_confidences)

            print(f"Learning on {new_train_set_confidences.shape} samples")
            dataset = tf.data.Dataset.from_tensor_slices((new_train_set_X, new_train_set_confidences, new_train_set_Y))
            dataset = dataset.shuffle(len(new_train_set_X))
            dataset = dataset.batch(20)
            update_memory(orig_model, learn_x, learn_y, learn_confidences, config.num_memorize)
            print("Trained:", end=" ")
            orig_model.fit(dataset, batch_size=20, epochs=1, validation_split=0, verbose=2, shuffle=True)
        # ============================================================ #
        ######################## END: Learn  #######################
        # ============================================================ #

        # Record Quality of Responses for each communication protocol
        if(learn_total_len > 0):
            if len(shared_y) > 0:
                error_correctly_shared[0].append((shared_y.argmax(1) == real_y).sum())
                count_correctly_shared_possible[0].append(shared_y.shape[0])
            else:
                error_correctly_shared[0].append(0)
                count_correctly_shared_possible[0].append(0)

            if len(shared_y_e) > 0:
                error_correctly_shared[1].append((shared_y_e.argmax(1) == real_y_e).sum())
                count_correctly_shared_possible[1].append(shared_y_e.shape[0])
            else:
                error_correctly_shared[1].append(0)
                count_correctly_shared_possible[1].append(0)

            if len(shared_y_m) > 0:
                error_correctly_shared[2].append((shared_y_m.argmax(1) == real_y_m).sum())
                count_correctly_shared_possible[2].append(shared_y_m.shape[0])
            else:
                error_correctly_shared[2].append(0)
                count_correctly_shared_possible[2].append(0)
            
            if len(shared_y_m_orig) > 0:
                error_correctly_shared[5].append((shared_y_m_orig.argmax(1) == real_y_m_orig).sum())
                count_correctly_shared_possible[5].append(shared_y_m_orig.shape[0])
            else:
                error_correctly_shared[5].append(0)
                count_correctly_shared_possible[5].append(0)

            if len(shared_y_m_e) > 0:
                error_correctly_shared[3].append((shared_y_m_e.argmax(1) == real_y_m_e).sum())
                count_correctly_shared_possible[3].append(shared_y_m_e.shape[0])
            else:
                error_correctly_shared[3].append(0)
                count_correctly_shared_possible[3].append(0)

            if len(shared_y_m_e_orig) > 0:
                error_correctly_shared[6].append((shared_y_m_e_orig.argmax(1) == real_y_m_e_orig).sum())
                count_correctly_shared_possible[6].append(shared_y_m_e_orig.shape[0])
            else:
                error_correctly_shared[6].append(0)
                count_correctly_shared_possible[6].append(0)
            
            if len(shared_y_c) > 0:
                error_correctly_shared[4].append((shared_y_c.argmax(1) == real_y_c).sum())
                count_correctly_shared_possible[4].append(shared_y_c.shape[0])           
            else:
                error_correctly_shared[4].append(0)
                count_correctly_shared_possible[4].append(0)

            false_negatives[0].append(fn_pick_all_cor)
            false_negatives[1].append(fn_pick_all_clash_cor)
            false_negatives[2].append(fn_maj_cor)
            false_negatives[3].append(fn_maj_clash_cor)
            false_negatives[4].append(fn_maj_orig_cor)
            false_negatives[5].append(fn_maj_orig_clash_cor)
            false_negatives[6].append(fn_entropy_cor)

            false_negatives_total[0].append(fn_pick_all_tot)
            false_negatives_total[1].append(fn_pick_all_clash_tot)
            false_negatives_total[2].append(fn_maj_tot)
            false_negatives_total[3].append(fn_maj_clash_tot)
            false_negatives_total[4].append(fn_maj_orig_tot)
            false_negatives_total[5].append(fn_maj_orig_clash_tot)
            false_negatives_total[6].append(fn_entropy_tot)

            final_total_calls_made.append(total_calls_made)

            print("Stats]",
                "Correctly Shared pick_all_no_clash: {}/{} ({}/{})".format(
                    error_correctly_shared[0][-1],
                    float(count_correctly_shared_possible[0][-1]),

                    sum(filter(None, error_correctly_shared[0])),
                    float(sum(filter(None,count_correctly_shared_possible[0])))
                ),
                "Correctly Shared pick_all_clash: {}/{} ({}/{})".format(
                    error_correctly_shared[1][-1],
                    float(count_correctly_shared_possible[1][-1]),

                    sum(filter(None, error_correctly_shared[1])),
                    float(sum(filter(None,count_correctly_shared_possible[1])))
                ),
                "Correctly Shared majority_no_clash: {}/{} ({}/{})".format(
                    error_correctly_shared[2][-1],
                    float(count_correctly_shared_possible[2][-1]),

                    sum(filter(None, error_correctly_shared[2])),
                    float(sum(filter(None,count_correctly_shared_possible[2])))
                ),
                "Correctly Shared majority_clash: {}/{} ({}/{})".format(
                    error_correctly_shared[3][-1],
                    float(count_correctly_shared_possible[3][-1]),

                    sum(filter(None, error_correctly_shared[3])),
                    float(sum(filter(None,count_correctly_shared_possible[3])))
                ),
                "Correctly Shared entropy: {}/{} ({}/{})".format(
                    error_correctly_shared[4][-1],
                    float(count_correctly_shared_possible[4][-1]),

                    sum(filter(None, error_correctly_shared[4])),
                    float(sum(filter(None,count_correctly_shared_possible[4])))
                ),
                "Correctly Shared majority_orig_no_clash: {}/{} ({}/{})".format(
                    error_correctly_shared[5][-1],
                    float(count_correctly_shared_possible[5][-1]),

                    sum(filter(None, error_correctly_shared[5])),
                    float(sum(filter(None,count_correctly_shared_possible[5])))
                ),
                "Correctly Shared majority_orig_clash: {}/{} ({}/{})".format(
                    error_correctly_shared[6][-1],
                    float(count_correctly_shared_possible[6][-1]),

                    sum(filter(None, error_correctly_shared[6])),
                    float(sum(filter(None,count_correctly_shared_possible[6])))
                ),
                "Rejected Comm: {}/20".format(
                    rejected_comm_because_qa_confident,
                ), 
                "Rejected responses: {}/400".format(
                    responses_not_received_because_qa_confident,
                ),
                "Total Calls Made: {}/20".format(
                    total_calls_made,
                ),
                "Avg Confidence: {}".format(
                    np.mean(query_agent_confidences),
                ),
            )


            print("Stats]",
                "False Negatives pick_all_no_clash: {}/{} ({}/{})".format(
                    false_negatives[0][-1],
                    false_negatives_total[0][-1],
                    sum(filter(None,false_negatives[0])),
                    sum(filter(None,false_negatives_total[0])),
                ),
                "False Negatives pick_all_clash: {}/{} ({}/{})".format(
                    false_negatives[1][-1],
                    false_negatives_total[1][-1],
                    sum(filter(None,false_negatives[1])),
                    sum(filter(None,false_negatives_total[1])),
                ),
                "False Negatives pick_maj_no_clash: {}/{} ({}/{})".format(
                    false_negatives[2][-1],
                    false_negatives_total[2][-1],
                    sum(filter(None,false_negatives[2])),
                    sum(filter(None,false_negatives_total[2])),
                ),
                "False Negatives pick_maj_clash: {}/{} ({}/{})".format(
                    false_negatives[3][-1],
                    false_negatives_total[3][-1],
                    sum(filter(None,false_negatives[3])),
                    sum(filter(None,false_negatives_total[3])),
                ),
                "False Negatives pick_maj_orig_no_clash: {}/{} ({}/{})".format(
                    false_negatives[4][-1],
                    false_negatives_total[4][-1],
                    sum(filter(None,false_negatives[4])),
                    sum(filter(None,false_negatives_total[4])),
                ),
                "False Negatives pick_maj_orig_clash: {}/{} ({}/{})".format(
                    false_negatives[5][-1],
                    false_negatives_total[5][-1],
                    sum(filter(None,false_negatives[5])),
                    sum(filter(None,false_negatives_total[5])),
                ),
                "False Negatives entropy: {}/{} ({}/{})".format(
                    false_negatives[6][-1],
                    false_negatives_total[6][-1],
                    sum(false_negatives[6]),
                    sum(filter(None,false_negatives_total[6])),
                ),
            )
        else:
            print("[Notice] Training on memories only")

            new_train_set_X, new_train_set_Y = generative_X, y_generative
            print("Trained:", end=" ")
            memory_confidences = orig_model.memorized_prediction
            dataset = tf.data.Dataset.from_tensor_slices((new_train_set_X, memory_confidences, new_train_set_Y))
            dataset = dataset.shuffle(len(new_train_set_X))
            dataset = dataset.batch(20)
            orig_model.fit(dataset, batch_size=20, epochs=epochs, validation_split=0, verbose=2, shuffle=True)

            error_correctly_shared[4].append(0)
            false_negatives[4].append(0)
            error_correctly_shared_only_c[4].append(0)
            false_negatives_only_c[4].append(0)
            error_wrong_shared[4].append(0)
            count_correctly_shared_possible[4].append(0)
            count_shared[4].append(0.0)
            count_shared_possible[4].append(xtrain.shape[0])
    else:
        raise RuntimeError()
    if print_stats == 1: 
        print("Pre-Trained:", end=" ")
        acc_pretrained[model_number].append(orig_model.evaluate(x_pretrained, y_pretrained, verbose=2)[0])
        if x_untrained is not None:
            print("Un-Trained:", end=" ")
            acc_untrained[model_number].append(orig_model.evaluate(x_untrained, y_untrained, verbose=2)[0])
        y_past = keras.utils.to_categorical(y_past, config.num_classes)
        print("Trained So Far:", end=" ")
        acc_past[model_number].append(orig_model.evaluate(x_past, y_past, verbose=2)[0])
        print("Complete Test Set:", end=" ")
        acc_complete[model_number].append(orig_model.evaluate(x_complete, y_complete, verbose=2)[0])

# Function Description:
# Used while Pretraining.
# Builds model and returns data allocated for pretraining of this model.
def build_model_data(
        name, x_train, y_train, x_test, y_test,
        model_i_classes, num_classes,
        input_shape, confidence_threshold
    ):
    x_train_i, y_train_i, x_test_i, y_test_i = datafilter(x_train, y_train, x_test, y_test, model_i_classes)
    y_train_i = keras.utils.to_categorical(y_train_i, num_classes)
    y_test_i = keras.utils.to_categorical(y_test_i, num_classes)
    model_i = VAE(input_shape, confidence_threshold=confidence_threshold, latent_dim=512)

    # working, change accordingly
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.000006)#b7
    model_i.compile(optimizer=optimizer, metrics=["accuracy"])
    return x_train_i, y_train_i, x_test_i, y_test_i, model_i

# Function Description:
# Part of old experimentation.
# Can ignore for TMLR.
def build_model_data_for_chunks(
        name, x_train, y_train, x_test, y_test,
        model_id, num_agents,
        input_shape, confidence_threshold, num_classes
    ):
    total_len = x_train.shape[0]
    total_len_t = x_test.shape[0]
    fraction = 1/num_agents
    x_train_i, y_train_i, x_test_i, y_test_i =\
        x_train[int(model_id*fraction*total_len):int((model_id+1)*fraction*total_len)],\
        y_train[int(model_id*fraction*total_len):int((model_id+1)*fraction*total_len)],\
        x_test[int(model_id*fraction*total_len_t):int((model_id+1)*fraction*total_len_t)],\
        y_test[int(model_id*fraction*total_len_t):int((model_id+1)*fraction*total_len_t)]
    
    model_i = VAE(input_shape, confidence_threshold=confidence_threshold, latent_dim=128)
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.00078)
    model_i.compile(optimizer=optimizer, metrics=["accuracy"])
    return x_train_i, y_train_i, x_test_i, y_test_i, model_i

# Function Description:
# Pretrains model.
def pretrain_model(
        name, model,
        batch_size, epochs,
        x_train, y_train,
        x_test, y_test,
        weights=None
    ):
    print("\n")
    print("training model {}".format(name))
    if weights is not None:
        model.set_weights(weights)
    model.train_mode = 0
    confidences_supervised = np.ones(len(y_train), dtype=np.float32)  # supervised: complete confidence in these samples
    # Shuffle the data
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]
    confidences_supervised_shuffled = confidences_supervised[indices]

    # Split the data into training and validation sets
    split_index = int(0.8 * len(x_train))  # 80% for training, 20% for validation

    x_train_new = x_train_shuffled[:split_index]
    y_train_new = y_train_shuffled[:split_index]
    confidences_train_new = confidences_supervised_shuffled[:split_index]

    x_val = x_train_shuffled[split_index:]
    y_val = y_train_shuffled[split_index:]

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_new, confidences_train_new, y_train_new))
    train_dataset = train_dataset.shuffle(len(x_train_new)).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    target_val_accuracy_callback = TargetValAccuracyCallback(0.75, 0.85)

    model.fit(
        train_dataset,
        batch_size=batch_size, epochs=100, shuffle=True,
        validation_data=val_dataset, verbose=2,
        callbacks=[target_val_accuracy_callback]
    )
    model.evaluate(x_test, y_test, verbose=2)

    # prepare for online lifelong learning
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.000006)#b7
    model.compile(optimizer=optimizer, metrics=["accuracy"])

# Function Description:
# Initializes memory for each model,
# given pretraining data of model and maximum limit of samples per class in memory.
def init_memories(
        model_i, x_train_i, y_train_i, num_memorize
    ):
    memorize = datafilter_perclass(
        x_train_i, y_train_i, num_memorize,
        model_i.memorized_targets, True
    )
    model_i.memorized_data.extend(memorize[0])
    model_i.memorized_targets.extend([np.expand_dims(m.argmax(0),-1) for m in memorize[1]])
    model_i.memorized_prediction = np.ones(len(memorize[0]), dtype=np.float32)*0.9

# GPU usage currently not supported.
with tf.device('/GPU:0'):

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
    config.input_shape = (32, 32, 3)
    # config.input_shape = (28, 28, 1)
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
    print('config.only_communicate_below_confidence = 0.62')
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
    

    #####################
    #   Experiment IDs
    # Use 2.2 or 5.0 for TMLR paper results
    # 2.2 -> PEEPLL, to choose communication protocol see line x [todo]
    # 5.0 -> TRUE and Selective Response Filter results
    # learning_type = 2 --> Supervised with replay
    # learning_type = 3 --> Communicative (1 model communicates with 19 others)
    #####################

    experiment_version = 2.2

    # Can ignore all but experiment_version == 2.2 or 5.0
    if experiment_version == 1.0:
        config.learning_type = 3
        config.figure_i = "1.0_1"
        config.task_learning_order = "ordered"
        config.only_communicate_below_confidence = 0.5

    elif experiment_version == 1.1:
        config.learning_type = 2
        config.figure_i = "1.1_1"
        config.task_learning_order = "ordered" 

    elif experiment_version == 2.0:
        config.learning_type = 3
        config.figure_i = "2.0_1"
        config.task_learning_order = "ordered"
        config.only_communicate_below_confidence = 0.8
        config.model0_subset_train = 100
        config.all_classes = list(range(10))
        config.model_classes = [[1, 3, 5, 7, 9], [1, 7], [2, 8], [0, 9], [6, 7, 9], [6, 4]]
        config.num_classes = 10


    # cifar10
    elif experiment_version == 2.1:
        config.learning_type = 3
        config.figure_i = "2.1_1"
        config.task_learning_order = "ordered"
        # config.task_learning_order = "random"
        config.only_communicate_below_confidence = 0.75
        config.model_classes = [[1, 3, 5, 7, 9], [1, 7], [2, 8], [0, 9, 5], [6, 7, 9], [6, 4, 3]]
        # config.num_classes = 10
        config.num_classes = 10
        config.all_classes = list(range(10))
        config.num_agents = 5
        # config.dataset = 'cifar10.load_data()'

    # cifar100
    elif experiment_version == 2.2 or experiment_version == 5.0:
        config.learning_type = 3 # PEEPLL communication
        # config.learning_type = 2 # Supervised with Replay
        config.figure_i = "2.0_1"
        config.task_learning_order = "ordered"
        # config.task_learning_order = "random"
        config.num_classes = 100

        # IMPORTANT: [todo] Set Confidence Threshold separately at line x, do not use below
        config.only_communicate_below_confidence = 0.65 
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

    
    elif experiment_version == 2.21:
        config.learning_type = 3
        # config.learning_type = 2 # IMPORTANT: CHANGE for communication
        config.figure_i = "2.0_1"
        config.task_learning_order = "ordered"
        # config.task_learning_order = "random"
        config.num_classes = 100

        config.only_communicate_below_confidence = 0.65
        config.model_classes = [\
        [3, 5, 44, 23, 68, 79, 92, 86],\

        [1, 11, 21, 31, 41, 10],\
        [0, 2, 12, 22, 32, 42, 52, 62, 72, 82, 92, 51, 23, 24, 25, 26, 27, 43, 44, 45, 46, 47, 48, 83, 84, 85, 86, 87, 36, 66, 96, 70],\
        [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 61, 0, 4, 8, 22, 24, 25, 26, 27, 49, 50, 51, 52, 54, 55, 82, 84, 85, 86, 87, 37, 20, 60],\
        [4, 14, 24, 34, 44, 54, 64, 74, 84, 94, 71, 0, 8, 28, 29, 30, 31, 32, 33, 56, 57, 58, 59, 60, 61, 62, 88, 89, 90, 91, 92, 93, 98],\
        [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 81, 1, 9, 28, 29, 30, 31, 32, 33, 34, 63, 64, 66, 67, 68, 69, 88, 89, 90, 91, 92, 93, 79],\
        [6, 16, 26, 36, 46, 56, 66, 76, 86, 96, 91, 1, 5, 9, 35, 37, 38, 39, 40, 41, 63, 64, 65, 67, 68, 69, 94, 95, 97, 98, 99, 48, 78],\
        [7, 17, 27, 37, 47, 57, 67, 77, 87, 97, 2, 6, 10, 11, 12, 13, 14, 35, 36, 38, 39, 40, 41, 70, 71, 72, 73, 74, 75, 94, 95, 96, 53],\
        [8, 18, 28, 38, 48, 58, 68, 78, 88, 98, 2, 6, 10, 11, 12, 13, 14, 42, 43, 44, 45, 46, 47, 76, 77, 79, 80, 81, 97, 99, 23, 19, 49],\
        [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 3, 7, 15, 16, 17, 18, 20, 21, 50, 51, 52, 53, 54, 55, 76, 77, 78, 80, 81, 22, 42, 82, 83],\
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 3, 7, 15, 16, 17, 18, 19, 21, 56, 57, 58, 59, 61, 62, 71, 72, 73, 74, 75, 4, 34, 5, 65],\
        ]
        config.num_agents = 10
        config.all_classes = list(range(100))

    elif experiment_version == 2.22:
        config.learning_type = 3
        # config.learning_type = 2 # IMPORTANT: CHANGE for communication
        config.figure_i = "2.0_1"
        config.task_learning_order = "ordered"
        # config.task_learning_order = "random"
        config.num_classes = 100

        config.only_communicate_below_confidence = 0.65
        config.model_classes = [\
        [3, 5, 44, 23, 68, 79, 92, 86],\

        [1, 11, 21, 31, 41, 10],\
        list(range(0, 5)) + list(range(5, 10)),\
        list(range(5, 10)) + list(range(10, 15)),\
        list(range(10, 15)) + list(range(15, 20)),\
        list(range(15, 20)) + list(range(20, 25)),\
        list(range(20, 25)) + list(range(25, 30)),\
        list(range(25, 30) )+ list(range(30, 35)),\
        list(range(30, 35)) + list(range(35, 40)),\
        list(range(35, 40)) + list(range(40, 45)),\
        list(range(40, 45)) + list(range(45, 50)),\
        list(range(45, 50)) + list(range(50, 55)),\
        list(range(50, 60)),\
        list(range(55, 65)),\
        list(range(60, 70)),\
        list(range(65, 75)),\
        list(range(70, 80)),\
        list(range(75, 85)),\
        list(range(80, 90)),\
        list(range(85, 95)),\
        list(range(90, 100)),\
        list(range(95, 100)) + list(range(0, 5)),\
        list(range(0, 5)) + list(range(10, 15)),\
        list(range(5, 10)) + list(range(15, 20)),\
        list(range(10, 15)) + list(range(20, 25)),\
        list(range(15, 20)) + list(range(25, 30)),\
        list(range(20, 25)) + list(range(30, 35)),\
        list(range(25, 30)) + list(range(35, 40)),\
        list(range(30, 35)) + list(range(40, 45)),\
        list(range(35, 40)) + list(range(45, 50)),\
        list(range(40, 45)) + list(range(50, 55)),\
        list(range(45, 50)) + list(range(55, 60)),\
        list(range(50, 55)) + list(range(60, 65)),\
        list(range(55, 60)) + list(range(65, 70)),\
        list(range(60, 65)) + list(range(70, 75)),\
        list(range(65, 70)) + list(range(75, 80)),\
        list(range(70, 75)) + list(range(80, 85)),\
        list(range(75, 80)) + list(range(85, 90)),\
        list(range(80, 85)) + list(range(90, 95)),\
        list(range(85, 90)) + list(range(95, 100)),\
        list(range(90, 95)) + list(range(5, 10)),\
        # list(range(95, 100)) + list(range(10, 15)),\
        ]
        config.num_agents = 40
        config.all_classes = list(range(100))
    
    elif experiment_version == 2.23:
        config.learning_type = 3
        # config.learning_type = 2 # IMPORTANT: CHANGE for communication
        config.figure_i = "2.0_1"
        config.task_learning_order = "ordered"
        config.num_classes = 100

        config.only_communicate_below_confidence = 0.65
        config.model_classes = [\
        [3, 5, 44, 23, 68, 79, 92, 86],\
        [1, 11, 21, 31, 41, 10],\
        ]
        for i in range(0, 98):
            config.model_classes.append([i, i+1, i+2])
        config.model_classes.append([98, 99, 0])
        config.num_agents = 100
        config.all_classes = list(range(100))

    # chunks
    elif experiment_version == 3.0:
        config.learning_type = 3
        config.figure_i = "2.0_1"
        config.task_learning_order = "ordered"
        config.only_communicate_below_confidence = 0.8
        config.class_wise = False
        config.num_agents = 5
        config.num_classes = 10
        config.all_classes = list(range(10))

    if config.learning_type == 2:
        learning_config = dict(
            learning_setup = 2,
        )
    elif config.learning_type == 3:
        learning_config = dict(
            compare_communication = config.compare_communication,
            learning_setup = 3,
            only_communicate_below_confidence = config.only_communicate_below_confidence,
        )
    else:
        raise RuntimeError("No such learning type: {}".format(config.learning_type))

    learning_config = argparse.Namespace(**learning_config)

    if config.load_i is None:
        config.load_i = config.figure_i

    #####################
    #   info
    #####################

    pprint(vars(learning_config), depth=2)
    pprint(vars(config), depth=2)

    print("\n\n\n")
    print("="*100)
    print("Building Models and Dataset")
    print("="*100)

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
        # Ignore for TMLR Results
        print("Chunks Experiment")
        indices = tf.range(start=0, limit=tf.shape(x_train)[0])
        shuffled_indices = tf.random.shuffle(indices)
        indices_t = tf.range(start=0, limit=tf.shape(x_test)[0])
        shuffled_indices_t = tf.random.shuffle(indices_t)
        x_train = tf.gather(x_train, shuffled_indices)
        y_train = tf.gather(y_train, shuffled_indices)
        x_test = tf.gather(x_test, shuffled_indices_t)
        y_test = tf.gather(y_test, shuffled_indices_t)

        total_len = x_train.shape[0]
        total_len_test = x_test.shape[0]
        x_train_baseline, y_train_baseline = x_train[0:int(0.1*total_len)], y_train[0:int(0.1*total_len)]
        x_test_baseline, y_test_baseline = x_test[0:int(0.1*total_len_test)], y_test[0:int(0.1*total_len_test)]
        x_train, y_train = x_train[int(0.1*total_len):], y_train[int(0.1*total_len):]
        x_test, y_test = x_test[int(0.1*total_len_test):], y_test[int(0.1*total_len_test):]

        print(x_train.shape)
        print(y_train.shape)
        print(x_train_baseline.shape)


    print("shapes of baselines:", x_train_baseline.shape, y_train_baseline.shape)
    y_train_baseline = keras.utils.to_categorical(y_train_baseline, config.num_classes)
    y_test_baseline = keras.utils.to_categorical(y_test_baseline, config.num_classes)
    model_baseline = VAE(config.input_shape, confidence_threshold=config.confidence_threshold, latent_dim=128)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.000009)
    model_baseline.compile(optimizer=optimizer, metrics=["accuracy"])


    x_trainN, y_trainN, x_testN, y_testN, modelN = [], [], [], [], []
    y_train_precateg_N, y_test_precateg_N = [], []

    for i in range(config.num_agents):
        if config.class_wise: 
            x_tr, y_tr, x_te, y_te, m = build_model_data(
            f"{i+1}", x_train, y_train, x_test, y_test,
            config.model_classes[i+1], config.num_classes,
            config.input_shape, config.confidence_threshold
            )
        else:
            x_tr, y_tr, x_te, y_te, m = build_model_data_for_chunks(
            "1", x_train, y_train, x_test, y_test,
            i, config.num_agents,
            config.input_shape, config.confidence_threshold, config.num_classes
            )
            y_train_precateg_N.append(y_tr)
            y_test_precateg_N.append(y_te)
            y_tr = keras.utils.to_categorical(y_tr, config.num_classes)
            y_te = keras.utils.to_categorical(y_te, config.num_classes)

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
        model_baseline.build(input_shape_b) # Note: baseline model was part of our OLD experimentation.
        for mi in modelN:
            mi.build(input_shape_b)
        # Working (1/27), (5/9)
        for i, mi in enumerate(modelN):
            mi.load_weights(f'/work/pi_hava_umass_edu/prithvi/save_cifar100/save_cifar100/{i+1}_newdatakl5_{config.figure_i}.h5')
            print(f'loaded model {i}')
        loaded_weights = True

        print('loaded weights')
        # print('Agent performances on their training set')
        # for i, mi in enumerate(modelN):
        #     print(f'Agent {i}: {mi.evaluate(x_trainN[i], y_trainN[i], verbose=0)}')
        # pdb.set_trace()

    except Exception as e:
        print("Failed to load Model due to exception, ", e)

    if not loaded_weights:

        print("\n")
        print(x_train.shape)
        for i, mi in enumerate(modelN):
            print(f'For model {i+1}: {x_trainN[i].shape, y_trainN[i].shape, np.unique(y_trainN[i])}')
            # Important note here:
            # The baseline model defined above is NOT used here in weights. It was part of old experimentation.
            pretrain_model(
            f"{i+1}", mi,
            config.batch_size, config.epochs,
            x_trainN[i], y_trainN[i],
            x_testN[i], y_testN[i],
            weights = None
            )
            mi.save_weights(f'/work/pi_hava_umass_edu/prithvi/save_cifar100/{i+1}_40ssagentserr_100a_{config.figure_i}.h5')
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
        
        tasks = len(modelN_untrained_classes[1])//5 #9
        total_samples = 0
        print(f'tasks: {tasks}')
        # tasks = len(modelN_untrained_classes[1])//2 #4
        for i in range(tasks): #8
            x_train_untrained1, y_train_untrained1, x_test_untrained, y_test_untrained = datafilter(x_train, y_train, x_test, y_test, modelN_untrained_classes[1][5*i:5*(i+1)])
            # x_train_untrained1, y_train_untrained1, _, _ = datafilter(x_train, y_train, x_test, y_test, modelN_untrained_classes[1][2*i:2*(i+1)])
            task_wise_x.append(x_train_untrained1)
            task_wise_y.append(y_train_untrained1)
            task_wise_x_test.append(x_test_untrained)
            task_wise_y_test.append(y_test_untrained)
            total_samples += x_train_untrained1.shape[0]
        x_train_untrained1, y_train_untrained1, x_test_untrained, y_test_untrained = datafilter(x_train, y_train, x_test, y_test, modelN_untrained_classes[1][5*tasks:])
        # x_train_untrained1, y_train_untrained1, _, _ = datafilter(x_train, y_train, x_test, y_test, modelN_untrained_classes[1][2*tasks:])
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

    communication_calls = []
    acc_pretrained = [[] for i in range(total_num_models)]
    acc_untrained = [[] for i in range(total_num_models)]
    acc_past = [[] for i in range(total_num_models)]
    acc_complete = [[] for i in range(total_num_models)]
    error_correctly_shared = [[] for i in range(total_num_models)]
    false_negatives = [[] for i in range(10)]
    false_negatives_total = [[] for i in range(10)]
    error_correctly_shared_only_c = [[] for i in range(total_num_models)]
    false_negatives_only_c = [[] for i in range(total_num_models)]
    error_wrong_shared = [[] for i in range(total_num_models)]
    count_shared = [[] for i in range(total_num_models)]
    count_shared_possible = [[] for i in range(total_num_models)]
    count_correctly_shared_possible = [[] for i in range(total_num_models)]
    count_correctly_shared_possible_only_c = [[] for i in range(total_num_models)]
    final_total_calls_made = []
    avg_qa_conf = []

    # Threshold Experiments Lists
    pick_all_corr_experiment = []
    pick_all_total_experiment = []

    pick_all_clash_corr_experiment = []
    pick_all_clash_tot_experiment = []

    majority_corr_experiment = []
    majority_tot_experiment = []

    majority_clash_corr_experiment = []
    majority_clash_tot_experiment = []

    maj_orig_corr_experiment = []
    maj_orig_tot_experiment = []

    maj_orig_clash_corr_experiment = []
    maj_orig_clash_tot_experiment = []

    entropy_corr_experiment = []
    entropy_tot_experiment = []

    total_correct_experiment = []



    mse = keras.losses.MeanSquaredError()


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
                # task_id = 18
                if i == training_len-1:
                    print_stats = 1
                else:
                    # change to 1 for always recording and printing,
                    # use 0 when trying to reduce runtime
                    print_stats = 0  

                if task_id != 18:
                    print(
                            "train", x_train[i0:i1].shape, y_train[i0:i1].shape, np.unique(y_train[i0:i1]),
                    "\n", "pretrained", x_testN[0].shape, y_testN[0].shape, np.unique(y_testN[0].argmax(1)),
                    "\n", "untrained", np.concatenate(task_wise_x_test[task_id+1:]).shape, keras.utils.to_categorical(np.concatenate(task_wise_y_test[task_id+1:]), config.num_classes).shape, np.unique(np.concatenate(task_wise_y_test[task_id+1:])),
                    "\n", "past", np.concatenate(task_wise_x_test[:task_id+1]).shape, np.concatenate(task_wise_y_test[:task_id+1]).shape, np.unique(np.concatenate(task_wise_y_test[:task_id+1])),
                    "\n", "generative", x_trainN[0].shape, y_trainN[0].shape, np.unique(y_trainN[0].argmax(1)),
                    "\n", "all", x_test.shape, y_test_categorical.shape, np.unique(y_test)
                    )
                    communication(
                        0, x_train[i0:i1], y_train[i0:i1],
                        modelN,
                        x_testN[0], y_testN[0],
                        np.concatenate(task_wise_x_test[task_id+1:]), keras.utils.to_categorical(np.concatenate(task_wise_y_test[task_id+1:]), config.num_classes),
                        np.concatenate(task_wise_x_test[:task_id+1]), np.concatenate(task_wise_y_test[:task_id+1]), x_trainN[0], y_trainN[0], x_test, y_test_categorical,
                        print_stats, config.batch_size_inner_loop, config.epochs_inner_loop)
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
                        print_stats, config.batch_size_inner_loop, config.epochs_inner_loop)


                communication_calls.append(i)
                print(f"error_wrong_shared: {error_wrong_shared}")
                print(f"total shared: {count_correctly_shared_possible}")
                # pdb.set_trace()
                graphing_data = (
                    communication_calls,
                    acc_pretrained, acc_untrained,
                    acc_past, acc_complete,
                    count_shared, count_shared_possible,
                    error_correctly_shared,
                    false_negatives,
                    count_correctly_shared_possible,
                    error_wrong_shared,
                )
                with open(f"./finalruns_cifar100_reproduce_confclash_8c_{config.figure_i}.pkl", 'wb') as f:
                    pickle.dump(graphing_data, f)

                experiments_data = (
                    pick_all_corr_experiment,
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
                    avg_qa_conf,
                )
                print("seed=1, rcc8c")
                with open(f"./finalruns_cifar100_reproduce_confclash_8c_data_{config.figure_i}.pkl", 'wb') as f:
                    pickle.dump(experiments_data, f)