import numpy as np
import tensorflow as tf
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
    _, memory_z, z_means, z_logs = model(memory) 
    _, queries_z, q_mean, q_log = model(queries) 

    all_scores = []
    for q, query in enumerate(queries): # For each query to the model

        # Get memory samples of the predicted class for this query
        matching_indices = np.isin(model.memorized_targets, pred[q])
        matching_indices = np.where(matching_indices)[0].flatten()

        if len(matching_indices) == 0: # If no memory, 0 confidence
            all_scores.append(0)
        else:
            # Get z_means for memory of predicted class for this query
            matching_z_means = np.array(z_means)[matching_indices, :] 
            matching_z_logs = np.array(z_logs)[matching_indices, :] 
            matching_z = np.array(memory_z)[matching_indices, :]
            query_var = np.exp(q_log[q, :])
            # Get z_var for memory of predicted class
            memory_var = np.exp(matching_z_logs)
            mean_memory_var = np.mean(memory_var, axis=0)
            mean_memory_mean = np.mean(matching_z_means, axis=0)

            entropy = confN[q][0]
            ms = ms_scores[q][0]

            # Dispersion Distance
            uncertainty_score = np.exp(-0.01*np.sum(np.abs(mean_memory_var-query_var)))
            uncertainty_score = (uncertainty_score - 0.4)/(0.5)
            uncertainty_score = max(uncertainty_score, 0)
            uncertainty_score = min(uncertainty_score, 1)

            # Semantic Distance
            mean_distance = np.exp(-0.1*tf.norm(mean_memory_mean-q_mean[q, :]))
            mean_distance = (mean_distance - 0.4)/(0.4)
            mean_distance = max(mean_distance, 0)
            mean_distance = min(mean_distance, 1)

            # Average of Semantic, Dispersion, and Entropy Scores
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
    print(f'Model thinks they know {filtered_elements}')
    confidences = identify_from_memory_scalable_forkl(model, xt, p, confN_append, ms_append)
    # Need to fix - can be reduced to 3
    return predN_append, conf_recon_p_append, ms_append, confN_append, classes_they_think_they_know_append, confidences



