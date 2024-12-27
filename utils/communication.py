import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.peepll_utils import identify_from_memory_scalable_forkl, call_response_agent, update_memory


# ================================#
####  Main Communication Method ####
# ================================#
def communication(
        model_number, xtrain, ytrain,
        models,
        x_pretrained, y_pretrained,
        x_untrained, y_untrained,
        x_past, y_past, x_generative, y_generative, x_complete, y_complete,
        print_stats, config, shared_variables, batch_size=1, epochs=1,
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
    if config.learning_type  == 2:
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
    # If you need to change communication protocol, change (a, b, c) on line 905-906:
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
    # Set the threshold on line 506, aa_threshold:
    # For MiniImageNet and CIFAR100 respectively,
    # Entropy                   -> 0.92, 0.9
    # TRUE                      -> 0.745, 0.79
    # TRUE + ICF                -> 0.635, 0.61
    # TRUE + Majority           -> 0.73, 0.77
    # TRUE + Majority + ICF     -> 0.623, 0.6
    # TRUE + MCG                -> 0.73, 0.75
    # TRUE + MCG + ICF (REFINE) -> 0.63, 0.585

    # These are suggested values for a 1:1 Sharing Ratio. The QA will be introduced
    # 20k queries in total, so further adjust the thresholds if you do not observe
    # 20k responses in total.
    # ================================================================#
    # ================================================================#
    elif config.learning_type  == 3:
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
        for idx, array in enumerate(classes_they_think_they_know): # Go through all agents
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
        print('no reducing with confn')
        preds = np.squeeze(np.expand_dims(keras.utils.to_categorical(np.array(conf_recon_p).T, config.num_classes),2), 2)
        pick_model = np.vstack((q_c, np.array(all_confidences))).T
        pick_model_all = pick_model
        pick_model = pick_model.argmax(1)
        
        query_agent_confidences = q_c
        shared_variables.avg_qa_conf.append(np.mean(query_agent_confidences))

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
            # pdb.set_trace()
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
        shared_variables.pick_all_corr_experiment.append(pick_all_corr)
        shared_variables.pick_all_total_experiment.append(pick_all_tot)

        shared_variables.pick_all_clash_corr_experiment.append(pick_all_clash_corr)
        shared_variables.pick_all_clash_tot_experiment.append(pick_all_clash_tot)

        shared_variables.majority_corr_experiment.append(maj_corr)
        shared_variables.majority_tot_experiment.append(maj_tot)

        shared_variables.majority_clash_corr_experiment.append(maj_corr_clash)
        shared_variables.majority_clash_tot_experiment.append(maj_tot_clash)

        shared_variables.maj_orig_corr_experiment.append(maj_corr_orig)
        shared_variables.maj_orig_tot_experiment.append(maj_tot_orig)

        shared_variables.maj_orig_clash_corr_experiment.append(maj_corr_orig_clash)
        shared_variables.maj_orig_clash_tot_experiment.append(maj_tot_orig_clash)

        shared_variables.entropy_corr_experiment.append(entropy_corr)
        shared_variables.entropy_tot_experiment.append(entropy_tot)

        ra_preds = np.array(conf_recon_p)
        ytrain_repeated = np.squeeze(np.repeat(ytrain[np.newaxis, :], ra_preds.shape[0], axis=0), axis=-1)
        fn_correct_mask = (ra_preds==ytrain_repeated).astype(int).T

        shared_variables.total_correct_experiment.append(np.sum(fn_correct_mask))
        # pdb.set_trace()
        # ``
        if config.true_results == 1:
            return


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
        ####################  END: False Negatives  ###################
        # ============================================================ #



        # ============================================================ #
        ######################## START: Learn  ########################
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
                shared_variables.error_correctly_shared[0].append((shared_y.argmax(1) == real_y).sum())
                shared_variables.count_correctly_shared_possible[0].append(shared_y.shape[0])
            else:
                shared_variables.error_correctly_shared[0].append(0)
                shared_variables.count_correctly_shared_possible[0].append(0)

            if len(shared_y_e) > 0:
                shared_variables.error_correctly_shared[1].append((shared_y_e.argmax(1) == real_y_e).sum())
                shared_variables.count_correctly_shared_possible[1].append(shared_y_e.shape[0])
            else:
                shared_variables.error_correctly_shared[1].append(0)
                shared_variables.count_correctly_shared_possible[1].append(0)

            if len(shared_y_m) > 0:
                shared_variables.error_correctly_shared[2].append((shared_y_m.argmax(1) == real_y_m).sum())
                shared_variables.count_correctly_shared_possible[2].append(shared_y_m.shape[0])
            else:
                shared_variables.error_correctly_shared[2].append(0)
                shared_variables.count_correctly_shared_possible[2].append(0)
            
            if len(shared_y_m_orig) > 0:
                shared_variables.error_correctly_shared[5].append((shared_y_m_orig.argmax(1) == real_y_m_orig).sum())
                shared_variables.count_correctly_shared_possible[5].append(shared_y_m_orig.shape[0])
            else:
                shared_variables.error_correctly_shared[5].append(0)
                shared_variables.count_correctly_shared_possible[5].append(0)

            if len(shared_y_m_e) > 0:
                shared_variables.error_correctly_shared[3].append((shared_y_m_e.argmax(1) == real_y_m_e).sum())
                shared_variables.count_correctly_shared_possible[3].append(shared_y_m_e.shape[0])
            else:
                shared_variables.error_correctly_shared[3].append(0)
                shared_variables.count_correctly_shared_possible[3].append(0)

            if len(shared_y_m_e_orig) > 0:
                shared_variables.error_correctly_shared[6].append((shared_y_m_e_orig.argmax(1) == real_y_m_e_orig).sum())
                shared_variables.count_correctly_shared_possible[6].append(shared_y_m_e_orig.shape[0])
            else:
                shared_variables.error_correctly_shared[6].append(0)
                shared_variables.count_correctly_shared_possible[6].append(0)
            
            if len(shared_y_c) > 0:
                shared_variables.error_correctly_shared[4].append((shared_y_c.argmax(1) == real_y_c).sum())
                shared_variables.count_correctly_shared_possible[4].append(shared_y_c.shape[0])           
            else:
                shared_variables.error_correctly_shared[4].append(0)     
                shared_variables.count_correctly_shared_possible[4].append(0)           

            shared_variables.false_negatives[0].append(fn_pick_all_cor)
            shared_variables.false_negatives[1].append(fn_pick_all_clash_cor)
            shared_variables.false_negatives[2].append(fn_maj_cor)
            shared_variables.false_negatives[3].append(fn_maj_clash_cor)
            shared_variables.false_negatives[4].append(fn_maj_orig_cor)
            shared_variables.false_negatives[5].append(fn_maj_orig_clash_cor)
            shared_variables.false_negatives[6].append(fn_entropy_cor)

            shared_variables.false_negatives_total[0].append(fn_pick_all_tot)
            shared_variables.false_negatives_total[1].append(fn_pick_all_clash_tot)
            shared_variables.false_negatives_total[2].append(fn_maj_tot)
            shared_variables.false_negatives_total[3].append(fn_maj_clash_tot)
            shared_variables.false_negatives_total[4].append(fn_maj_orig_tot)
            shared_variables.false_negatives_total[5].append(fn_maj_orig_clash_tot)
            shared_variables.false_negatives_total[6].append(fn_entropy_tot)

            shared_variables.final_total_calls_made.append(total_calls_made)

            print("Stats]",
                "Correctly Shared pick_all_no_clash: {}/{} ({}/{})".format(
                    shared_variables.error_correctly_shared[0][-1],
                    float(shared_variables.count_correctly_shared_possible[0][-1]),

                    sum(filter(None, shared_variables.error_correctly_shared[0])),
                    float(sum(filter(None,shared_variables.count_correctly_shared_possible[0])))
                ),
                "Correctly Shared pick_all_clash: {}/{} ({}/{})".format(
                    shared_variables.error_correctly_shared[1][-1],
                    float(shared_variables.count_correctly_shared_possible[1][-1]),

                    sum(filter(None, shared_variables.error_correctly_shared[1])),
                    float(sum(filter(None,shared_variables.count_correctly_shared_possible[1])))
                ),
                "Correctly Shared majority_no_clash: {}/{} ({}/{})".format(
                    shared_variables.error_correctly_shared[2][-1],
                    float(shared_variables.count_correctly_shared_possible[2][-1]),

                    sum(filter(None, shared_variables.error_correctly_shared[2])),
                    float(sum(filter(None,shared_variables.count_correctly_shared_possible[2])))
                ),
                "Correctly Shared majority_clash: {}/{} ({}/{})".format(
                    shared_variables.error_correctly_shared[3][-1],
                    float(shared_variables.count_correctly_shared_possible[3][-1]),

                    sum(filter(None, shared_variables.error_correctly_shared[3])),
                    float(sum(filter(None,shared_variables.count_correctly_shared_possible[3])))
                ),
                "Correctly Shared entropy: {}/{} ({}/{})".format(
                    shared_variables.error_correctly_shared[4][-1],
                    float(shared_variables.count_correctly_shared_possible[4][-1]),

                    sum(filter(None, shared_variables.error_correctly_shared[4])),
                    float(sum(filter(None,shared_variables.count_correctly_shared_possible[4])))
                ),
                "Correctly Shared majority_orig_no_clash: {}/{} ({}/{})".format(
                    shared_variables.error_correctly_shared[5][-1],
                    float(shared_variables.count_correctly_shared_possible[5][-1]),

                    sum(filter(None, shared_variables.error_correctly_shared[5])),
                    float(sum(filter(None,shared_variables.count_correctly_shared_possible[5])))
                ),
                "Correctly Shared majority_orig_clash: {}/{} ({}/{})".format(
                    shared_variables.error_correctly_shared[6][-1],
                    float(shared_variables.count_correctly_shared_possible[6][-1]),

                    sum(filter(None, shared_variables.error_correctly_shared[6])),
                    float(sum(filter(None,shared_variables.count_correctly_shared_possible[6])))
                ),
                "Correctly Shared majority_orig_clash: {}/{}".format(
                    maj_corr_orig_clash,
                    maj_tot_orig_clash,
                ),
                "Rejected Comm: {}/64".format(
                    rejected_comm_because_qa_confident,
                ), 
                "Rejected responses: {}/1280".format(
                    responses_not_received_because_qa_confident,
                )
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

            shared_variables.error_correctly_shared[4].append(0)
            shared_variables.false_negatives[4].append(0)
            shared_variables.error_correctly_shared_only_c[4].append(0)
            shared_variables.false_negatives_only_c[4].append(0)
            shared_variables.error_wrong_shared[4].append(0)
            shared_variables.count_correctly_shared_possible[4].append(0)
            shared_variables.count_shared[4].append(0.0)
            shared_variables.count_shared_possible[4].append(xtrain.shape[0])
    else:
        raise RuntimeError()

    if print_stats == 1:
        print("Pre-Trained:", end=" ")
        shared_variables.acc_pretrained[model_number].append(orig_model.evaluate(x_pretrained, y_pretrained, verbose=2)[0])
        if x_untrained is not None:
            print("Un-Trained:", end=" ")
            shared_variables.acc_untrained[model_number].append(orig_model.evaluate(x_untrained, y_untrained, verbose=2)[0])
        y_past = keras.utils.to_categorical(y_past, config.num_classes)
        print("Trained So Far:", end=" ")
        shared_variables.acc_past[model_number].append(orig_model.evaluate(x_past, y_past, verbose=2)[0])
        print("Complete Test Set:", end=" ")
        shared_variables.acc_complete[model_number].append(orig_model.evaluate(x_complete, y_complete, verbose=2)[0])
