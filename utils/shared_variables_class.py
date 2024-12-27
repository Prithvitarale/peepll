class SharedVariables:
    def __init__(self, total_num_models):
        self.communication_calls = []
        self.acc_pretrained = [[] for i in range(total_num_models)]
        self.acc_untrained = [[] for i in range(total_num_models)]
        self.acc_past = [[] for i in range(total_num_models)]
        self.acc_complete = [[] for i in range(total_num_models)]
        self.error_correctly_shared = [[] for i in range(total_num_models)]
        self.false_negatives = [[] for i in range(10)]
        self.false_negatives_total = [[] for i in range(10)]
        self.error_correctly_shared_only_c = [[] for i in range(total_num_models)]
        self.false_negatives_only_c = [[] for i in range(total_num_models)]
        self.error_wrong_shared = [[] for i in range(total_num_models)]
        self.count_shared = [[] for i in range(total_num_models)]
        self.count_shared_possible = [[] for i in range(total_num_models)]
        self.count_correctly_shared_possible = [[] for i in range(total_num_models)]
        self.count_correctly_shared_possible_only_c = [[] for i in range(total_num_models)]

        # Threshold Experiments Lists
        self.pick_all_corr_experiment = []
        self.pick_all_total_experiment = []

        self.pick_all_clash_corr_experiment = []
        self.pick_all_clash_tot_experiment = []

        self.majority_corr_experiment = []
        self.majority_tot_experiment = []

        self.majority_clash_corr_experiment = []
        self.majority_clash_tot_experiment = []

        self.maj_orig_corr_experiment = []
        self.maj_orig_tot_experiment = []

        self.maj_orig_clash_corr_experiment = []
        self.maj_orig_clash_tot_experiment = []

        self.entropy_corr_experiment = []
        self.entropy_tot_experiment = []

        self.total_correct_experiment = []
        self.final_total_calls_made = []
        self.avg_qa_conf = []