# Distributed Mult-Agent Lifelong Learning (PEEPLL Framework)

Official repository of [Distributed Multi-Agent Lifelong Learning](https://openreview.net/forum?id=IIVr4Hu3Oi) (TMLR 2025).

This repository contains the implementation of the **Peer Parallel Lifelong Learning (PEEPLL)** framework for distributed multi-agent lifelong learning. The scripts provided reproduce the results presented in the paper. To review specific implementations of our proposed methods in the paper, see Section 5 below.

## **Overview**
- For CIFAR100 and MiniImageNet, the `peepll.py` script simulates lifelong learning through peer-to-peer communication.
- **Protocols**:
  - For **CIFAR100**, **TRUE + REFINE** is our best-performing protocol.  
  - For **MiniImageNet**, **TRUE + Majority** is our best-performing protocol.
---

## **Usage Instructions**

### **1. Install Dependencies**
To install all necessary dependencies, create a Conda environment using the provided `environment.yml` file:
- `conda env create -f environment.yml`
- `conda activate <your_env_name>`

### **2. Saved Models**
If you wish to use our saved pre-trained models, they can be found [here](https://drive.google.com/drive/folders/1kA5V5Rp-ZN5QgPtfKJ8SusCl8DEKm42L?usp=sharing).

### **3. Run the Scripts**
The following are the instructions to reproduce the experiments:
Run `./python3 peepll.py --dataset --experiment_id --learning_type`

- '--dataset': M for MiniImageNet, C for CIFAR100 (input type: string)
- '--experiment_id': 2.2 for Communication and LL, 5 for TRUE and Filter results (input type: float)
- '--learning_type': 2 for Supervised, 3 for Communicative (input type: integer)

For visualizing results, 
- TRUE vs Entropy: `python3 ./visualization/compare-entropy-ours.py`
- Selective Response Filters Comparison - `python3 ./visualization/compare-filtering.py`
- Lifelong Learning (Performance on (1) pre-trained data, (2) untrained data (future tasks), (3) Past Trained (tasks introduced so far), (4) Complete Test set) - `python3 ./visualization/lifelong_results.py`
- QA's Increasing Confidence in Tasks and Subsequent Need for Less Communication - `python3 ./visualization/reducing.py`
 
We have included simple and short instructions on how to run each of these files correctly (filename, etc.) in the respective files.

### **4. Customizing Communication Protocols**
To run different communication protocols, update the following lines in `communication.py` under `utils/`:

#### 1. Line 544-545: Modify (a, b, c) values:
- learn_x, learn_y = a, b
- learn_confidences = c

Use the following mappings for `(a, b, c)`:

- **Entropy**: `shared_x_c, shared_y_c, confidences_x_c`
- **TRUE**: `shared_x, shared_y, confidences_x`
- **TRUE + ICF**: `shared_x_e, shared_y_e, confidences_x_e`
- **TRUE + Majority**: `shared_x_m, shared_y_m, confidences_x_m`
- **TRUE + Majority + ICF**: `shared_x_m_e, shared_y_m_e, confidences_x_m_e`
- **TRUE + MCG**: `shared_x_m_orig, shared_y_m_orig, confidences_x_m_orig`
- **TRUE + MCG + ICF (REFINE)**: `shared_x_m_e_orig, shared_y_m_e_orig, confidences_x_m_e_orig`


#### 2. Line 145: Adjust the aa_threshold if needed: aa_threshold = [value]
Recommended thresholds for a 1:1 Sharing Ratio are given on lines 60-77.

### **5. Our Algorithms**
We have thoroughly documented all our functions, and our introduced algorithms (*Dynamic Memory-Update*, TRUE, ICF, MCG, Majority). Following are where you can find each algorithm:
- ***Dynamic Memory-Update*** and **TRUE:** `peepll_utils.py`
- **ICF:** Lines 176-189 in `communication.py`
- **MCG:** Lines 292-293 in `communication.py`
- **Majority:** Lines 297-298 in `communication.py`

### **6. Notes for Reproducibility**
- Important: This code is designed to run on CPU only. All our TMLR results were reported by running experiments on the CPU. GPU support will be added soon.
- Results are averaged over multiple runs, as described in the paper. For consistency, ensure the random seeds are set as in the scripts.
- Logs and outputs will include key metrics and figures to match the results in the paper.




