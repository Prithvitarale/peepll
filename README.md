# PEEPLL Framework Codebase

This repository contains the implementation of the **Peer Parallel Lifelong Learning (PEEPLL)** framework for distributed multi-agent lifelong learning. The scripts provided reproduce the results presented in our paper.

## **Overview**
- **Default Protocols**:  
  - For **CIFAR100**, the script runs **TRUE + REFINE** (our best-performing protocol).  
  - For **MiniImageNet**, the script runs **TRUE + Majority** (our best-performing protocol).
- **Experiment Selection**:
  - In `both peell_cifar.py` and `peell_miniImageNet.py`, use the `experiment_number` variable on line 1304 to choose between experiments for **PEEPLL with communication (2.2)** or **TRUE results (5)**:
    - `2.2` → PEEPLL with Communication (default)
    - `5.0` → TRUE Results

---

## **Usage Instructions**

### **1. Install Dependencies**
To install all necessary dependencies, create a Conda environment using the provided `environment.yml` file:
- `conda env create -f environment.yml`
- `conda activate <your_env_name>`

### **2. Run the Scripts**
Use the following commands to run experiments for the respective datasets:

- CIFAR100: `python3 peell_cifar.py`

- MiniImageNet: `python3 peell_miniImageNet.py`

By default, these will reproduce the TRUE + REFINE results for CIFAR100 and TRUE + Majority results for MiniImageNet, our best performing Communication Protocols for respective datasets.

### **3. Customizing Communication Protocols**
To run different communication protocols, update the following lines in the script:

#### 1. Line 905-906: Modify (a, b, c) values:
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


#### 2. Line 506: Adjust the aa_threshold if needed: aa_threshold = [value]

### **4. Our Algorithms**
We have thoroughly documented all our functions, and our introduced algorithms (*Dynamic Memory-Update*, TRUE, ICF, MCG, Majority). Following are where you can find each algorithm:
- ***Dynamic Memory-Update:*** Lines 235-299
- **TRUE:** Lines 301-354
- **ICF:** Lines 476-493 and Lines 534-553
- **MCG:** Lines 651-654
- **Majority:** Lines 656-659

### **5. Notes for Reproducibility**
- Important: This code is designed to run on CPU only. GPU support will be added soon.
- Results are averaged over multiple runs, as described in the paper. For consistency, ensure the random seeds are set as in the scripts.
- Logs and outputs will include key metrics and figures to match the results in the paper.




