# nCMD

This repository contains the code and resources for the paper "nCMD: Benign-Anchored Feature Selection for Imbalanced Network Intrusion Detection" authored by Abu Fuad Ahmad and Istiaque Ahmed, submitted at the IEEE MILCOM 2026 Conference.
 

## Requirements & Setup
To install, 
You can run the following command:

To create an environment (name : 'ncmd'):

    conda create -n ncmd python=3.11

 To activate the environment:

    conda activate ncmd 

A full list of requirements can be found
in `requirements.txt`. To install, 

    pip install -r requirements.txt


## Repository Structure
```
├── data/                        # Folder for datasets (contents ignored in .gitignore)
├── feature_selection.py         # Feature selection logic and methods
├── models.py                    # Model definitions and training scripts
├── utils.py                     # General utility functions including training loops
├── utils_eval.py                # Evaluation utility functions and results saving
├── utils_feature_selection.py   # Utilities specific to feature selection method
├── run_all.sh                   # Execute for every possible arguments combination
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Ignore rules for data and other files
```

## Runnig the Code

Put the datasets in 'data/' folder. 
CICIDS datasets can be found here: https://eltnmsu-my.sharepoint.com/:f:/r/personal/hcao_nmsu_edu/Documents/DATA/2025_ICMLA_Fuad_Ahmad_DATA?csf=1&web=1&e=LZUyRo 

Instructions for running the code are below.

To get the results on CICIDS2017 dataset using proposed feature selection method with default argument settings, run: 

    python3 feature_selection.py

Available arguments:
    Dataset, d: "CICIDS2017" "CICIDS2019"  "NSL"  "UNSW" 
    Feature Selection method, f:  "cmd"  "ncmd"  "corr"  "thr"   "fish"   "mi"
    Number of feature to select, n: 40 30 20 10 5
    Classifier, c: 'mlp' 'DT' 

To run on UNSW dataset using Fisher score method for 40 feature and DT classifier:

    python3 feature_selection.py -d="UNSW" -m="fish" -n=40 -c='DT' 
 




