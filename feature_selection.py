import os
import time
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

import torch
import torch.nn as nn

import utils
import models 
import utils_feature_selection as utils_fs

import utils_eval

import time


def fs_correlation(df, num_feature, target_class):
    top_features = utils_fs.get_selected_features_using_correlation(df, n=num_feature, target_column = 'label')
    return top_features 
def fs_mutualInfo(df, num_feature, target_class):
    top_features = utils_fs.get_selected_features_using_mutual_info(df, num_feature, target_column='label')
    return top_features
def fs_threshold(df, num_feature, target_class):
    top_features = utils_fs.get_selected_features_using_threshold(df, num_feature)
    return top_features 

# save features
def save_features(selected_features, file_name, end_time, start_time):
    #file_name = "NSL_fisher_top_features.txt" #Specify the file name
    with open(file_name, "w") as file:
        file.write(f"Execution time: {end_time - start_time:.4f} seconds\n")
        for item in selected_features:
            file.write(item + "\n")

def fs_fisher(df, num_feature, target_class):
    start_time = time.time()
    top_features = utils_fs.get_selected_features_using_fisher_score(df, num_feature)
    #print(top_features)
    save_features(top_features, "UNSW_fisher_top_features.txt", start_time, time.time())
    return top_features 

def fs_cmd(df, num_feature, target_class):
    top_features = utils_fs.get_selected_features_using_CMD(df, n = num_feature)
    return top_features # idx = utils_fs.get_selected_features_using_CMD(X, y)

def fs_ncmd(df, num_feature, target_class):
    top_features = utils_fs.get_selected_features_using_nCMD(df, n = num_feature, target_class = target_class)
    return top_features

SELECTORS = {    "fish": fs_fisher,
    "corr": fs_correlation,
    "mi": fs_mutualInfo,
    "thr": fs_threshold,
    "cmd": fs_cmd,
    "ncmd": fs_ncmd,

}

def load_KDD(train_data_file, test_data_file):
    train_data = pd.read_csv(train_data_file) 
    test_data = pd.read_csv(test_data_file)
    return train_data, test_data

def run_exe_for_separate_train_test_data(args):
    print(args.dataset, "  dataset, loading...") 
    target_class = 0
    if args.dataset == 'NSL':
        train_data_file = 'data/NSL_KDD/KDDTrain+_20Percent.txt' 
        test_data_file = 'data/NSL_KDD/KDDTest+.txt'
        target_class = 'normal' 
        train_data, test_data = utils.load_and_preprocess_nsl_kdd_data(train_data_file, test_data_file)
    elif args.dataset == 'UNSW':
        train_data_file = 'data/UNSW_NB15/UNSW_NB15_training-set.csv' 
        test_data_file = 'data/UNSW_NB15/UNSW_NB15_testing-set.csv'
        target_class = 'Normal' 
        train_data, test_data = utils.load_and_preprocess_unsw_nb15_data(train_data_file, test_data_file)
    
    print("train data.shape: ", train_data.shape)

    print("Selecting top "+str(args.num_feature)+" features using : "+args.fea_sel_method)
    selector = SELECTORS[args.fea_sel_method] #fs_cmd 
    
    top_features = selector(train_data, args.num_feature, target_class)
    top_features.append('label')
    train_data = train_data[top_features]
    test_data = test_data[top_features]

    metrics=['accuracy', 'weighted_f1',  'macro_f1'] #,   

    output_path = f"outputs/{args.dataset}_{args.fea_sel_method}_{args.num_feature}_{args.clf}_results.txt"
    utils.train_test_with_separate_test_data(train_data, test_data, output_path, batch_size=512, epochs=10, learning_rate=0.001, metrics=metrics,clf=args.clf)


def main(args):
    target_class = 0
    if args.dataset in ['KDD','NSL', 'UNSW']:
        run_exe_for_separate_train_test_data(args)
        return
    
    if args.dataset == 'CICIDS2019':
        data_file = 'data/CICIDS2019/CICIDS2019_Day1-20-percent_common_features_un_norm.csv'
        target_class = 'BENIGN'
    else:    
        data_file = "data/CICIDS2017/CICIDS2017_common_features_un_norm.csv"
        target_class = 0
    assert os.path.isfile(data_file),  'dataset not found.'

    print(args.dataset, "  dataset, loading...")
    df = pd.read_csv(data_file) 
    print("data.shape: ", df.shape)
    

    print("Training and testing using K-Fold...")
    selector = SELECTORS[args.fea_sel_method] #fs_cmd 
    metrics=['accuracy', 'weighted_f1',  'macro_f1']
    output_path = f"outputs/{args.dataset}_{args.fea_sel_method}_{args.num_feature}_{args.clf}_results.txt"
    utils.train_test_kfold_avg(df, target_class, selector, args.num_feature, output_path, n_splits=5, batch_size=512, epochs=args.epochs, learning_rate=0.001, metrics=metrics,clf=args.clf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',
        help="Dataset to use: 'CICIDS2017', 'CICIDS2019', 'UNSW' or 'NSL' ", required=False, default='CICIDS2017', type=str)

    parser.add_argument('-m', '--fea_sel_method',
        help="Feature selection method.",
        default='ncmd', type=str)
    
    parser.add_argument('-n', '--num_feature',
        help="Number of features to select.", required=False, default=40, type=int )
    
    parser.add_argument('-c', '--clf', help="Classification method.", default='mlp', type=str)
    
    parser.add_argument('-e', '--epochs',
        help="Number of epochs to use for training.", required=False, default=10, type=int)
    
    args = parser.parse_args()

    main(args)

