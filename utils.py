

import os
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
#import tensorflow as tf

from tqdm import tqdm 
from sklearn.model_selection import StratifiedKFold

import models


# Class to load custom dataset
class CustomDataset(Dataset):

    def __init__(self, X_data, y_data, for_binary = False):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        if for_binary:
            self.y = torch.tensor(y_data, dtype=torch.float32)    # Binary classification
        else:
            self.y = torch.tensor(y_data, dtype=torch.long)         # Muti class classification
        self.length = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.length

def get_dataloaders(train_df, test_df, batch_size = 128):
    #X_train_ft, y_train_ft)
    y_train_ft = train_df['label']
    X_train_ft = train_df.drop(columns=['label'])
    
    # process TEST data #X_test_ft, y_test_ft)
    y_test_ft = test_df['label'] 
    X_test_ft = test_df.drop(columns=['label'])

    #y_test = get_binary_labels(y_test)
    CLASS_LABELS = np.unique(y_train_ft) #LE.inverse_transform(y)#d.unique() 
    print(CLASS_LABELS)
    num_classes = len(CLASS_LABELS)

    #encode labels to numbers
    LE = LabelEncoder()
    LE.fit(y_train_ft)
    y_train_ft = LE.transform(y_train_ft)
    y_test_ft = LE.transform(y_test_ft)

    # Sclae independently or on train?
    testscaler = MinMaxScaler()
    X_train_ft = testscaler.fit_transform(X_train_ft)
    X_test_ft = testscaler.transform(X_test_ft)

    if num_classes == 2:
        train_data_ft = CustomDataset(X_train_ft, y_train_ft, for_binary=True)
        test_data_ft = CustomDataset(X_test_ft, y_test_ft, for_binary=True)
    else:
        test_data_ft = CustomDataset(X_test_ft, y_test_ft)
        train_data_ft = CustomDataset(X_train_ft, y_train_ft)

    train_loader = DataLoader(train_data_ft, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data_ft, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_all_labels_all_preds(model, test_loader, DEVICE = "cpu"):
    model.to(DEVICE)
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move the input data and labels to the device
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

def train_torch_model(model, num_epochs, train_loader, criterion, optimizer, DEVICE = "cpu", logoutput=True):
    model.to(DEVICE).train()
    for epoch in range(num_epochs): # num_epochs = 10  # Example
        epoch_loss = 0
        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        #for i, (inputs, labels) in enumerate(train_loader): # Assuming you have a DataLoader
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels) # Labels should be LongTensor for CrossEntropyLoss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        #if (i+1) % 1000 == 0: # Print every 100 steps
        #    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
        if logoutput:
            print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader)}")
    return model

def test_torch_model(model, test_loader, DEVICE = "cpu"):
    model.to(DEVICE)
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move the input data and labels to the device
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Perform the forward pass
            outputs = model(inputs)

            # Calculate loss or evaluate performance metrics
            #loss = loss_fn(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print the results
    #print('Accuracy on the test data: %d %%' % (100 * correct / total))
    report = classification_report(all_labels, all_preds, digits=4)
    print(report)


def get_combined_df(directory):
    df_comb = pd.DataFrame()
    for root, dirs, files in os.walk(directory):
        print(f"Directory: {root}")
        #for dir_name in dirs:
        #    print(f"  Folder: {dir_name}")
        for file_name in files:
            if file_name.endswith('.csv'):
                file_path = os.path.join(root, file_name) #f"  File: {root}/{file_name}"
                print(file_path)
                df = pd.read_csv(file_path)
                print(df.shape) 
                df_comb = pd.concat([df_comb, df])
                #break
    return df_comb

# Cross Validation
def train_test_kfold(df, n_splits=5, batch_size=512, epochs=10, learning_rate=0.001, n=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    for fold, (train_idx, test_idx) in enumerate(skf.split(df, df.label)):
        print(f"\nFold {fold + 1}/{n_splits}")
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]

        #print("train_data.shape:", train_data.shape)
        #print(train_data.label.value_counts())

        num_classes = len(train_data.label.value_counts())
        #print('Number of classes:', num_classes)

        train_loader, test_loader = get_dataloaders(train_data, test_data, batch_size=batch_size)

        model = models.MLP_Mult(input_shape=train_data.shape[1] - 1, num_classes=num_classes).to(DEVICE).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        model = train_torch_model(
            model=model,
            num_epochs=epochs,
            train_loader=train_loader,
            criterion=loss_fn,
            optimizer=optimizer,
            DEVICE=DEVICE
        )

        test_torch_model(model, test_loader, DEVICE=DEVICE)

def train_test_kfold_avg(df, target_class, selector, num_feature, output_path, n_splits=5, batch_size=512, epochs=10, learning_rate=0.001, metrics=['accuracy'],clf='mlp'):
    """
    Evaluates specified metrics using K-Fold cross-validation.

    Parameters:
    - df: dataset to train and test on.
    - metrics: list of metric names to extract from classification_report
    - n_splits: number of folds for cross-validation

    Returns:
    - Dictionary with mean and std for each requested metric
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    results = {metric: [] for metric in metrics}
    for fold, (train_idx, test_idx) in enumerate(skf.split(df, df.label)):
        print(f"\nFold {fold + 1}/{n_splits}")
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]


        top_features = selector(train_data, num_feature, target_class)
        top_features.append('label')
        train_data = train_data[top_features]
        test_data = test_data[top_features]

        print("train_data.shape:", train_data.shape)
        #print(train_data.label.value_counts())

        y_test, y_pred = Train_model_get_ytest_ypred(train_data, test_data, batch_size, epochs, learning_rate,clf)

        #report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results = utils_eval.update_results(results, metrics, y_test, y_pred)

    # Compute mean and std
    utils_eval.save_results(results, output_path)
    #return summary

import utils_eval
def train_test_with_separate_test_data(train_data, test_data, output_path, batch_size=512, epochs=10, learning_rate=0.001, metrics=['accuracy', 'micro avg', 'weighted avg'],clf='mlp'):
    results = {metric: [] for metric in metrics}
    y_test, y_pred = Train_model_get_ytest_ypred(train_data, test_data, batch_size, epochs, learning_rate,clf)
    results = utils_eval.update_results(results, metrics, y_test, y_pred)
    utils_eval.save_results(results, output_path)


def Train_model_get_ytest_ypred(train_data, test_data, batch_size, epochs, learning_rate, clf):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)
    num_classes = len(train_data.label.value_counts())
    #print('Number of classes:', num_classes)
    if clf=='mlp':
        train_loader, test_loader = get_dataloaders(train_data, test_data, batch_size=batch_size)

        model = models.MLP_Mult(input_shape=train_data.shape[1] - 1, num_classes=num_classes).to(DEVICE).train()
        #print("Using MLP_Mult_Class_1-hidden-layer")
        #model = models.MLP_Mult_hCMD(input_shape=train_data.shape[1] - 1, num_classes=num_classes).to(DEVICE).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        model = train_torch_model(
            model=model,
            num_epochs=epochs,
            train_loader=train_loader,
            criterion=loss_fn,
            optimizer=optimizer,
            DEVICE=DEVICE, logoutput=False
        )

        #utils.test_torch_model(model, test_loader, DEVICE=DEVICE)
        y_test, y_pred = get_all_labels_all_preds(model, test_loader, DEVICE = "cpu")
    else:
        y_train_ft = train_data['label']
        X_train_ft = train_data.drop(columns=['label'])
        y_test = test_data['label']
        X_test = test_data.drop(columns=['label'])
        
        from sklearn.tree import DecisionTreeClassifier
        print("Using Decision Tree")
        model = DecisionTreeClassifier(random_state=42)
        # Train
        model.fit(X_train_ft, y_train_ft)

        # Predict
        y_pred = model.predict(X_test)

        # Output compatible with your pipeline
        y_test = list(y_test)
        y_pred = list(y_pred)
    return y_test, y_pred


def load_and_preprocess_nsl_kdd_data(train_path, test_path):

    columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
    'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate', 'label','difficulty']

    train_df = pd.read_csv(train_path, names=columns)
    test_df  = pd.read_csv(test_path, names=columns)
    #print("train df shape: ", train_df.shape)
    #print("test df shape: ", test_df.shape)
    train_df.drop(columns=['difficulty'], inplace=True)
    test_df.drop(columns=['difficulty'], inplace=True)

    # Find common labels
    #common_labels = set(y_train.unique()).intersection(set(y_test.unique()))
    common_labels = set(train_df['label'].unique()).intersection(set(test_df['label'].unique()))
    # Filter BOTH train and test
    train_df = train_df[train_df['label'].isin(common_labels)]
    test_df  = test_df[test_df['label'].isin(common_labels)]

    X_train, y_train = train_df.drop('label', axis=1), train_df['label']
    X_test, y_test   = test_df.drop('label', axis=1), test_df['label']

    #print("train X shape: ", X_train.shape)
    #print("test X shape: ", X_test.shape)

    # One-hot encode categorical
    X_train = pd.get_dummies(X_train)
    X_test  = pd.get_dummies(X_test)
    #print("train X shape: ", X_train.shape)
    #print("test X shape: ", X_test.shape)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    #print("train X shape: ", X_train.shape)
    #print("test X shape: ", X_test.shape)

    X_train['label'] = y_train
    X_test['label']  = y_test

    return X_train, X_test

def load_and_preprocess_unsw_nb15_data(train_path, test_path, target_col='attack_cat'):

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    #print("train df shape: ", train_df.shape)
    #print("test df shape: ", test_df.shape)

    # Drop unnecessary columns
    for col in ['id']:
        if col in train_df.columns:
            train_df.drop(columns=[col], inplace=True)
            test_df.drop(columns=[col], inplace=True)

    # Find common labels
    #common_labels = set(y_train.unique()).intersection(set(y_test.unique()))
    common_labels = set(train_df[target_col].unique()).intersection(set(test_df[target_col].unique()))
    #print("#Labels: train-", len(set(train_df[target_col].unique())), " test-", len(set(test_df[target_col].unique())), " common-", len(common_labels))
    # Filter BOTH train and test
    train_df = train_df[train_df[target_col].isin(common_labels)]
    test_df  = test_df[test_df[target_col].isin(common_labels)]

    # Features and labels
    X_train = train_df.drop(columns=['label', 'attack_cat'])
    y_train = train_df[target_col] #train_df['label']

    X_test  = test_df.drop(columns=['label', 'attack_cat'])
    y_test  = test_df[target_col]

    # One-hot encode categorical
    X_train = pd.get_dummies(X_train)
    X_test  = pd.get_dummies(X_test)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    X_train['label'] = y_train
    X_test['label']  = y_test
    #print("train df shape: ", X_train.shape)    #print("test df shape: ", X_test.shape)
    return X_train, X_test