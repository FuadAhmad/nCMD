import numpy as np
from sklearn.metrics import classification_report, confusion_matrix # ,f1_score, ConfusionMatrixDisplay
from sklearn import model_selection

import sys
sys.path.append("../")  # add parent folder
import scipy.io

def get_data_XY(dataset):
    # load data
    mat = scipy.io.loadmat('../skfeature/data/'+ dataset +'.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    return X, y


def run_experiment(dataset, n_folds, num_fea, selector, clf, metrics, output_path):
    
    X, y = get_data_XY(dataset)

    ss = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {metric: [] for metric in metrics}
    for train, test in ss.split(X, y): #---------------------Stratified
        # obtain the index of each feature on the training set
        #idx = MIFS.mifs(X[train], y[train], n_selected_features=num_fea)
        #idx = utils_feature_selection.get_selected_features_using_CMD(X[train], y[train])
        idx = selector(X[train], y[train], num_fea)
        # obtain the dataset on the selected features
        features = X[:, idx[0][:num_fea]] #---------------features = X[:, idx[0:num_fea]]

        # train a classification model with the selected features on the training dataset
        clf.fit(features[train], y[train])

        # predict the class labels of test data
        y_predict = clf.predict(features[test])

        # obtain the classification accuracy on the test data
        results = update_results(results, metrics, y[test], y_predict)

    save_results(results, output_path)


def update_results(results, metrics, y_test, y_pred):
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    for metric in metrics:
        print(metric, end=' ')

        # Accuracy
        if metric == 'accuracy':
            value = report['accuracy']

        # F1-score
        elif metric.endswith('_f1'):
            avg = metric.replace('_f1', ' avg')  # macro / weighted
            value = report[avg]['f1-score']

        # Precision
        elif metric.endswith('_precision'):
            avg = metric.replace('_precision', ' avg')
            value = report[avg]['precision']

        # Recall
        elif metric.endswith('_recall'):
            avg = metric.replace('_recall', ' avg')
            value = report[avg]['recall']

        # False Positive Rate
        elif metric == 'fp_rate':
            value = get_false_positive_rate(y_test, y_pred)

        else:
            raise ValueError(f"Unknown metric: {metric}")

        results[metric].append(value)
        print(value)
    return results

def get_false_positive_rate(y_test, y_pred):
    conf_mat = confusion_matrix(y_test, y_pred)
                    
    # Calculate false positive rate for each class and take the average
    fp_rate_per_class = []
    for i in range(conf_mat.shape[0]):
        fp = conf_mat[:, i].sum() - conf_mat[i, i]
        tn = conf_mat.sum() - (conf_mat[i, :].sum() + conf_mat[:, i].sum() - conf_mat[i, i])
        if fp + tn > 0:
            fp_rate = fp / (fp + tn)
            fp_rate_per_class.append(fp_rate)
    false_positive_rate = np.mean(fp_rate_per_class) if fp_rate_per_class else 0.0
    return false_positive_rate

def save_results(results, output_path = "classification_results.txt"):
    # Compute summary
    summary = {
        metric: {
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
        for metric, scores in results.items()
    }
    
    #print(summary)
    for metric, stats in summary.items():
        print(f"{metric}:\t{stats['mean']:.4f} ± {stats['std']:.4f}")

    with open(output_path, "w") as f:
        f.write("=== Raw Metric Scores ===\n")
        for metric, scores in results.items():
            f.write(f"{metric}: {scores}\n")

        f.write("\n=== Summary (Mean ± Std) ===\n")
        for metric, stats in summary.items():
            f.write(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            #f"{metric}: " f"mean = {stats['mean']:.4f}, "f"std = {stats['std']:.4f}\n" )

    print(f"Results saved to {output_path}")