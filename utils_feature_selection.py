import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.feature_selection import mutual_info_classif



def get_selected_features_using_CMD(df, n=20):

    constant_columns = df.columns[df.nunique() == 1].tolist()
    df = df.drop(columns=constant_columns)
    # if df contain NaN, replace with mean
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Separate the features and the label
    features = df.iloc[:, :-1]  # all columns except the last
    labels = df.iloc[:, -1]     # the last column

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    df = pd.concat([scaled_df.reset_index(drop=True), labels.reset_index(drop=True)], axis=1) #final_df
    # after MinMax scale: #  if contain NaN, replace with 0 
    df.fillna(0, inplace=True) # optional may be

    mean_per_class = df.groupby('label').mean()
    # Calculate the overall mean of the DataFrame (excluding the 'Label' column)
    overall_mean = df.drop(columns=['label']).mean()

    mean_diff = mean_per_class - overall_mean
    mean_diff_abs = mean_diff.abs()
    # Compute column-wise sums
    column_sums = mean_diff_abs.sum()

    #Select the top n features with the highest sums
    top_features = column_sums.sort_values(ascending=False).head(n).index.tolist() 

    return top_features

def get_selected_features_using_nCMD(df, n=20, target_class = 0):
    X = df.iloc[:, :-1]  # all columns except the last
    ranked_feature_names = compute_nc_md_scores(df, target_class)
    return ranked_feature_names[:n] 
    

def compute_nc_md_scores(df, target_class): #['benign']
    constant_columns = df.columns[df.nunique() == 1].tolist()
    df = df.drop(columns=constant_columns)
    # if df contain NaN, replace with mean
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Separate the features and the label
    features = df.iloc[:, :-1]  # all columns except the last
    labels = df.iloc[:, -1]     # the last column

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    df = pd.concat([scaled_df.reset_index(drop=True), labels.reset_index(drop=True)], axis=1) #final_df
    # after MinMax scale: #  if contain NaN, replace with 0 
    df.fillna(0, inplace=True) # optional may be

    mean_per_class = df.groupby('label').mean()
    
    # modifications/adaptations    # Calculate the overall mean of the DataFrame (excluding the 'Label' column)
    benign_mean = mean_per_class.loc[target_class] #overall_mean = df.drop(columns=['label']).mean()
    non_benign_means = mean_per_class.drop(index=target_class)

    mean_diff = non_benign_means - benign_mean #mean_diff = mean_per_class - overall_mean
    mean_diff_abs = mean_diff.abs()
    # Compute column-wise sums
    column_sums = mean_diff_abs.sum()

    #Select the top n features with the highest sums scores
    top_features = column_sums.sort_values(ascending=False).index.tolist() #.sort_values(ascending=False).head(n).index.tolist() 

    return top_features


def get_selected_features_using_threshold(df, n):
    # if df contain NaN, replace with mean
    df.fillna(df.mean(numeric_only=True), inplace=True)
    # Select only numeric features
    numeric_data = df.select_dtypes(include=[np.number])

    # Compute variance for each feature
    variances = numeric_data.var()

    # Sort features by variance in descending order
    sorted_variances = variances.sort_values(ascending=False)

    # Select top n features
    top_n_features = sorted_variances.head(n).index.tolist()

    return top_n_features



def get_selected_features_using_mutual_info(df, n, target_column='label'):
    """
    Selects the top n features based on mutual information with the target column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    n (int): Number of top features to select.

    Returns:
    List[str]: Names of the top n features.
    """

    # if df contain NaN, replace with mean
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical features
    X_encoded = pd.get_dummies(X)

    X_encoded.fillna(0, inplace=True)

    # Encode target if it's categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)

    # Compute mutual information
    mi = mutual_info_classif(X_encoded, y, discrete_features='auto', random_state=42)
    mi_series = pd.Series(mi, index=X_encoded.columns)

    # Select top n features
    top_features = mi_series.sort_values(ascending=False).head(n).index.tolist()

    return top_features

def get_selected_features_using_correlation(df, n, target_column = 'label'):

    # if df contain NaN, replace with mean
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Encode target if it's not numeric
    le = None
    if df[target_column].dtype == 'object':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])

    # Select only numeric features
    numeric_data = df.select_dtypes(include=[np.number])

    # Ensure target column is in the data
    if target_column not in numeric_data.columns:
        raise ValueError(f"Target column '{target_column}' not found or not numeric.")
    # Compute correlation with the target
    correlations = numeric_data.corr()[target_column].drop(target_column).abs()

    # Select top n features
    top_n_features = correlations.sort_values(ascending=False).head(n).index.tolist() #sorted_variances.head(n).index.tolist()
    if le:
        df[target_column] = le.inverse_transform(df[target_column])
    return top_n_features

def fisher_score_cal(X, y):
    """
    Generalized Fisher score: between-class scatter / within-class scatter.
    """
    classes = np.unique(y)
    mu = X.mean(axis=0)
    d = X.shape[1]
    num = np.zeros(d)
    den = np.zeros(d)
    for c in classes:
        Xc = X[y == c]
        nc = Xc.shape[0]
        mu_c = Xc.mean(axis=0)
        var_c = Xc.var(axis=0)
        num += nc * (mu_c - mu) ** 2
        den += nc * var_c
    den[den == 0] = 1e-12
    return num / den

def get_selected_features_using_fisher_score(df, n=13, target_column='label'):
    # Encode the target column
    labelencoder = LabelEncoder()
    df[target_column] = labelencoder.fit_transform(df[target_column])

    # Separate features and target
    y = df[target_column].values
    X = df.drop(columns=[target_column])

    # Convert X to NumPy array for compatibility with FCBF
    X_np = X.values
    X_np = X_np.astype(float)
   
    scores = fisher_score_cal(X_np, y)
    k = min(n, X.shape[1])
    # Get selected feature indices and names
    selected_indices = np.argsort(scores)[::-1][:k] #idx[:n]  # Limit to top n features
    selected_features = X.columns[selected_indices]

    df[target_column] = labelencoder.inverse_transform(df[target_column])
    return selected_features.tolist()


