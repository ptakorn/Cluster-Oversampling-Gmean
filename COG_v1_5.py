
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from imblearn.over_sampling import SMOTE

# G-mean calculation with pos_label
def g_mean(y_true, y_pred, pos_label=1):
    y_true_bin = [1 if y == pos_label else 0 for y in y_true]
    y_pred_bin = [1 if y == pos_label else 0 for y in y_pred]
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    return np.sqrt(sensitivity * specificity)

# Load dataset and remove categorical columns
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.select_dtypes(exclude=['object'])
    return df

# Split dataset
def preprocess_data(df):
    X = df.drop(columns=['Target'])
    y = df['Target']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Oversample each cluster
def oversample_cluster(cluster_data, current_ir, target_ir, minority_class):
    counts = cluster_data['Target'].value_counts()
    minority_count = counts[minority_class]
    majority_class = counts.drop(index=minority_class).idxmax()
    majority_count = counts[majority_class]

    if minority_count < 2 or (minority_count / majority_count) >= target_ir:
        return cluster_data.drop(columns=['Target']), cluster_data['Target'], 0

    try:
        smote_ratio = min((target_ir - current_ir) / (1 - current_ir), 0.5)
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=42)
        X_res, y_res = smote.fit_resample(cluster_data.drop(columns=['Target']), cluster_data['Target'])
        oversampled_instances = len(y_res) - len(cluster_data)
        return X_res, y_res, oversampled_instances
    except ValueError:
        return cluster_data.drop(columns=['Target']), cluster_data['Target'], 0

def apply_algorithm(file_path, n_clusters, target_ir, minority_class=1, patience=3, base_classifier=None):
    if base_classifier is None:
        base_classifier = DecisionTreeClassifier(random_state=42)

    df = load_data(file_path)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)

    # Initial model training and baseline G-mean
    model_init = clone(base_classifier)
    model_init.fit(X_train, y_train)
    best_gmean = g_mean(y_val, model_init.predict(X_val), pos_label=minority_class)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_train)
    S = pd.DataFrame(X_train, columns=df.drop(columns=['Target']).columns)
    S['Target'] = y_train.values
    S['Cluster'] = clusters
    resampling_counts = {i: 0 for i in range(n_clusters)}
    total_synthetic = 0

    for i in range(n_clusters):
        cluster_data = S[S['Cluster'] == i]
        if (cluster_data['Target'] == minority_class).sum() < 2:
            continue

        counts = cluster_data['Target'].value_counts()
        minority_count = counts[minority_class]
        majority_class = counts.drop(index=minority_class).idxmax()
        majority_count = counts[majority_class]
        current_ir = minority_count / majority_count
        no_improve_count = 0

        while current_ir < target_ir:
            X_res, y_res, added_instances = oversample_cluster(cluster_data, current_ir, target_ir, minority_class)
            if added_instances == 0:
                break

            temp_S = pd.concat([S.drop(S[S['Cluster'] == i].index),
                                pd.concat([X_res, y_res], axis=1)], ignore_index=True)
            temp_S['Cluster'] = -1  # Reset cluster info after resampling

            model_tmp = clone(base_classifier)
            model_tmp.fit(temp_S.drop(columns=['Target', 'Cluster']), temp_S['Target'])
            gmean_tmp = g_mean(y_val, model_tmp.predict(X_val), pos_label=minority_class)

            if gmean_tmp > best_gmean:
                best_gmean = gmean_tmp
                S = temp_S
                counts_res = y_res.value_counts()
                minority_count = counts_res[minority_class]
                majority_count = counts_res.drop(index=minority_class).max()
                current_ir = minority_count / majority_count
                resampling_counts[i] += added_instances
                total_synthetic += added_instances
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    break

    X_final, y_final = S.drop(columns=['Target', 'Cluster']), S['Target']
    final_model = clone(base_classifier)
    final_model.fit(X_final, y_final)
    final_score = g_mean(y_test, final_model.predict(X_test), pos_label=minority_class)

    return final_score, total_synthetic

