import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from paradoteo_A1 import load_data_from_csv_parquet_format


def evaluate_binary_classification(data_samples):
    """
    Evaluates SVM for binary classification (Label column - malicious/benign)
    """
    # Dictionary to store results for comparison
    results = {}

    for sample_name, file_path in data_samples.items():
        print(f"\n\n===== Binary Classification: {sample_name} dataset =====")

        try:
            # Load data
            df = load_data_from_csv_parquet_format(file_path)
            print(f"Loaded {df.height} rows and {df.width} columns")

            # Prepare binary target (Label column)
            y = df.select(pl.col("Label")).to_numpy().flatten()
            if np.issubdtype(y.dtype, np.number):
                # If numeric, assume 0 = benign, anything else = malicious
                y_binary = (y != 0).astype(int)
            else:
                # If text, convert "Benign"/"Normal" to 0, everything else to 1
                # Case-insensitive check for common benign labels
                y_binary = np.array([0 if str(label).lower() in ['benign', 'normal'] else 1
                                     for label in y], dtype=int)

            # Extract numeric features
            X = df.select([
                col for col in df.columns
                if col not in ['Label', 'Traffic Type']
                and df.schema[col] in (pl.Float64, pl.Int64, pl.Float32, pl.Int32)
            ]).fill_null(0).to_numpy()

            # Class distribution
            benign = np.sum(y_binary == 0)
            malicious = np.sum(y_binary == 1)
            print(f"Class distribution: {benign} benign, {malicious} malicious "
                  f"({malicious/(benign+malicious)*100:.2f}% malicious)")

            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
            )

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train SVM with balanced class weights
            print("Training SVM model with balanced class weights...")
            svm_model = SVC(kernel='rbf', C=10.0, gamma='scale',
                            probability=True, random_state=321, class_weight='balanced')
            svm_model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = svm_model.predict(X_test_scaled)
            y_prob = svm_model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Per-class metrics (more important for imbalanced data)
            benign_precision = precision_score(y_test, y_pred, pos_label=0)
            benign_recall = recall_score(y_test, y_pred, pos_label=0)
            benign_f1 = f1_score(y_test, y_pred, pos_label=0)

            malicious_precision = precision_score(y_test, y_pred, pos_label=1)
            malicious_recall = recall_score(y_test, y_pred, pos_label=1)
            malicious_f1 = f1_score(y_test, y_pred, pos_label=1)

            # Overall metrics
            weighted_precision = precision_score(
                y_test, y_pred, average='weighted')
            weighted_recall = recall_score(y_test, y_pred, average='weighted')
            weighted_f1 = f1_score(y_test, y_pred, average='weighted')
            auc = roc_auc_score(y_test, y_prob)

            # Store results
            results[sample_name] = {
                'accuracy': accuracy,
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1': weighted_f1,
                'auc': auc,
                'benign_precision': benign_precision,
                'benign_recall': benign_recall,
                'benign_f1': benign_f1,
                'malicious_precision': malicious_precision,
                'malicious_recall': malicious_recall,
                'malicious_f1': malicious_f1
            }

            # Print confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            print("\nConfusion Matrix:")
            print(conf_matrix)

            # Print classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # Print key metrics
            print(f"Overall accuracy: {accuracy:.4f}")
            print(
                f"Benign - Precision: {benign_precision:.4f}, Recall: {benign_recall:.4f}, F1: {benign_f1:.4f}")
            print(
                f"Malicious - Precision: {malicious_precision:.4f}, Recall: {malicious_recall:.4f}, F1: {malicious_f1:.4f}")
            print(
                f"Weighted - Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1: {weighted_f1:.4f}")
            print(f"ROC AUC: {auc:.4f}")

        except Exception as e:
            print(f"Error processing {sample_name} dataset: {e}")
            results[sample_name] = {'error': str(e)}
    if svm_model is not None:
        diagnose_perfect_scores(df, X, y_binary, svm_model)
    return results


def evaluate_multiclass_classification(data_samples):
    """
    Evaluates SVM for multi-class classification (Traffic Type column)
    """
    # Dictionary to store results for comparison
    results = {}

    for sample_name, file_path in data_samples.items():
        print(
            f"\n\n===== Multi-class Classification: {sample_name} dataset =====")

        try:
            # Load data
            df = load_data_from_csv_parquet_format(file_path)
            print(f"Loaded {df.height} rows and {df.width} columns")

            # Prepare multi-class target (Traffic Type column)
            y = df.select(pl.col("Traffic Type")).to_numpy().flatten()

            # Extract numeric features
            X = df.select([
                col for col in df.columns
                if col not in ['Label', 'Traffic Type']
                and df.schema[col] in (pl.Float64, pl.Int64, pl.Float32, pl.Int32)
            ]).fill_null(0).to_numpy()

            # Class distribution
            unique_classes, counts = np.unique(y, return_counts=True)
            print("Class distribution:")
            for cls, count in zip(unique_classes, counts):
                print(f"  {cls}: {count} samples ({count/len(y)*100:.2f}%)")

            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train SVM with balanced class weights
            print("Training SVM model with balanced class weights...")
            svm_model = SVC(kernel='rbf', C=10.0, gamma='scale',
                            probability=True, random_state=42, class_weight='balanced')
            svm_model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = svm_model.predict(X_test_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            weighted_precision = precision_score(
                y_test, y_pred, average='weighted')
            weighted_recall = recall_score(y_test, y_pred, average='weighted')
            weighted_f1 = f1_score(y_test, y_pred, average='weighted')

            # Store results
            results[sample_name] = {
                'accuracy': accuracy,
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1': weighted_f1
            }

            # Print confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            print("\nConfusion Matrix:")
            print(conf_matrix)

            # Print classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # Print key metrics
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Weighted precision: {weighted_precision:.4f}")
            print(f"Weighted recall: {weighted_recall:.4f}")
            print(f"Weighted F1 score: {weighted_f1:.4f}")

        except Exception as e:
            print(f"Error processing {sample_name} dataset: {e}")
            results[sample_name] = {'error': str(e)}

    return results


def save_and_visualize_results(binary_results, multiclass_results):
    """
    Saves and visualizes SVM results for both classification tasks
    """
    # Create results directory
    os.makedirs('svm_results', exist_ok=True)

    # Process binary classification results
    binary_df = pd.DataFrame(
        columns=['Dataset', 'Accuracy', 'Weighted_F1', 'Benign_F1', 'Malicious_F1', 'AUC'])

    for sample, metrics in binary_results.items():
        if 'error' not in metrics:
            binary_df = binary_df._append({
                'Dataset': sample,
                'Accuracy': metrics['accuracy'],
                'Weighted_F1': metrics['weighted_f1'],
                'Benign_F1': metrics['benign_f1'],
                'Malicious_F1': metrics['malicious_f1'],
                'AUC': metrics['auc']
            }, ignore_index=True)

    # Process multiclass classification results
    multiclass_df = pd.DataFrame(
        columns=['Dataset', 'Accuracy', 'Weighted_F1'])

    for sample, metrics in multiclass_results.items():
        if 'error' not in metrics:
            multiclass_df = multiclass_df._append({
                'Dataset': sample,
                'Accuracy': metrics['accuracy'],
                'Weighted_F1': metrics['weighted_f1']
            }, ignore_index=True)

    # Save results to CSV
    binary_df.to_csv(
        'svm_results/binary_classification_results.csv', index=False)
    multiclass_df.to_csv(
        'svm_results/multiclass_classification_results.csv', index=False)

    # Save detailed results as JSON
    with open('svm_results/detailed_results.json', 'w') as f:
        json.dump({
            'binary': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv
                           for kk, vv in v.items()}
                       for k, v in binary_results.items() if 'error' not in v},
            'multiclass': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv
                               for kk, vv in v.items()}
                           for k, v in multiclass_results.items() if 'error' not in v}
        }, f, indent=2)

    # Create visualizations

    # 1. Binary classification comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(binary_df))
    width = 0.2

    plt.bar(x - width*1.5, binary_df['Accuracy'], width, label='Accuracy')
    plt.bar(x - width/2, binary_df['Weighted_F1'], width, label='Weighted F1')
    plt.bar(x + width/2, binary_df['Benign_F1'], width, label='Benign F1')
    plt.bar(x + width*1.5, binary_df['AUC'], width, label='AUC')

    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.title('Binary Classification Performance (Malicious/Benign)')
    plt.xticks(x, binary_df['Dataset'])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('svm_results/binary_classification_comparison.png')

    # 2. Multiclass classification comparison
    plt.figure(figsize=(8, 5))
    x = np.arange(len(multiclass_df))
    width = 0.3

    plt.bar(x - width/2, multiclass_df['Accuracy'], width, label='Accuracy')
    plt.bar(x + width/2,
            multiclass_df['Weighted_F1'], width, label='Weighted F1')

    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.title('Multi-class Classification Performance (Traffic Types)')
    plt.xticks(x, multiclass_df['Dataset'])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('svm_results/multiclass_classification_comparison.png')

    print("Results saved to 'svm_results/' directory")
    return 'svm_results/'


def diagnose_perfect_scores(df, X, y_binary, svm_model=None):
    """
    Diagnostic function to investigate why a model might be getting perfect scores

    Args:
        df (pl.DataFrame): Original dataframe
        X (np.ndarray): Feature matrix
        y_binary (np.ndarray): Binary target labels
        svm_model (sklearn model, optional): Trained model to analyze
    """
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances

    print("\n===== DIAGNOSING CLASSIFICATION PERFORMANCE =====")

    # Get indices for each class
    benign_indices = np.where(y_binary == 0)[0]
    malicious_indices = np.where(y_binary == 1)[0]

    print(
        f"Found {len(benign_indices)} benign samples and {len(malicious_indices)} malicious samples")

    # 1. Check model coefficients (if available)
    if svm_model is not None and hasattr(svm_model, 'coef_'):
        print("\n1. FEATURE IMPORTANCE ANALYSIS (TOP 10 FEATURES):")

        # For linear kernel
        importance = np.abs(svm_model.coef_[0])
        feature_names = [col for col in df.columns
                         if col not in ['Label', 'Traffic Type']
                         and df.schema[col] in (pl.Float64, pl.Int64, pl.Float32, pl.Int32)]

        feature_importance = [(feature_names[i], importance[i])
                              for i in range(len(importance))]
        for feat, imp in sorted(feature_importance, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {feat}: {imp:.4f}")
    else:
        print("\n1. FEATURE IMPORTANCE ANALYSIS: Not available for non-linear kernels")

    # 2. Statistical comparison of features between classes
    print("\n2. STATISTICAL COMPARISON OF KEY FEATURES:")

    # Get feature names that are numeric
    numeric_cols = [col for col in df.columns
                    if col not in ['Label', 'Traffic Type']
                    and df.schema[col] in (pl.Float64, pl.Int64, pl.Float32, pl.Int32)]

    # Compare statistics using the indices rather than filtering the DataFrame
    top_diffs = []

    # Look at first 30 columns
    for col_idx, col in enumerate(numeric_cols[:30]):
        try:
            if col_idx < X.shape[1]:
                benign_vals = X[benign_indices, col_idx]
                # Sample malicious for efficiency
                mal_sample_indices = np.random.choice(
                    malicious_indices, min(100, len(malicious_indices)), replace=False)
                malicious_vals = X[mal_sample_indices, col_idx]

                benign_mean = np.mean(benign_vals)
                malicious_mean = np.mean(malicious_vals)

                # Calculate normalized difference
                if abs(benign_mean) > 0.0001 or abs(malicious_mean) > 0.0001:
                    norm_diff = abs(benign_mean - malicious_mean) / \
                        max(abs(benign_mean), abs(malicious_mean), 1.0)
                else:
                    norm_diff = 0

                top_diffs.append((col, norm_diff, benign_mean, malicious_mean))
        except Exception as e:
            print(f"  Error analyzing feature {col}: {e}")

    # Show features with largest differences
    print("  Features with largest differences between benign and malicious:")
    for col, diff, b_mean, m_mean in sorted(top_diffs, key=lambda x: x[1], reverse=True)[:10]:
        print(
            f"  {col}: Benign mean={b_mean:.4f}, Malicious mean={m_mean:.4f}, Diff={diff:.4f}")

    # 3. Check for perfect separators using X directly
    print("\n3. CHECKING FOR POTENTIAL SEPARATOR FEATURES:")

    # Check a sample of features
    # Look at first 30 columns
    for col_idx, col in enumerate(numeric_cols[:30]):
        try:
            if col_idx < X.shape[1]:
                benign_vals = X[benign_indices, col_idx]
                malicious_vals = X[np.random.choice(malicious_indices, min(
                    1000, len(malicious_indices)), replace=False), col_idx]

                benign_min = np.min(benign_vals)
                benign_max = np.max(benign_vals)
                malicious_min = np.min(malicious_vals)
                malicious_max = np.max(malicious_vals)

                # Check for non-overlapping ranges
                if benign_max < malicious_min or benign_min > malicious_max:
                    print(
                        f"  {col} potentially separates benign from malicious:")
                    print(
                        f"    Benign range: {benign_min:.4f} to {benign_max:.4f}")
                    print(
                        f"    Malicious range: {malicious_min:.4f} to {malicious_max:.4f}")
        except Exception as e:
            print(f"  Error checking feature {col}: {e}")

    # 4. Check for duplicates
    print("\n4. CHECKING FOR DUPLICATE OR NEAR-DUPLICATE SAMPLES:")
    if len(benign_indices) <= 1:
        print("  Only one benign sample, can't check for duplicates")
    else:
        # Extract feature values for benign samples
        benign_data = X[benign_indices]

        # Calculate distances between benign samples
        try:
            dists = euclidean_distances(benign_data)
            np.fill_diagonal(dists, np.inf)  # Ignore self-comparisons
            min_dists = np.min(dists, axis=1)

            print(
                f"  Min distance between benign samples: {np.min(min_dists):.4f}")
            print(
                f"  Max distance between benign samples: {np.max(min_dists):.4f}")
            print(
                f"  Avg distance between benign samples: {np.mean(min_dists):.4f}")

            # Compare to random malicious samples
            if len(malicious_indices) > 1:
                sample_size = min(len(malicious_indices), len(benign_indices))
                mal_sample_indices = np.random.choice(
                    malicious_indices, sample_size, replace=False)
                mal_data = X[mal_sample_indices]

                mal_dists = euclidean_distances(mal_data)
                np.fill_diagonal(mal_dists, np.inf)
                mal_min_dists = np.min(mal_dists, axis=1)

                print(
                    f"  For comparison - avg distance between random malicious samples: {np.mean(mal_min_dists):.4f}")

                # Compare benign to malicious
                cross_dists = euclidean_distances(benign_data, mal_data)
                min_cross_dists = np.min(cross_dists, axis=1)

                print(
                    f"  Min distance between benign and malicious samples: {np.min(min_cross_dists):.4f}")
                print(
                    f"  Avg distance between benign and malicious samples: {np.mean(cross_dists):.4f}")

                if np.min(cross_dists) > np.max(min_dists):
                    print(
                        "  INSIGHT: Benign samples are closer to each other than to any malicious sample")
                    print("  This suggests benign traffic forms a distinct cluster")
        except Exception as e:
            print(f"  Error in distance calculation: {e}")

    print("\n===== DIAGNOSIS COMPLETE =====\n")

# At the end of each evaluate_* function, add:


if __name__ == "__main__":
    # Define data sample paths
    data_samples = {
        'original': "stratified_sampling_files/sampled_data_10k",
        # 'kmeans': "kmeans_sampled_data",
        # 'hdbscan': "hdbscan_files/hdbscan_sampled_data"
    }

    # Run binary classification (malicious/benign)
    print("\n===== EVALUATING BINARY CLASSIFICATION (MALICIOUS/BENIGN) =====")
    binary_results = evaluate_binary_classification(data_samples)

    # # Run multi-class classification (traffic types)
    # print("\n===== EVALUATING MULTI-CLASS CLASSIFICATION (TRAFFIC TYPES) =====")
    # multiclass_results = evaluate_multiclass_classification(data_samples)

    # # Save and visualize results
    # save_and_visualize_results(binary_results, multiclass_results)

    # print("\nEvaluation completed.")
