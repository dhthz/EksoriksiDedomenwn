import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from paradoteo_A1 import load_data_from_csv_parquet_format
import warnings  # To remove some warnings that dont cause an error
warnings.filterwarnings('ignore')


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
    # if svm_model is not None:
        # diagnose_scores(df, X, y_binary, svm_model)
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
            class_distribution = {}
            rare_classes = []

            print("Class distribution:")
            for cls, count in zip(unique_classes, counts):
                class_pct = count/len(y)*100
                print(f"  {cls}: {count} samples ({class_pct:.2f}%)")
                class_distribution[str(cls)] = float(class_pct)

                if count <= 2:  # Consider classes with ≤3 samples as rare
                    rare_classes.append(cls)

            # Handle rare classes specially
            if rare_classes:
                print(
                    f"Found {len(rare_classes)} rare classes with ≤3 samples")

                # Store indices of rare class samples
                rare_indices = np.array([], dtype=int)
                for cls in rare_classes:
                    cls_indices = np.where(y == cls)[0]
                    rare_indices = np.append(rare_indices, cls_indices)

                if len(rare_indices) > 0:
                    # Split non-rare samples normally
                    non_rare_mask = ~np.isin(np.arange(len(y)), rare_indices)
                    X_common, X_rare = X[non_rare_mask], X[rare_indices]
                    y_common, y_rare = y[non_rare_mask], y[rare_indices]

                    # Split common classes
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_common, y_common, test_size=0.25, random_state=42, stratify=y_common
                    )

                    # Split rare classes, ensuring at least one in both training and test sets
                    X_rare_train, X_rare_test = [], []
                    y_rare_train, y_rare_test = [], []

                    # For each rare class, divide samples between train and test
                    for cls in rare_classes:
                        cls_indices = np.where(y_rare == cls)[0]
                        n_samples = len(cls_indices)

                        if n_samples == 1:
                            # If only one sample, add to training
                            X_rare_train.append(X_rare[cls_indices[0]])
                            y_rare_train.append(cls)
                        elif n_samples == 2:
                            # If two samples, add one to training, one to test
                            X_rare_train.append(X_rare[cls_indices[0]])
                            y_rare_train.append(cls)
                            X_rare_test.append(X_rare[cls_indices[1]])
                            y_rare_test.append(cls)
                        else:
                            # If more than two, split 70/30
                            n_train = max(1, int(n_samples * 0.7))
                            train_indices = cls_indices[:n_train]
                            test_indices = cls_indices[n_train:]

                            for idx in train_indices:
                                X_rare_train.append(X_rare[idx])
                                y_rare_train.append(cls)

                            for idx in test_indices:
                                X_rare_test.append(X_rare[idx])
                                y_rare_test.append(cls)

                    # Convert lists to arrays
                    if X_rare_train:
                        X_rare_train = np.array(X_rare_train)
                        y_rare_train = np.array(y_rare_train)
                        # Add rare classes to training set
                        X_train = np.vstack([X_train, X_rare_train])
                        y_train = np.append(y_train, y_rare_train)

                    if X_rare_test:
                        X_rare_test = np.array(X_rare_test)
                        y_rare_test = np.array(y_rare_test)
                        # Add rare classes to test set
                        X_test = np.vstack([X_test, X_rare_test])
                        y_test = np.append(y_test, y_rare_test)

                    print(f"Added rare samples to both training and test sets")
                    print(
                        f"  Training: {len(y_rare_train)} samples from {len(np.unique(y_rare_train))} rare classes")
                    print(
                        f"  Testing: {len(y_rare_test)} samples from {len(np.unique(y_rare_test))} rare classes")
            else:
                # Standard split if no rare classes
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

            # Check which classes are in the test set
            test_classes = np.unique(y_test)
            train_classes = np.unique(y_train)
            print(
                f"Classes in training set: {len(train_classes)} out of {len(unique_classes)}")
            print(
                f"Classes in test set: {len(test_classes)} out of {len(unique_classes)}")

            # Print a warning if some classes aren't in test set
            missing_classes = set(unique_classes) - set(test_classes)
            if missing_classes:
                print(
                    f"Warning: {len(missing_classes)} classes not in test set: {missing_classes}")

            # Calculate overall metrics
            accuracy = accuracy_score(y_test, y_pred)
            weighted_precision = precision_score(
                y_test, y_pred, average='weighted', zero_division=0)
            weighted_recall = recall_score(
                y_test, y_pred, average='weighted', zero_division=0)
            weighted_f1 = f1_score(
                y_test, y_pred, average='weighted', zero_division=0)

            # Calculate per-class metrics
            report = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0)

            # Store class-specific metrics
            class_metrics = {}
            for cls in test_classes:
                cls_str = str(cls)
                if cls_str in report:
                    class_metrics[cls_str] = {
                        'precision': report[cls_str]['precision'],
                        'recall': report[cls_str]['recall'],
                        'f1-score': report[cls_str]['f1-score'],
                        'support': report[cls_str]['support']
                    }

            # Add training-only class information
            for cls in train_classes:
                if cls not in test_classes:
                    cls_str = str(cls)
                    class_metrics[cls_str] = {
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0,
                        'training_only': True,
                        'training_samples': np.sum(y_train == cls)
                    }

            # Store results
            results[sample_name] = {
                'accuracy': accuracy,
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1': weighted_f1,
                'class_distribution': class_distribution,
                'class_metrics': class_metrics,
                'rare_classes': [str(cls) for cls in rare_classes]
            }

            # Print confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            print("\nConfusion Matrix:")
            print(conf_matrix)

            # Optionally, create a more readable confusion matrix with labels
            if len(test_classes) < 10:  # Only for a reasonable number of classes
                try:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(conf_matrix, interpolation='nearest',
                               cmap=plt.cm.Blues)
                    plt.title(f"Confusion Matrix - {sample_name}")
                    plt.colorbar()
                    tick_marks = np.arange(len(test_classes))
                    plt.xticks(tick_marks, test_classes, rotation=45)
                    plt.yticks(tick_marks, test_classes)

                    # Add text annotations
                    thresh = conf_matrix.max() / 2
                    for i in range(conf_matrix.shape[0]):
                        for j in range(conf_matrix.shape[1]):
                            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                                     ha="center", va="center",
                                     color="white" if conf_matrix[i, j] > thresh else "black")

                    plt.tight_layout()
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')

                    # Save the confusion matrix
                    os.makedirs('svm_results', exist_ok=True)
                    plt.savefig(
                        f'svm_results/confusion_matrix_{sample_name}.png')
                    plt.close()
                except Exception as e:
                    print(
                        f"Error creating confusion matrix visualization: {e}")

            # Print classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))

            # Print key metrics
            print(f"Overall accuracy: {accuracy:.4f}")
            print(f"Weighted precision: {weighted_precision:.4f}")
            print(f"Weighted recall: {weighted_recall:.4f}")
            print(f"Weighted F1 score: {weighted_f1:.4f}")

            # Print top/bottom 3 classes by F1 score (if we have many classes)
            if len(test_classes) > 6:
                print("\nTop performing classes (by F1 score):")
                class_f1_scores = [(cls, report[cls]['f1-score'], report[cls]['support'])
                                   for cls in report if cls not in ['accuracy', 'macro avg', 'weighted avg']]

                # Top classes
                for cls, f1, support in sorted(class_f1_scores, key=lambda x: x[1], reverse=True)[:3]:
                    print(f"  {cls}: F1={f1:.4f}, Support={support}")

                # Bottom classes
                print("\nBottom performing classes (by F1 score):")
                for cls, f1, support in sorted(class_f1_scores, key=lambda x: x[1])[:3]:
                    print(f"  {cls}: F1={f1:.4f}, Support={support}")

        except Exception as e:
            import traceback
            print(f"Error processing {sample_name} dataset: {e}")
            print(traceback.format_exc())
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


def diagnose_scores(df, X, y_binary, svm_model=None):
    """
    Diagnostic function to investigate why a model might be getting perfect scores

    Args:
        df (pl.DataFrame): Original dataframe
        X (np.ndarray): Feature matrix
        y_binary (np.ndarray): Binary target labels
        svm_model (sklearn model, optional): Trained model to analyze
    """

    print("\n===== DIAGNOSING CLASSIFICATION PERFORMANCE =====")

    # Get indices for each class
    benign_indices = np.where(y_binary == 0)[0]
    malicious_indices = np.where(y_binary == 1)[0]

    print(
        f"Found {len(benign_indices)} benign samples and {len(malicious_indices)} malicious samples")

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

                if np.min(cross_dists) > np.max(min_dists):  # TODO na bgoun ta siberasmata
                    print(
                        "  INSIGHT: Benign samples are closer to each other than to any malicious sample")
                    print("  This suggests benign traffic forms a distinct cluster")
        except Exception as e:
            print(f"  Error in distance calculation: {e}")

    print("\n===== DIAGNOSIS COMPLETE =====\n")


def svm():
    # Define data sample paths
    data_samples = {
        'stratified_sampling': "sampled_data/stratified_sampled_data",
        'kmeans': "sampled_data/kmeans_sampled_data",
        'hdbscan': "sampled_data/hdbscan_sampled_data"
    }

    # Run binary classification (malicious/benign)
    print("\n===== EVALUATING BINARY CLASSIFICATION (MALICIOUS/BENIGN) =====")
    binary_results = evaluate_binary_classification(data_samples)

    # Run multi-class classification (traffic types)
    print("\n===== EVALUATING MULTI-CLASS CLASSIFICATION (TRAFFIC TYPES) =====")
    multiclass_results = evaluate_multiclass_classification(data_samples)

    # Save and visualize results
    save_and_visualize_results(binary_results, multiclass_results)

    print("\nEvaluation completed.")

    return