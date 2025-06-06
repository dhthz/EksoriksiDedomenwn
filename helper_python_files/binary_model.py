import os
import csv
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# This function builds a binary classification neural network
def build_binary_neural_network(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),

        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )

    return model

# Function to train and evaluate neural network on a dataset


def train_binary_neural_network(X_train, y_train, X_test, y_test, dataset_name, epochs=100):
    print(
        f"\n===== STRATIFIED 5-FOLD CROSS-VALIDATION ON {dataset_name.upper()} =====")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Store metrics for each fold
    fold_metrics = []
    fold_histories = []
    fold_val_gaps = []

    # Perform 5-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n----- Fold {fold+1}/5 -----")

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Get input shape from training data
        input_shape = X_train_fold.shape[1]

        model = build_binary_neural_network(input_shape)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_recall',
            patience=25,
            min_delta=0.001,
            mode='max',
            restore_best_weights=True
        )

        lr_reduction = keras.callbacks.ReduceLROnPlateau(
            monitor='val_recall',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            mode='max',
            cooldown=3,
            verbose=1
        )

        # Calculate class weights
        class_counts = np.bincount(y_train_fold.astype(int))
        total_samples = len(y_train_fold)
        class_weights = {
            0: total_samples / (2 * class_counts[0]),
            1: total_samples / (2 * class_counts[1])
        }

        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=200,
            batch_size=32,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stopping, lr_reduction],
            class_weight=class_weights,
            verbose=1
        )

        model_evaluation = model.evaluate(X_val_fold, y_val_fold, verbose=0)

        # Calculate generalization metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        val_gap = final_train_acc - final_val_acc

        fold_metrics.append(model_evaluation)
        fold_histories.append(history.history)
        fold_val_gaps.append(val_gap)

    # Calculate average metrics across all folds
    fold_metrics = np.array(fold_metrics)
    avg_metrics = np.mean(fold_metrics, axis=0)
    std_metrics = np.std(fold_metrics, axis=0)

    # Calculate average generalization gap
    avg_val_gap = np.mean(fold_val_gaps)
    std_val_gap = np.std(fold_val_gaps)

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = (y_pred > 0.5).astype(int)

    print("\n===== CROSS-VALIDATION SUMMARY =====")
    print(f"Average Loss: {avg_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"Average Accuracy: {avg_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"Average Precision: {avg_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    print(f"Average Recall: {avg_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
    print(f"Average AUC: {avg_metrics[4]:.4f} ± {std_metrics[4]:.4f}")
    print(f"Average Generalization Gap: {avg_val_gap:.4f} ± {std_val_gap:.4f}")

    evaluation_dir = "binary_model"
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)

    # Save evaluation metrics to CSV
    csv_path = f"{evaluation_dir}/{dataset_name}_sample_binary_model.csv"
    # Save average results
    with open(f"{csv_path}", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Average', 'Std Dev'])
        writer.writerow(['Loss', avg_metrics[0], std_metrics[0]])
        writer.writerow(['Accuracy', avg_metrics[1], std_metrics[1]])
        writer.writerow(['Precision', avg_metrics[2], std_metrics[2]])
        writer.writerow(['Recall', avg_metrics[3], std_metrics[3]])
        writer.writerow(['AUC', avg_metrics[4], std_metrics[4]])
        writer.writerow(['Generalization Gap', avg_val_gap, std_val_gap])

    print(f"Evaluation metrics saved to {csv_path}")

    # Plot training history
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy - {dataset_name}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss - {dataset_name}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{evaluation_dir}/training_history_{dataset_name}_sample.png')

    # Create and save confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{evaluation_dir}/confusion_matrix_{dataset_name}.png')

    return model, history

# Function for setting up feature columns and preprocessor


def setup_preprocessor():
    df = pd.read_csv('sampled_data/stratified_sampled_data.csv')
    # 1. Features to remove (constant columns)
    constant_cols = ['Bwd URG Flags', 'Bwd PSH Flags', 'Fwd Bytes/Bulk Avg',
                     'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg']

    # 2. Binary features (stayes as is)
    binary_cols = ['Fwd URG Flags', 'Fwd PSH Flags']

    # 3. Categorical features for one-hot encoding
    categorical_cols = ['Protocol', 'Down/Up Ratio',
                        'RST Flag Count', 'FIN Flag Count']

    # 4. Flag features for ordinal encoding
    flag_cols_ordinal = ['SYN Flag Count', 'CWR Flag Count', 'ECE Flag Count']

    # 5. Port columns for special encoding
    port_cols = ['Src Port', 'Dst Port']

    # 6. All remaining numerical features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Create the preprocessor with StandardScaler for all numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            # All numerical features with standard scaling
            ('numerical', StandardScaler(), numeric_cols),

            # Flag features with ordinal encoding
            ('flag_ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
             flag_cols_ordinal),

            # Port columns
            ('port', FunctionTransformer(encode_port_categories), port_cols),

            # Categorical features
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
             categorical_cols)
        ],
        remainder='passthrough'  # Keep binary features as is
    )

    return preprocessor, constant_cols


def encode_port_categories(X):
    # Convert to DataFrame if not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=['Src Port', 'Dst Port'])

    # Create new columns
    result = X.copy()
    for col in ['Src Port', 'Dst Port']:
        # Well-known ports (0-1023)
        result[f'{col}_well_known'] = (result[col] <= 1023).astype(int)

        # Registered ports (1024-49151)
        result[f'{col}_registered'] = (
            (result[col] > 1023) & (result[col] <= 49151)).astype(int)

        # Dynamic/private ports (49152-65535)
        result[f'{col}_dynamic'] = (result[col] > 49151).astype(int)

        # Common service ports (add more as needed)
        result[f'{col}_web'] = result[col].isin(
            [80, 443, 8080, 8443]).astype(int)
        result[f'{col}_email'] = result[col].isin(
            [25, 465, 587, 110, 143, 993, 995]).astype(int)
        result[f'{col}_file_transfer'] = result[col].isin(
            [20, 21, 22, 69]).astype(int)
        result[f'{col}_dns'] = result[col].isin([53]).astype(int)

    # Drop original port columns
    return result.drop(['Src Port', 'Dst Port'], axis=1)


def process_stratified_sample_for_binay_classification():
    # Setup preprocessor
    preprocessor, constant_cols = setup_preprocessor()

    df = pd.read_csv('sampled_data/stratified_sampled_data.csv')

    # Remove constant columns
    df = df.drop(columns=constant_cols, errors='ignore')

    X_stratified = df.drop(['Label', 'Traffic Type'], axis=1)
    y_stratified = df['Label']

    # Convert string labels to numeric (crucial step)
    # 'Malicious' and 'Benign' maped them to 1 and 0
    if y_stratified.dtype == 'object':
        label_map = {'Malicious': 1, 'Benign': 0}
        y_stratified = y_stratified.map(label_map)

    # Splitting 90% for training and 10% for testing
    X_train_stratified = X_stratified.sample(frac=0.9, random_state=42)
    X_test_stratified = X_stratified.drop(X_train_stratified.index)
    y_train_stratified = y_stratified.loc[X_train_stratified.index]
    y_test_stratified = y_stratified.loc[X_test_stratified.index]

    # Preprocess the data
    X_train_processed = preprocessor.fit_transform(X_train_stratified)
    X_test_processed = preprocessor.transform(X_test_stratified)

    # Convert to numpy arrays with float32 dtype (crucial for TensorFlow)
    X_train_processed = np.asarray(X_train_processed).astype('float32')
    X_test_processed = np.asarray(X_test_processed).astype('float32')
    y_train_processed = np.asarray(y_train_stratified).astype('float32')
    y_test_processed = np.asarray(y_test_stratified).astype('float32')

    # Check for any remaining non-numeric values
    if np.any(np.isnan(X_train_processed)) or np.any(np.isinf(X_train_processed)):
        print("Warning: NaN or Inf values found in training data. Replacing with zeros.")
        X_train_processed = np.nan_to_num(X_train_processed)

    if np.any(np.isnan(X_test_processed)) or np.any(np.isinf(X_test_processed)):
        print("Warning: NaN or Inf values found in test data. Replacing with zeros.")
        X_test_processed = np.nan_to_num(X_test_processed)

    processed_data = {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train_processed,
        'y_test': y_test_processed
    }

    print(f"\nTraining neural network with data shapes:")
    print(
        f"X_train: {processed_data['X_train'].shape}, dtype: {processed_data['X_train'].dtype}")
    print(
        f"y_train: {processed_data['y_train'].shape}, dtype: {processed_data['y_train'].dtype}")

    train_binary_neural_network(
        processed_data['X_train'],
        processed_data['y_train'],
        processed_data['X_test'],
        processed_data['y_test'],
        'stratified'
    )

    return preprocessor, constant_cols


# Process other Clustering samples using the fitted preprocessor
def process_clustering_samples_for_binary_classification(preprocessor, constant_cols):
    other_datasets = {
        'kmeans': pd.read_csv('sampled_data/sampled_data_kmeans_sample_data.csv'),
        'hdbscan': pd.read_csv('sampled_data/hdbscan_sampled_data.csv')
    }

    for name, df in other_datasets.items():
        # Remove constant columns
        df = df.drop(columns=constant_cols, errors='ignore')

        X = df.drop('Label', axis=1)
        y = df['Label']

        # Convert string labels to numeric
        # 'Malicious' and 'Benign', maped them to 1 and 0
        if y.dtype == 'object':
            print("\nConverting string labels to numeric values...")
            label_map = {'Malicious': 1, 'Benign': 0}
            y = y.map(label_map)

        # Display class distribution
        print(f"\nClass distribution in {name} dataset:")
        print(
            f"  Class 'Benign': {sum(y == 0)} samples ({sum(y == 0)/len(y)*100:.2f}%)")
        print(
            f"  Class 'Malicious': {sum(y == 1)} samples ({sum(y == 1)/len(y)*100:.2f}%)")

        # Split data manually
        X_train = X.sample(frac=0.9, random_state=42)
        X_test = X.drop(X_train.index)
        y_train = y.loc[X_train.index]
        y_test = y.loc[X_test.index]

        # Use the ALREADY FITTED preprocessor
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Convert to numpy arrays with float32 dtype (crucial for TensorFlow)
        X_train_processed = np.asarray(X_train_processed).astype('float32')
        X_test_processed = np.asarray(X_test_processed).astype('float32')
        y_train_processed = np.asarray(y_train).astype('float32')
        y_test_processed = np.asarray(y_test).astype('float32')

        # Check for any remaining non-numeric values
        if np.any(np.isnan(X_train_processed)) or np.any(np.isinf(X_train_processed)):
            print(
                f"Warning: NaN or Inf values found in {name} training data. Replacing with zeros.")
            X_train_processed = np.nan_to_num(X_train_processed)

        if np.any(np.isnan(X_test_processed)) or np.any(np.isinf(X_test_processed)):
            print(
                f"Warning: NaN or Inf values found in {name} test data. Replacing with zeros.")
            X_test_processed = np.nan_to_num(X_test_processed)

        processed_data = {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train_processed,
            'y_test': y_test_processed
        }

        print(f"\nTraining neural network with data shapes:")
        print(
            f"X_train: {processed_data['X_train'].shape}, dtype: {processed_data['X_train'].dtype}")
        print(
            f"y_train: {processed_data['y_train'].shape}, dtype: {processed_data['y_train'].dtype}")

        try:
            train_binary_neural_network(
                processed_data['X_train'],
                processed_data['y_train'],
                processed_data['X_test'],
                processed_data['y_test'],
                name
            )

        except Exception as e:
            print(f"\nError training neural network: {str(e)}")
            # Return what we have even if neural network training fails

    return


def binary_nn():
    # Process stratified sample first and get fitted preprocessor
    preprocessor, constant_cols = process_stratified_sample_for_binay_classification()

    # Process Clustering samples using the fitted preprocessor
    process_clustering_samples_for_binary_classification(
        preprocessor, constant_cols)

    return
