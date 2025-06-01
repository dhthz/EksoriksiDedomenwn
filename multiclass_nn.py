import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, RobustScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score

def build_multiclass_neural_network(input_shape, num_classes):
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_shape,)),
        
        # First hidden layer
        layers.Dense(128, activation='elu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second hidden layer
        layers.Dense(64, activation='elu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Third hidden layer
        layers.Dense(32, activation='elu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        # Output layer for multiclass classification
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )
    
    return model

# Function to train and evaluate multiclass neural network
def train_multiclass_neural_network(X_train, y_train, X_test, y_test, dataset_name, class_names, epochs=200):
    print(f"\n===== MULTICLASS TRAFFIC TYPE CLASSIFICATION ON {dataset_name.upper()} =====")
    
    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store metrics for each fold
    fold_metrics = []
    fold_histories = []
    fold_val_gaps = []
    fold_generalization_metrics = []

    # Perform k-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n----- Fold {fold+1}/5 -----")
        
        # Split data for this fold
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        # Get input shape and number of classes
        input_shape = X_train_fold.shape[1]
        num_classes = len(class_names)
        
        # Build the neural network
        model = build_multiclass_neural_network(input_shape, num_classes)
        
        # Set up early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_recall',     # Monitor recall instead of F1
            patience=25,              # More patient
            min_delta=0.001,          # Require meaningful improvement
            mode='max',               # Maximize recall
            restore_best_weights=True
        )
        
        # Set up learning rate reduction - RECALL FOCUSED
        lr_reduction = keras.callbacks.ReduceLROnPlateau(
            monitor='val_recall',     # Monitor recall for LR reduction too
            factor=0.5,               # Moderate reduction (50%)
            patience=10,              # Patient with LR reduction
            min_lr=1e-7,             # Allow very small learning rates
            mode='max',              # Maximize recall
            cooldown=3,              # Wait 3 epochs after reduction
            verbose=1
        )

        # Train the model
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stopping, lr_reduction],
            verbose=1
        )
    
        # Evaluate the model
        model_evaluation = model.evaluate(X_val_fold, y_val_fold, verbose=0)

        # Calculate generalization metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        val_gap = final_train_acc - final_val_acc
    
        # Store results
        fold_metrics.append(model_evaluation)
        fold_histories.append(history.history)
        fold_val_gaps.append(val_gap)

        # Compile generalization metrics for this fold
        # fold_gen = {
        #     'train_acc': final_train_acc,
        #     'val_acc': final_val_acc,
        #     'val_gap': val_gap,
        #     'loss': model_evaluation[0],
        #     'accuracy': model_evaluation[1]
        # }
        # fold_generalization_metrics.append(fold_gen)
    
    # Calculate average metrics across all folds
    fold_metrics = np.array(fold_metrics)
    avg_metrics = np.mean(fold_metrics, axis=0)
    std_metrics = np.std(fold_metrics, axis=0)
    
    # Calculate average generalization gap
    avg_val_gap = np.mean(fold_val_gaps)
    std_val_gap = np.std(fold_val_gaps)
    
    # Print evaluation metrics
    print("\n===== MULTICLASS CROSS-VALIDATION SUMMARY =====")
    print(f"Average Loss: {avg_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"Average Accuracy: {avg_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"Average Generalization Gap: {avg_val_gap:.4f} ± {std_val_gap:.4f}")
    
    # Generate predictions for detailed evaluation
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred_classes, average='macro')
    recall = recall_score(y_test, y_pred_classes, average='macro')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Print detailed classification report
    print("\n===== DETAILED CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))
    
    # Create directory for saving results if it doesn't exist
    evaluation_dir = "multiclass_nn_evaluation_and_plots"
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    
    # Save evaluation metrics to CSV
    csv_path = f"{evaluation_dir}/multiclass_{dataset_name}_evaluation.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Average', 'Std Dev'])
        writer.writerow(['Loss', avg_metrics[0], std_metrics[0]])
        writer.writerow(['Accuracy', avg_metrics[1], std_metrics[1]])
        writer.writerow(['Precision', precision, 'N/A'])
        writer.writerow(['Recall', recall, 'N/A'])
        writer.writerow(['Generalization Gap', avg_val_gap, std_val_gap])
    
    print(f"Evaluation metrics saved to {csv_path}")
        
    # # Figure 1: Training and validation metrics
    # plt.figure(figsize=(12, 6))
    
    # # Plot training & validation accuracy
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title(f'Model Accuracy - {dataset_name}')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='lower right')
    
    # # Plot training & validation loss
    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title(f'Model Loss - {dataset_name}')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper right')
    
    # plt.tight_layout()
    # plt.savefig(f'{evaluation_dir}/multiclass_training_metrics_{dataset_name}.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    # # Figure 2: Confusion matrix
    # plt.figure(figsize=(10, 8))
    # cm = confusion_matrix(y_test, y_pred_classes)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
    #             xticklabels=class_names, yticklabels=class_names)
    # plt.title(f'Confusion Matrix - {dataset_name}')
    # plt.ylabel('True Label')
    # plt.xlabel('Predicted Label')
    # plt.tight_layout()
    # plt.savefig(f'{evaluation_dir}/multiclass_confusion_matrix_{dataset_name}.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    # # Figure 3: Class distribution
    # plt.figure(figsize=(10, 6))
    # unique, counts = np.unique(y_test, return_counts=True)
    # plt.bar([class_names[i] for i in unique], counts)
    # plt.title(f'Test Set Class Distribution - {dataset_name}')
    # plt.xlabel('Traffic Type')
    # plt.ylabel('Count')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(f'{evaluation_dir}/multiclass_class_distribution_{dataset_name}.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    return model, history

# Function for setting up feature columns and preprocessor
def setup_preprocessor():    
    # Constant columns (same)
    constant_cols = ['Bwd URG Flags', 'Bwd PSH Flags', 'Fwd Bytes/Bulk Avg', 
                    'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg']

    # Group features by characteristics for multiclass
    
    # 1. Flow timing features (often crucial for traffic type)
    timing_features = [
        'Flow Duration', 'Flow Packets/s', 'Flow Bytes/s',
        'Fwd IAT Std', 'Bwd IAT Max', 'Idle Std'
    ]
    
    # 2. Packet size features (important for multiclass)
    size_features = [
        'Packet Length Std', 'Fwd Packet Length Std', 'Average Packet Size',
        'Bwd Packet Length Min', 'Bwd Packet Length Max', 'Fwd Seg Size Min'
    ]
    
    # 3. Protocol behavior features
    protocol_features = [
        'Subflow Fwd Packets', 'Subflow Bwd Packets', 
        'Subflow Fwd Bytes', 'Subflow Bwd Bytes',
        'Bwd Packets/s', 'Active Std', 'Active Max'
    ]
    
    # 4. Window and connection features
    connection_features = [
        'Bwd Init Win Bytes', 'FWD Init Win Bytes',
        'Bwd Bulk Rate Avg'
    ]
    
    # 5. Flag features - use ordinal encoding but group better
    flag_features = ['SYN Flag Count', 'CWR Flag Count', 'ECE Flag Count', 'ACK Flag Count']
    categorical_features = ['Protocol', 'RST Flag Count', 'FIN Flag Count']
    
    # 6. Ratio features (often important for traffic classification)
    ratio_features = ['Down/Up Ratio']
    
    # Port columns
    port_cols = ['Src Port', 'Dst Port']

    # Create multiclass-optimized preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            # Timing features - use StandardScaler for better multiclass separation
            ('timing', Pipeline([
                ('power', PowerTransformer(method='yeo-johnson')),
                ('scaler', StandardScaler())  # Changed from RobustScaler
            ]), timing_features),
            
            # Size features - normalize for better class boundaries
            ('size', Pipeline([
                ('scaler', StandardScaler())
            ]), size_features),
            
            # Protocol features - robust scaling
            ('protocol', RobustScaler(), protocol_features),
            
            # Connection features
            ('connection', StandardScaler(), connection_features),
            
            # Flag features - ordinal encoding
            ('flags', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), 
             flag_features),
            
            # Categorical features - one-hot
            ('categorical', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), 
             categorical_features),
            
            # Ratio features - special handling
            ('ratio', Pipeline([
                ('clip', FunctionTransformer(lambda x: np.clip(x, 0, 10))),  # Clip extreme ratios
                ('scaler', StandardScaler())
            ]), ratio_features),
            
            # Port columns - simplified encoding
            ('ports', FunctionTransformer(encode_port_categories_multiclass), port_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor, constant_cols

def encode_port_categories_multiclass(X):
    """Simplified port encoding for multiclass"""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=['Src Port', 'Dst Port'])
    
    result = X.copy()
    for col in ['Src Port', 'Dst Port']:
        # Simpler categorization
        result[f'{col}_category'] = pd.cut(
            result[col], 
            bins=[0, 1023, 49151, 65535], 
            labels=[0, 1, 2],  # well-known, registered, dynamic
            include_lowest=True
        ).astype(int)
        
        # Only most important service flags
        result[f'{col}_is_web'] = result[col].isin([80, 443, 8080]).astype(int)
        result[f'{col}_is_secure'] = result[col].isin([443, 22, 993, 995]).astype(int)
    
    return result.drop(['Src Port', 'Dst Port'], axis=1)

# Apply multiclass resampling to balance classes
def apply_multiclass_resampling(X, y, class_names):
    """
    Apply resampling techniques to balance multiclass data.
    
    Parameters:
    - X: Features DataFrame/array
    - y: Target labels (encoded as integers)
    - class_names: List of class names for display
    
    Returns:
    - X_resampled, y_resampled: Balanced dataset
    """
    # Check initial class distribution
    counter_before = Counter(y)
    print(f"\nOriginal multiclass distribution:")
    for i, class_name in enumerate(class_names):
        count = counter_before.get(i, 0)
        percentage = count / len(y) * 100 if len(y) > 0 else 0
        print(f"  {class_name}: {count} samples ({percentage:.2f}%)")
    
    # Calculate target sample size - use a more reasonable target
    class_counts = list(counter_before.values())
    # Use mean of non-extreme classes or set a fixed reasonable target
    target_size = max(50, int(np.mean([c for c in class_counts if c > 20 and c < 1000])))
    
    print(f"\nTarget size per class: {target_size}")
    
    # Step 1: Undersample ALL classes to target size first
    undersample_strategy = {}
    for class_id, count in counter_before.items():
        if count > target_size:
            undersample_strategy[class_id] = target_size
    
    if undersample_strategy:
        print(f"\nStep 1: Undersampling classes with strategy: {undersample_strategy}")
        undersampler = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
        X_under, y_under = undersampler.fit_resample(X, y)
        
        # Check intermediate distribution
        counter_under = Counter(y_under)
        print("After undersampling:")
        for i, class_name in enumerate(class_names):
            count = counter_under.get(i, 0)
            percentage = count / len(y_under) * 100 if len(y_under) > 0 else 0
            print(f"  {class_name}: {count} samples ({percentage:.2f}%)")
    else:
        print("\nNo undersampling needed.")
        X_under, y_under = X, y
        counter_under = counter_before
    
    # Step 2: Apply SMOTE to bring ALL classes to exactly target_size
    print(f"\nStep 2: Applying SMOTE to balance all classes to exactly {target_size} samples each...")
    
    # Create sampling strategy for SMOTE (bring ALL classes to target size)
    smote_strategy = {}
    for class_id in np.unique(y_under):
        current_count = counter_under.get(class_id, 0)
        if current_count != target_size:  # Changed condition to ensure exact balance
            smote_strategy[class_id] = target_size
    
    if smote_strategy:
        print(f"SMOTE strategy: {smote_strategy}")
        
        # Use SMOTE with adjusted k_neighbors based on smallest class
        min_samples = min(counter_under.values())
        k_neighbors = min(5, max(1, min_samples - 1))
        
        try:
            smote = SMOTE(sampling_strategy=smote_strategy, random_state=42, k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X_under, y_under)
        except ValueError as e:
            print(f"SMOTE failed with k_neighbors={k_neighbors}: {e}")
            print("Trying with k_neighbors=1...")
            try:
                smote = SMOTE(sampling_strategy=smote_strategy, random_state=42, k_neighbors=1)
                X_resampled, y_resampled = smote.fit_resample(X_under, y_under)
            except ValueError as e2:
                print(f"SMOTE failed completely: {e2}")
                print("Using original data without SMOTE...")
                X_resampled, y_resampled = X_under, y_under
    else:
        print("All classes already at target size.")
        X_resampled, y_resampled = X_under, y_under
    
    # Check final distribution
    counter_after = Counter(y_resampled)
    print(f"\nFinal balanced distribution:")
    for i, class_name in enumerate(class_names):
        count = counter_after.get(i, 0)
        percentage = count / len(y_resampled) * 100 if len(y_resampled) > 0 else 0
        print(f"  {class_name}: {count} samples ({percentage:.2f}%)")
    
    return X_resampled, y_resampled

def process_stratified_sample_for_multiclass_classification():
    # Setup preprocessor for multiclass
    preprocessor, constant_cols = setup_preprocessor()
    
    # Load stratified dataset
    df = pd.read_csv('sampled_data/stratified_sampled_data.csv')
    
    # Remove constant columns
    df = df.drop(columns=constant_cols, errors='ignore')
    
    # Split data - Traffic Type is our target, Label is excluded
    X_stratified = df.drop(['Traffic Type', 'Label'], axis=1)
    y_stratified = df['Traffic Type']
    
    # Get unique class names and create label encoding
    class_names = sorted(y_stratified.unique())
    print(f"\nTraffic Type classes found: {class_names}")
    
    # Convert string labels to numeric
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_stratified_encoded = label_encoder.fit_transform(y_stratified)

    # Split the dataset into training and test sets
    X_train_stratified, X_test_stratified, y_train_stratified, y_test_stratified = train_test_split(
        X_stratified, y_stratified_encoded, test_size=0.2, random_state=42, stratify=y_stratified_encoded
    )

    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train_stratified)
    
    # APPLY MULTICLASS RESAMPLING TO THE TRAINING DATA ONLY
    print("\nApplying multiclass resampling to balance training data...")
    X_train_resampled, y_train_resampled = apply_multiclass_resampling(
        X_train_processed, y_train_stratified, class_names
    )
    
    X_test_processed = preprocessor.transform(X_test_stratified)

    # Convert to numpy arrays with appropriate dtypes
    X_train_processed = np.asarray(X_train_resampled).astype('float32')  # Use balanced data
    X_test_processed = np.asarray(X_test_processed).astype('float32') 
    y_train_processed = np.asarray(y_train_resampled).astype('int32')  # Use balanced labels
    y_test_processed = np.asarray(y_test_stratified).astype('int32')
    
    # Check for any remaining non-numeric values
    if np.any(np.isnan(X_train_processed)) or np.any(np.isinf(X_train_processed)):
        print("Warning: NaN or Inf values found in training data. Replacing with zeros.")
        X_train_processed = np.nan_to_num(X_train_processed)
    
    if np.any(np.isnan(X_test_processed)) or np.any(np.isinf(X_test_processed)):
        print("Warning: NaN or Inf values found in test data. Replacing with zeros.")
        X_test_processed = np.nan_to_num(X_test_processed)
    
    # Print data shapes and types before training neural network
    print(f"\nTraining multiclass neural network with data shapes:")
    print(f"X_train: {X_train_processed.shape}, dtype: {X_train_processed.dtype}")
    print(f"y_train: {y_train_processed.shape}, dtype: {y_train_processed.dtype}")
    print(f"Number of classes: {len(class_names)}")

    # Train neural network
    train_multiclass_neural_network(
        X_train_processed, 
        y_train_processed,
        X_test_processed,
        y_test_processed,
        'stratified',
        class_names
    )

    return preprocessor, constant_cols, label_encoder, class_names

def process_clustering_samples_for_multiclass_classification(preprocessor, constant_cols, label_encoder, class_names):
    other_datasets = {
        'kmeans': pd.read_csv('sampled_data/sampled_data_kmeans_sample_data.csv'),
        'hdbscan': pd.read_csv('sampled_data/hdbscan_sampled_data.csv')
    }
    
    for name, df in other_datasets.items():
        print(f"\n===== Processing {name.upper()} dataset for multiclass classification =====")
        
        # Remove constant columns
        df = df.drop(columns=constant_cols, errors='ignore')
        
        # Split data - Traffic Type is our target, Label is excluded
        X = df.drop(['Traffic Type', 'Label'], axis=1)
        y = df['Traffic Type']
        
        # Check if all classes are present
        present_classes = set(y.unique())
        expected_classes = set(class_names)
        missing_classes = expected_classes - present_classes
        
        if missing_classes:
            print(f"Warning: Missing classes in {name} dataset: {missing_classes}")
        
        # Convert string labels to numeric using the fitted label encoder
        try:
            y_encoded = label_encoder.transform(y)
        except ValueError as e:
            print(f"Error encoding labels in {name} dataset: {e}")
            print("Skipping this dataset...")
            continue
        
        # Display class distribution
        print(f"\nClass distribution in {name} dataset:")
        for i, class_name in enumerate(class_names):
            count = sum(y_encoded == i)
            percentage = count / len(y_encoded) * 100 if len(y_encoded) > 0 else 0
            print(f"  Class '{class_name}': {count} samples ({percentage:.2f}%)")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Use the ALREADY FITTED preprocessor
        X_train_processed = preprocessor.transform(X_train)
        
        # APPLY MULTICLASS RESAMPLING TO THE TRAINING DATA ONLY
        print(f"\nApplying multiclass resampling to {name} training data...")
        X_train_resampled, y_train_resampled = apply_multiclass_resampling(
            X_train_processed, y_train, class_names
        )
        
        X_test_processed = preprocessor.transform(X_test)
        
        # Convert to numpy arrays with appropriate dtypes
        X_train_processed = np.asarray(X_train_resampled).astype('float32')  # Use balanced data
        X_test_processed = np.asarray(X_test_processed).astype('float32')
        y_train_processed = np.asarray(y_train_resampled).astype('int32')  # Use balanced labels
        y_test_processed = np.asarray(y_test).astype('int32')
        
        # Check for any remaining non-numeric values
        if np.any(np.isnan(X_train_processed)) or np.any(np.isinf(X_train_processed)):
            print(f"Warning: NaN or Inf values found in {name} training data. Replacing with zeros.")
            X_train_processed = np.nan_to_num(X_train_processed)
        
        if np.any(np.isnan(X_test_processed)) or np.any(np.isinf(X_test_processed)):
            print(f"Warning: NaN or Inf values found in {name} test data. Replacing with zeros.")
            X_test_processed = np.nan_to_num(X_test_processed)

        # Print data shapes and types before training neural network
        print(f"\nTraining multiclass neural network with data shapes:")
        print(f"X_train: {X_train_processed.shape}, dtype: {X_train_processed.dtype}")
        print(f"y_train: {y_train_processed.shape}, dtype: {y_train_processed.dtype}")
        
        try:
            train_multiclass_neural_network(
                X_train_processed, 
                y_train_processed,
                X_test_processed,
                y_test_processed,
                name,
                class_names
            )

        except Exception as e:
            print(f"\nError training multiclass neural network on {name}: {str(e)}")
            continue

def nn_multiclass_classification():
    """Main function for multiclass traffic type classification"""
    
    # Process stratified sample first and get fitted preprocessor
    preprocessor, constant_cols, label_encoder, class_names = process_stratified_sample_for_multiclass_classification()
    
    # Process Clustering samples using the fitted preprocessor
    #process_clustering_samples_for_multiclass_classification(preprocessor, constant_cols, label_encoder, class_names)
        
    return

def main():

    nn_multiclass_classification()

if __name__ == "__main__":
    main()