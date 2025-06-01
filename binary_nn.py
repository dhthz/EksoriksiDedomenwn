import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from collections import Counter

from sklearn.model_selection import StratifiedKFold

# This function builds a binary classification neural network
def build_binary_neural_network(input_shape, act, loss):
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_shape,)),
        
        # First hidden layer
        layers.Dense(128, activation=act),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second hidden layer
        layers.Dense(64, act),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Third hidden layer
        layers.Dense(32, act),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        # Output layer for binary classification
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

# Function to train and evaluate neural network on a dataset
def train_binary_neural_network(X_train, y_train, X_test, y_test, dataset_name, epochs=50):
    print(f"\n===== STRATIFIED 5-FOLD CROSS-VALIDATION ON {dataset_name.upper()} =====")
    
    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    

    activation= ['swish', 'tanh', 'elu', 'prelu', layers.LeakyReLU(alpha=0.1)]
    loss= ['binary_crossentropy', 'mse']
    batch_size = [16, 32]
    epochs = [50, 100, 150, 200]

    # Perform k-fold cross validation
    for act in activation:
        for l in loss:
            for bs in batch_size:
                for epoch in epochs:
                    print(f"\n----- Training with Activation: {act}, Loss: {l}, Batch Size: {bs}, Epochs: {epoch} -----")
                    
                    # Store metrics for each fold
                    fold_metrics = []
                    fold_histories = []
                    fold_val_gaps = []
                    fold_generalization_metrics = []
                    
                    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                        print(f"\n----- Fold {fold+1}/5 -----")
                        
                        # Split data for this fold
                        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                        # Get input shape from training data
                        input_shape = X_train_fold.shape[1]
                        
                        # Build the neural network
                        model = build_binary_neural_network(input_shape, act, l)
                        
                        # Set up early stopping
                        early_stopping = keras.callbacks.EarlyStopping(
                            monitor='val_f1_score',  # Monitor F1 score instead of loss
                            patience=15,  # Increased patience
                            restore_best_weights=True,
                            mode='max'  # Maximize F1 score
                        )
                        
                        # Set up learning rate reduction
                        lr_reduction = keras.callbacks.ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.2,
                            patience=5,
                            min_lr=0.00001
                        )
                        
                        # ADD CLASS WEIGHTS FOR ADDITIONAL BALANCING
                        # Calculate class weights based on the training data
                        neg, pos = np.bincount(y_train.astype(int))
                        total = neg + pos
                        weight_for_0 = (1 / neg) * (total / 2.0)
                        weight_for_1 = (1 / pos) * (total / 2.0)
                        class_weight = {0: weight_for_0, 1: weight_for_1}
                        
                        # print(f"\nClass weights: {class_weight}")

                        # Train the model
                        history = model.fit(
                            X_train_fold, y_train_fold,
                            epochs=epoch,
                            batch_size=bs,
                            validation_data=(X_val_fold, y_val_fold),
                            callbacks=[early_stopping, lr_reduction],
                            # class_weight=class_weight, # ADD CLASS WEIGHTING
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
                        fold_gen = {
                            'train_acc': final_train_acc,
                            'val_acc': final_val_acc,
                            'val_gap': val_gap,
                            'loss': model_evaluation[0],
                            'accuracy': model_evaluation[1],
                            'precision': model_evaluation[2],
                            'recall': model_evaluation[3],
                            'auc': model_evaluation[4]
                        }
                        fold_generalization_metrics.append(fold_gen)
                    
                    # Calculate average metrics across all folds
                    fold_metrics = np.array(fold_metrics)
                    avg_metrics = np.mean(fold_metrics, axis=0)
                    std_metrics = np.std(fold_metrics, axis=0)
                    
                    # Calculate average generalization gap
                    avg_val_gap = np.mean(fold_val_gaps)
                    std_val_gap = np.std(fold_val_gaps)
                    
                    # Print evaluation metrics
                    print("\n===== CROSS-VALIDATION SUMMARY =====")
                    print(f"Average Loss: {avg_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
                    print(f"Average Accuracy: {avg_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
                    print(f"Average Precision: {avg_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
                    print(f"Average Recall: {avg_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
                    print(f"Average AUC: {avg_metrics[4]:.4f} ± {std_metrics[4]:.4f}")
                    print(f"Average Generalization Gap: {avg_val_gap:.4f} ± {std_val_gap:.4f}")
                    
                    # Create directory for saving results if it doesn't exist
                    evaluation_dir = "binary_nn_training"
                    if not os.path.exists(evaluation_dir):
                        os.makedirs(evaluation_dir)
                    
                    # Save evaluation metrics to CSV
                    csv_path = f"{evaluation_dir}/model-{act}-{l}-{bs}-{epoch}.csv"
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
    # plt.figure(figsize=(12, 5))
    
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
    # plt.savefig(f'classification_nn_evaluation_and_plots/training_history_{dataset_name}.png')
    
    return model, history

# Function for setting up feature columns and preprocessor
def setup_preprocessor():
    # 1. Features to remove (constant columns)
    constant_cols = ['Bwd URG Flags', 'Bwd PSH Flags', 'Fwd Bytes/Bulk Avg', 
                    'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg']

    # 2. Binary features : 'Fwd URG Flags', 'Fwd PSH Flags' (keep as is)

    # 3. Categorical features for one-hot encoding
    categorical_cols = ['Protocol', 'Down/Up Ratio', 
                    'RST Flag Count', 'FIN Flag Count']

    # 4. Flag features with low-medium cardinality for ordinal encoding
    flag_cols_ordinal = ['SYN Flag Count', 'CWR Flag Count', 'ECE Flag Count']

    # 5. Features that now need scaling due to higher cardinality
    medium_cardinality_cols = [
        'Subflow Fwd Packets', 'Subflow Bwd Packets', 
        'Bwd Init Win Bytes', 'FWD Init Win Bytes',
        'Subflow Bwd Bytes', 'Subflow Fwd Bytes',
        'Bwd Bulk Rate Avg', 'ACK Flag Count'
    ]

    # 6. Port columns for special encoding
    port_cols = ['Src Port', 'Dst Port']

    # 7. High-cardinality numerical features
    high_card_num_cols = [
        'Flow Packets/s', 'Bwd Packets/s', 'Flow Duration', 
        'Flow Bytes/s', 'Fwd IAT Std', 'Idle Std'
    ]

    # 8. Medium-cardinality numerical features
    med_card_num_cols = [
        'Packet Length Std', 'Fwd Packet Length Std', 'Average Packet Size', 
        'Active Std', 'Active Max'
    ]

    # 9. Lower-cardinality numerical features
    low_card_num_cols = [
        'Bwd IAT Max', 'Bwd Packet Length Min', 
        'Bwd Packet Length Max', 'Fwd Seg Size Min'
    ]

    # Create the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            # High-cardinality numerical features
            ('high_card_num', Pipeline([
                ('power', PowerTransformer(method='yeo-johnson')),
                ('scaler', RobustScaler())
            ]), high_card_num_cols),
            
            # Medium-cardinality numerical features
            ('med_card_num', RobustScaler(), med_card_num_cols),
            
            # Lower-cardinality numerical features
            ('low_card_num', RobustScaler(), low_card_num_cols),
            
            # Features with higher cardinality in stratified sample
            ('medium_cardinality', RobustScaler(), medium_cardinality_cols),
            
            # Flag features with low cardinality
            ('flag_ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), 
            flag_cols_ordinal),
            
            # Port columns
            ('port', FunctionTransformer(encode_port_categories), port_cols),
            
            # Categorical features
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), 
            categorical_cols)
        ],
        remainder='passthrough'
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
        result[f'{col}_registered'] = ((result[col] > 1023) & (result[col] <= 49151)).astype(int)
        
        # Dynamic/private ports (49152-65535)
        result[f'{col}_dynamic'] = (result[col] > 49151).astype(int)
        
        # Common service ports (add more as needed)
        result[f'{col}_web'] = result[col].isin([80, 443, 8080, 8443]).astype(int)
        result[f'{col}_email'] = result[col].isin([25, 465, 587, 110, 143, 993, 995]).astype(int)
        result[f'{col}_file_transfer'] = result[col].isin([20, 21, 22, 69]).astype(int)
        result[f'{col}_dns'] = result[col].isin([53]).astype(int)
        
    # Drop original port columns
    return result.drop(['Src Port', 'Dst Port'], axis=1)

# Apply hybrid resampling (oversampling + undersampling) to balance classes
def apply_binary_resampling(X, y):
    """
    Apply a hybrid approach of undersampling and oversampling to balance classes.
    
    Parameters:
    - X: Features DataFrame/array
    - y: Target labels (0 for Benign, 1 for Malicious)
    - approach: 'hybrid' (default), 'smote_tomek', or 'smote_enn'
    - display_plots: Whether to display distribution plots
    
    Returns:
    - X_resampled, y_resampled: Balanced dataset
    """
    # Check initial class distribution
    counter_before = Counter(y)
    print(f"\nOriginal class distribution:")
    print(f"  Benign: {counter_before[0]} samples ({counter_before[0]/len(y)*100:.2f}%)")
    print(f"  Malicious: {counter_before[1]} samples ({counter_before[1]/len(y)*100:.2f}%)")
    
    # Step 1: Undersample the majority class
    print("\nStep 1: Applying undersampling to reduce majority class...")
    undersampler = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
    X_under, y_under = undersampler.fit_resample(X, y)
    
    # Check intermediate distribution
    counter_under = Counter(y_under)
    print(f"  After undersampling - Benign: {counter_under[0]} samples ({counter_under[0]/len(y_under)*100:.2f}%)")
    print(f"  After undersampling - Malicious: {counter_under[1]} samples ({counter_under[1]/len(y_under)*100:.2f}%)")
       
    # First apply SMOTE
    smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_under, y_under)
    
    # Then apply BorderlineSMOTE to refine boundary samples
    b_smote = BorderlineSMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=5)
    X_resampled, y_resampled = b_smote.fit_resample(X_resampled, y_resampled)
    
    # ANOTHER APPROACH
    # First integrated method
    # smote_tomek = SMOTETomek(sampling_strategy=0.25, random_state=42)
    # X_tomek, y_tomek = smote_tomek.fit_resample(X, y)

    # Second integrated method for further cleaning
    # smote_enn = SMOTEENN(sampling_strategy=0.5, random_state=42)
    # X_resampled, y_resampled = smote_enn.fit_resample(X_tomek, y_tomek)

    # Check final distribution
    counter_after = Counter(y_resampled)
    print(f"\nFinal class distribution after resampling:")
    print(f"\n   Benign: {counter_after[0]} samples ({counter_after[0]/len(y_resampled)*100:.2f}%)")
    print(f"   Malicious: {counter_after[1]} samples ({counter_after[1]/len(y_resampled)*100:.2f}%)")
    
    
    return X_resampled, y_resampled

def process_stratified_sample_for_binay_classification():
    # Setup preprocessor
    preprocessor, constant_cols = setup_preprocessor()
    
    # Load stratified dataset
    df = pd.read_csv('sampled_data/stratified_sampled_data.csv')
    
    # Remove constant columns
    df = df.drop(columns=constant_cols, errors='ignore')
    
    # Split data
    X_stratified = df.drop(['Label', 'Traffic Type'], axis=1)
    y_stratified = df['Label']
    
    # Convert string labels to numeric (crucial step)
    # If your labels are 'Malicious' and 'Benign', map them to 1 and 0
    if y_stratified.dtype == 'object':  # Check if labels are strings/objects
        label_map = {'Malicious': 1, 'Benign': 0}  # Adjust these values based on your actual labels
        y_stratified = y_stratified.map(label_map)

    # split (one time) the dataset into training and test sets
    X_train_stratified, X_test_stratified, y_train_stratified, y_test_stratified = train_test_split(
        X_stratified, y_stratified, test_size=0.2, random_state=42
    )

    X_train_processed = preprocessor.fit_transform(X_train_stratified)

    # APPLY HYBRID RESAMPLING TO THE TRAINING DATA ONLY
    # COMMENTED OUT WHEN WHEN USING THE PREPROCESSED UNBALANCED DATASET
    print("\nApplying hybrid resampling to balance training data...")
    X_train_resampled, y_train_resampled = apply_binary_resampling(X_train_processed, y_train_stratified)
    
    X_test_processed = preprocessor.transform(X_test_stratified)

    # Convert to numpy arrays with float32 dtype (crucial for TensorFlow)
    # X_train_processed = np.asarray(X_train_processed).astype('float32') # OPTION UNBALANCED
    X_train_processed = np.asarray(X_train_resampled).astype('float32') # OPTION BALANCED
    X_test_processed = np.asarray(X_test_processed).astype('float32') 
    # y_train_processed = np.asarray(y_train_stratified).astype('float32') # OPTION UNBALANCED
    y_train_processed = np.asarray(y_train_resampled).astype('float32') # OPTION BALANCED
    y_test_processed = np.asarray(y_test_stratified).astype('float32')
    
    # Check for any remaining non-numeric values
    if np.any(np.isnan(X_train_processed)) or np.any(np.isinf(X_train_processed)):
        print("Warning: NaN or Inf values found in training data. Replacing with zeros.")
        X_train_processed = np.nan_to_num(X_train_processed)
    
    if np.any(np.isnan(X_test_processed)) or np.any(np.isinf(X_test_processed)):
        print("Warning: NaN or Inf values found in test data. Replacing with zeros.")
        X_test_processed = np.nan_to_num(X_test_processed)
    
    # Store processed data
    processed_data = {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train_processed,
        'y_test': y_test_processed
    }
    
    # Print data shapes and types before training neural network
    print(f"\nTraining neural network with data shapes:")
    print(f"X_train: {processed_data['X_train'].shape}, dtype: {processed_data['X_train'].dtype}")
    print(f"y_train: {processed_data['y_train'].shape}, dtype: {processed_data['y_train'].dtype}")

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
    
    nn_results = {}  # For Neural Network results
    
    for name, df in other_datasets.items():
        # Remove constant columns
        df = df.drop(columns=constant_cols, errors='ignore')
        
        # Split data
        X = df.drop('Label', axis=1)
        y = df['Label']

            
        # Convert string labels to numeric (crucial step)
        # If your labels are 'Malicious' and 'Benign', map them to 1 and 0
        if y.dtype == 'object':  # Check if labels are strings/objects
            print("\nConverting string labels to numeric values...")
            label_map = {'Malicious': 1, 'Benign': 0}  # Adjust these values based on your actual labels
            y = y.map(label_map)

        # Display class distribution
        print(f"\nClass distribution in {name} dataset:")
        print(f"  Class 'Benign': {sum(y == 0)} samples ({sum(y == 0)/len(y)*100:.2f}%)")
        print(f"  Class 'Malicious': {sum(y == 1)} samples ({sum(y == 1)/len(y)*100:.2f}%)")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Use the ALREADY FITTED preprocessor
        X_train_processed = preprocessor.transform(X_train)

        # APPLY HYBRID RESAMPLING TO THE TRAINING DATA ONLY
        X_train_resampled, y_train_resampled = apply_binary_resampling(
            X_train_processed, y_train,
        )

        # Use the ALREADY FITTED preprocessor
        # X_train_processed = preprocessor.transform(X_train_resampled)
        X_test_processed = preprocessor.transform(X_test)
        
        # Convert to numpy arrays with float32 dtype (crucial for TensorFlow)
        # X_train_processed = np.asarray(X_train_processed).astype('float32') # OPTION UNBALANCED
        X_train_processed = np.asarray(X_train_resampled).astype('float32') # OPTION BALANCED
        X_test_processed = np.asarray(X_test_processed).astype('float32')
        # y_train_processed = np.asarray(y_train).astype('float32') # OPTION UNBALANCED
        y_train_processed = np.asarray(y_train_resampled).astype('float32') # OPTION BALANCED
        y_test_processed = np.asarray(y_test).astype('float32')
        
        # Check for any remaining non-numeric values
        if np.any(np.isnan(X_train_processed)) or np.any(np.isinf(X_train_processed)):
            print(f"Warning: NaN or Inf values found in {name} training data. Replacing with zeros.")
            X_train_processed = np.nan_to_num(X_train_processed)
        
        if np.any(np.isnan(X_test_processed)) or np.any(np.isinf(X_test_processed)):
            print(f"Warning: NaN or Inf values found in {name} test data. Replacing with zeros.")
            X_test_processed = np.nan_to_num(X_test_processed)

        # Store processed data
        processed_data = {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train_processed,
            'y_test': y_test_processed
        }

        # Print data shapes and types before training neural network
        print(f"\nTraining neural network with data shapes:")
        print(f"X_train: {processed_data['X_train'].shape}, dtype: {processed_data['X_train'].dtype}")
        print(f"y_train: {processed_data['y_train'].shape}, dtype: {processed_data['y_train'].dtype}")
        
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

def nn_binary_classification():
    '''' proccess_stratified_sample return the fit preprocessor and process_clustering_samples uses it'''

    # Process stratified sample first and get fitted preprocessor
    preprocessor, constant_cols = process_stratified_sample_for_binay_classification()
    
    # Process Clustering samples using the fitted preprocessor
    process_clustering_samples_for_binary_classification(preprocessor, constant_cols)
        
    return 


def main():
    # Count unique values of columns
    #count_unique_values()
    nn_binary_classification()

if __name__ == "__main__":
    main()

# check which column i will not pre process and which i will encode 
# def count_unique_values():
#     file = ['sampled_data/hdbscan_sampled_data.csv', 'sampled_data/sampled_data_kmeans_sample_data.csv', 'sampled_data/stratified_sampled_data.csv']
#     for name in file:
#         print(f"Processing file: {name}")
        
#         df = pl.read_csv(name)
        
#         # Remove 'sampled_data/' from the file name for output
#         name = name.replace('sampled_data/', '')

#         # Create a dictionary to store column names and their unique value counts
#         unique_counts = {}

#         # Open a file to write the results
#         with open(f'columns_unique_values_count_in_{name}.txt', 'w') as f:
#             for col in df.columns:
#                 count = df[col].n_unique()
#                 unique_counts[col] = count
#                 f.write(f"{col}, : {count}\n")
        
#         print(f"Unique value counts for each column have been written to 'columns_unique_values_count_in_{name}.txt.")
#     return 

# def print_class_distribution(dataset_name, y):
#     """Print the distribution of classes in a dataset"""
#     if y.dtype == 'object':
#         # For string labels
#         class_counts = y.value_counts()
#     else:
#         # For numeric labels
#         class_counts = pd.Series(y).value_counts()
    
#     total = len(y)
#     print(f"\nClass distribution in {dataset_name} dataset:")
#     for class_label, count in class_counts.items():
#         percentage = (count / total) * 100
#         print(f"  Class '{class_label}': {count} samples ({percentage:.2f}%)")

# def check_class_distributions():
#     datasets = {
#         'stratified': 'sampled_data/stratified_sampled_data.csv',
#         'kmeans': 'sampled_data/sampled_data_kmeans_sample_data.csv',
#         'hdbscan': 'sampled_data/hdbscan_sampled_data.csv'
#     }
    
#     print("\n===== CLASS DISTRIBUTIONS =====")
#     for name, path in datasets.items():
#         df = pd.read_csv(path)
#         y = df['Label']
#         print_class_distribution(name, y)