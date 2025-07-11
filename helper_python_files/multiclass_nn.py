import os
import csv
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Multiclass neural network model
def build_multiclass_neural_network(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )
    
    return model

# Function to train and evaluate multiclass neural network
def train_multiclass_neural_network(X_train, y_train, X_test, y_test, dataset_name, class_names, epochs=100):
    print(f"\n===== MULTICLASS TRAFFIC TYPE CLASSIFICATION ON {dataset_name.upper()} =====")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store metrics for each fold
    fold_metrics = []
    fold_histories = []
    fold_val_gaps = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n----- Fold {fold+1}/5 -----")
        
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
       
        # Get input shape and number of classes
        input_shape = X_train_fold.shape[1]
        num_classes = len(class_names)
        
        model = build_multiclass_neural_network(input_shape, num_classes)
        
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
        
        # Get unique classes and their weights
        classes = np.unique(y_train_fold)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train_fold
        )
        
        # Create class weight dictionary
        class_weight_dict = {i: weight for i, weight in zip(classes, class_weights)}
        print(f"  Class weights: {class_weight_dict}")

        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stopping, lr_reduction],
            class_weight=class_weight_dict,  
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
    
    fold_metrics = np.array(fold_metrics)
    avg_metrics = np.mean(fold_metrics, axis=0)
    std_metrics = np.std(fold_metrics, axis=0)
    
    avg_val_gap = np.mean(fold_val_gaps)
    std_val_gap = np.std(fold_val_gaps)
    
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
    
    evaluation_dir = "multiclass_model"
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    
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
        
    # Figure 1: Training and validation metrics
    plt.figure(figsize=(12, 6))
    
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
    plt.savefig(f'{evaluation_dir}/multiclass_training_metrics_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{evaluation_dir}/multiclass_confusion_matrix_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Class distribution
    plt.figure(figsize=(10, 6))
    unique, counts = np.unique(y_test, return_counts=True)
    plt.bar([class_names[i] for i in unique], counts)
    plt.title(f'Test Set Class Distribution - {dataset_name}')
    plt.xlabel('Traffic Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{evaluation_dir}/multiclass_class_distribution_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, history

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

def process_stratified_sample_for_multiclass_classification():
    # Setup preprocessor for multiclass
    preprocessor, constant_cols = setup_preprocessor()
    
    df = pd.read_csv('sampled_data/stratified_sampled_data.csv')
    
    # Remove constant columns
    df = df.drop(columns=constant_cols, errors='ignore')
    
    # Split data - Traffic Type is our target, Label is excluded
    X_stratified = df.drop(['Traffic Type', 'Label'], axis=1)
    y_stratified = df['Traffic Type']
    
    # Get unique class names and create label encoding
    global class_names
    class_names = sorted(y_stratified.unique())
    print(f"\nTraffic Type classes found: {class_names}")
    
    # Convert string labels to numeric
    global label_encoder
    label_encoder = LabelEncoder()
    y_stratified_encoded = label_encoder.fit_transform(y_stratified)

    # Use all data for training (no test split)
    X_train_stratified = X_stratified
    y_train_stratified = y_stratified_encoded
    X_test_stratified = X_stratified  # Use same data for testing
    y_test_stratified = y_stratified_encoded  # Use same data for testing

    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train_stratified)
    X_test_processed = preprocessor.transform(X_test_stratified)

    # Convert to numpy arrays with appropriate dtypes
    X_train_processed = np.asarray(X_train_processed).astype('float32')
    X_test_processed = np.asarray(X_test_processed).astype('float32') 
    y_train_processed = np.asarray(y_train_stratified).astype('int32')
    y_test_processed = np.asarray(y_test_stratified).astype('int32')
    
    # Check for any remaining non-numeric values
    if np.any(np.isnan(X_train_processed)) or np.any(np.isinf(X_train_processed)):
        print("Warning: NaN or Inf values found in training data. Replacing with zeros.")
        X_train_processed = np.nan_to_num(X_train_processed)
    
    if np.any(np.isnan(X_test_processed)) or np.any(np.isinf(X_test_processed)):
        print("Warning: NaN or Inf values found in test data. Replacing with zeros.")
        X_test_processed = np.nan_to_num(X_test_processed)
    
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

    return preprocessor, constant_cols

def process_clustering_samples_for_multiclass_classification(preprocessor, constant_cols):
    other_datasets = {
        'kmeans': pd.read_csv('sampled_data/sampled_data_kmeans_sample_data.csv'),
        'hdbscan': pd.read_csv('sampled_data/hdbscan_sampled_data.csv')
    }
    
    for name, df in other_datasets.items():
        print(f"\n===== Processing {name.upper()} dataset for multiclass classification =====")
        
        # Remove constant columns
        df = df.drop(columns=constant_cols, errors='ignore')
        
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

        # Use all data (no train/test split)
        X_train = X
        y_train = y_encoded
        X_test = X  # Use same data for testing
        y_test = y_encoded  # Use same data for testing

        # Use the ALREADY FITTED preprocessor
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Convert to numpy arrays with appropriate dtypes
        X_train_processed = np.asarray(X_train_processed).astype('float32')
        X_test_processed = np.asarray(X_test_processed).astype('float32')
        y_train_processed = np.asarray(y_train).astype('int32')
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

def multiclass_nn():
    # Process stratified sample dataset first and return fitted preprocessor
    preprocessor, constant_cols = process_stratified_sample_for_multiclass_classification()
    
    # Process Clustering sample datasets using the fitted preprocessor
    process_clustering_samples_for_multiclass_classification(preprocessor, constant_cols)

    return