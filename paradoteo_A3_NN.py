import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, RobustScaler, OneHotEncoder, OrdinalEncoder
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

# This function builds a binary classification neural network
def build_neural_network(input_shape):
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_shape,)),
        
        # First hidden layer
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second hidden layer
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Third hidden layer
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        # Output layer for binary classification
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
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
def train_neural_network(X_train, y_train, X_test, y_test, dataset_name, epochs=50):
    print(f"\n===== Training Neural Network on {dataset_name} dataset =====")
    
    # Get input shape from training data
    input_shape = X_train.shape[1]
    
    # Build the neural network
    model = build_neural_network(input_shape)
    
    # Set up early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Set up learning rate reduction
    lr_reduction = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, lr_reduction],
        verbose=1
    )
    
    # Evaluate the model
    test_results = model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'], 
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{dataset_name}.png')
    
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
    plt.savefig(f'training_history_{dataset_name}.png')
    
    # Print results summary
    print(f"\nNeural Network Results for {dataset_name}:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision: {report['1']['precision']:.4f}")
    print(f"Recall: {report['1']['recall']:.4f}")
    print(f"F1-Score: {report['1']['f1-score']:.4f}")
    
    return model, history, report

# Function for setting up feature columns and preprocessor
def setup_preprocessor():
    # 1. Features to remove (constant columns)
    constant_cols = ['Bwd URG Flags', 'Bwd PSH Flags', 'Fwd Bytes/Bulk Avg', 
                    'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg']

    # 2. Binary features (keep as is)
    binary_cols = ['Fwd URG Flags', 'Fwd PSH Flags']

    # 3. Categorical features for one-hot encoding
    categorical_cols = ['Protocol', 'Traffic Type', 'Down/Up Ratio', 
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

def process_stratified_sample():
    # Setup preprocessor
    preprocessor, constant_cols = setup_preprocessor()
    
    # Load stratified dataset
    df_stratified = pd.read_csv('sampled_data/stratified_sampled_data.csv')
    
    # Remove constant columns
    df_stratified = df_stratified.drop(columns=constant_cols, errors='ignore')
    
    # Split data
    X_stratified = df_stratified.drop('Label', axis=1)
    y_stratified = df_stratified['Label']
    
    # Convert string labels to numeric (crucial step)
    # If your labels are 'Malicious' and 'Benign', map them to 1 and 0
    if y_stratified.dtype == 'object':  # Check if labels are strings/objects
        print("Converting string labels to numeric values...")
        label_map = {'Malicious': 1, 'Benign': 0}  # Adjust these values based on your actual labels
        y_stratified = y_stratified.map(label_map)

    X_train_stratified, X_test_stratified, y_train_stratified, y_test_stratified = train_test_split(
        X_stratified, y_stratified, test_size=0.2, random_state=42
    )

    # Fit the preprocessor and transform stratified data
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
    
    # Store processed data
    processed_data = {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train_processed,
        'y_test': y_test_processed
    }
    
    # Train and evaluate Random Forest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(processed_data['X_train'], processed_data['y_train'])
    
    # Evaluate Random Forest
    y_pred = clf.predict(processed_data['X_test'])
    rf_results = classification_report(processed_data['y_test'], y_pred, output_dict=True)
    
    # Print Random Forest summary
    print(f"\nResults for Random Forest on stratified sample:")
    print(f"Accuracy: {rf_results['accuracy']:.4f}")
    print(f"Weighted F1: {rf_results['weighted avg']['f1-score']:.4f}")
    
    # Print data shapes and types before training neural network
    print(f"\nTraining neural network with data shapes:")
    print(f"X_train: {processed_data['X_train'].shape}, dtype: {processed_data['X_train'].dtype}")
    print(f"y_train: {processed_data['y_train'].shape}, dtype: {processed_data['y_train'].dtype}")
    
    # Train Neural Network model
    input_shape = processed_data['X_train'].shape[1]
    try:
        nn_model, nn_history = train_neural_network(
            processed_data['X_train'], 
            processed_data['y_train'],
            processed_data['X_test'],
            processed_data['y_test'],
            input_shape
        )
        
        # Evaluate Neural Network
        nn_eval = nn_model.evaluate(processed_data['X_test'], processed_data['y_test'])
        nn_y_pred = (nn_model.predict(processed_data['X_test']) > 0.5).astype(int)
        nn_results = classification_report(processed_data['y_test'], nn_y_pred, output_dict=True)
        
        # Print Neural Network summary
        print(f"\nResults for Neural Network on stratified sample:")
        print(f"Accuracy: {nn_results['accuracy']:.4f}")
        print(f"Weighted F1: {nn_results['weighted avg']['f1-score']:.4f}")
        
        return preprocessor, constant_cols, nn_results
    
    except Exception as e:
        print(f"\nError training neural network: {str(e)}")
        # Return what we have even if neural network training fails
        return preprocessor, constant_cols, {
            'rf_results': rf_results
        }

# Process other Clustering samples using the fitted preprocessor 
def process_clustering_samples(preprocessor, constant_cols):
    other_datasets = {
        'kmeans': pd.read_csv('sampled_data/sampled_data_kmeans_sample_data.csv'),
        'hdbscan': pd.read_csv('sampled_data/hdbscan_sampled_data.csv')
    }
    
    results = {}
    
    for name, df in other_datasets.items():
        # Remove constant columns
        df = df.drop(columns=constant_cols, errors='ignore')
        
        # Split data
        X = df.drop('Label', axis=1)
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use the ALREADY FITTED preprocessor
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Store processed data
        processed_data = {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_test': y_test
        }
        
        # Train and evaluate model
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(processed_data['X_train'], processed_data['y_train'])
        
        # Evaluate
        y_pred = clf.predict(processed_data['X_test'])
        
        # Store results
        results[name] = classification_report(processed_data['y_test'], y_pred, output_dict=True)
        
        # Print summary
        print(f"\nResults for {name} sample:")
        print(f"Accuracy: {results[name]['accuracy']:.4f}")
        print(f"Weighted F1: {results[name]['weighted avg']['f1-score']:.4f}")
    
    return results


def nn_binary_classification():
    # Process stratified sample first and get fitted preprocessor
    preprocessor, constant_cols, stratified_results = process_stratified_sample()
    
    # Process Clustering samples using the fitted preprocessor
    clustering_results = process_clustering_samples(preprocessor, constant_cols)
    
    # Print comparative summary
    print("\n===== COMPARATIVE SUMMARY =====")
    
    if 'nn_results' in stratified_results:
        print(f"STRATIFIED (Neural Network): Accuracy={stratified_results['nn_results']['accuracy']:.4f}, "
              f"F1={stratified_results['nn_results']['weighted avg']['f1-score']:.4f}")
    
    # Print clustering sample results
    for name, results in clustering_results.items():
        print(f"{name.upper()}: Accuracy={results['accuracy']:.4f}, "
              f"F1={results['weighted avg']['f1-score']:.4f}")
    
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