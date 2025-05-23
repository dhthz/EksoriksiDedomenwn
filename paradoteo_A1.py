import os
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


# CORRECT Helper function gia na sanitizarei ta onomata twn columns
def sanitize_filename(original_name):
    """
    Sanitizes string for use in filenames and paths.
    Args: 
        - original_name: String to sanitize
    Returns: 
        - Sanitized string safe for filenames
    """
    # Replace problematic characters
    invalid_chars = '<>:"/\\|?*() '
    sanitized = original_name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')

    # Remove multiple underscores
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')

    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    return sanitized


def load_data_from_csv_parquet_format(file_name):  # CORRECT
    """
    Loads Data from Parquet or CSV file given its path and returns it as a polars Dataframe.
    Args: data (pl.DataFrame): Dataset (Polars DataFrame)
    """
    try:
        if (os.path.exists(f"{file_name}.parquet")):
            print(f"Loading data from file {file_name}.parquet")
            data_parquet = pl.read_parquet(f"{file_name}.parquet")
            print(f"Data loaded successfully from {file_name}.parquet")
        else:
            print(f"Loading data from file {file_name}.csv")
            data_csv = pl.read_csv(f"{file_name}.csv")
            print(f"Data loaded successfully from {file_name}.csv")

            print("Saving data to Parquet format...")
            data_csv.write_parquet(f"{file_name}.parquet")

            data_parquet = pl.read_parquet(f"{file_name}.parquet")
            print(f"Data loaded successfully from {file_name}.parquet")
        return data_parquet
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        return None


def analyze_data(data):  # CORRECT
    """
    Analyzes the data and returns statistics.
    Args: data (pl.DataFrame): Dataset (Polars DataFrame)
    """
    try:
        print("Performing Data Analysis...")

        # Dataset description
        description = data.describe()
        description.write_csv(
            f"data_analysis/data_description.csv")
        print("Data description saved to data_analysis/data_description.csv")

        # Data type analysis
        dtypes = data.dtypes
        dtypes_df = pl.DataFrame({"column": data.columns, "dtype": [
                                 str(dtype) for dtype in dtypes]})
        dtypes_df.write_csv("data_analysis/data_dtypes.csv")
        print("Data dtypes saved to data_analysis/data_dtypes.csv")

        # Dataset null values analysis
        missing_values_df = data.null_count()
        missing_values_df.write_csv("data_analysis/missing_values_full.csv")
        print("Missing values saved to data_analysis/missing_values_full.csv")

    except Exception as e:
        print(f"An error occurred during analysis: {e}")


def plot_histograms_grouped_by_column(data, column_name=None):  # Working
    """
    Plots memory-efficient grouped histograms for numeric columns.
    If column_name is provided, plots grouped histograms by that column.
    If column_name is None, plots regular histograms.
    """
    # Select numeric columns using Polars
    numeric_data = data.select(
        [col for col, dtype in data.schema.items() if dtype in (pl.Float64, pl.Int64)]
    )

    # Create output folder
    if column_name is not None:
        output_folder = f"grouped_histograms_by_{sanitize_filename(column_name)}"
    else:
        output_folder = "ungrouped_histograms"
    os.makedirs(output_folder, exist_ok=True)

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # If grouping is required
    if column_name is not None:
        # Get unique categories
        categories = data.select(column_name).unique().to_series().to_list()
        # Print unique categories
        print(f"Categories found in '{column_name}': {categories}")

        # Process each numeric column
        for col in numeric_data.columns:
            print(f"Plotting grouped histogram for: {col}")
            plt.figure(figsize=(12, 6))

            # Process each category separately
            for idx, category in enumerate(categories):
                # Filter and select data for each category using Polars
                category_data = (data
                                 .filter(pl.col(column_name) == category)
                                 .select(col)
                                 .to_numpy()
                                 .flatten())

                # Check count
                print(
                    f"Category '{category}' data count for {col}: {len(category_data)}")

                # If there is data for this category, plot the histogram
                if len(category_data) > 0:
                    plt.hist(category_data, bins=50, alpha=0.5,
                             label=str(category),
                             color=colors[idx % len(colors)],
                             density=True)  # Density plot

            plt.title(f"Distribution of {col} grouped by {column_name}")
            plt.xlabel(col)
            plt.ylabel("Density")
            plt.legend()

            # Save and cleanup
            sanitized_col = sanitize_filename(col)
            output_path = os.path.join(
                output_folder, f"{sanitized_col}_grouped_histogram.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Saved grouped histogram to {output_path}")

    else:
        # No grouping — just plot histograms for each numeric column
        for col in numeric_data.columns:
            print(f"Plotting histogram for: {col}")
            data_array = data.select(col).to_numpy().flatten()

            if len(data_array) > 0:
                plt.figure(figsize=(12, 6))
                # Compute 99th percentile for upper bound
                upper_bound = np.percentile(data_array, 99)

                # Clip only the upper bound
                clipped_data = data_array[data_array <= upper_bound]

                # Plot density instead of count
                plt.hist(clipped_data, bins=50, color='blue',
                         alpha=0.7, density=True)

                # Only set the upper bound for x-axis
                plt.xlim(None, upper_bound)
                plt.title(f"Density of {col}")
                plt.xlabel(col)
                plt.ylabel("Density")

                output_path = os.path.join(
                    output_folder, f"{sanitize_filename(col)}_histogram.png")
                plt.savefig(output_path)
                plt.close()
                print(f"Saved histogram to {output_path}")


def plot_boxplots_grouped_by_column(data, column_name=None):
    """
    Plots boxplots for all numeric columns grouped by the column_name.
    Args:
        - data (pl.DataFrame): Dataset
        - column_name (str): Column to group by
    """
    numeric_data = data.select(
        [col for col, dtype in data.schema.items() if dtype in (pl.Float64, pl.Int64)]
    )

    df = data.to_pandas()

    if column_name is not None:
        output_folder = f"boxplots_by_{sanitize_filename(column_name)}"
    else:
        output_folder = "ungrouped_boxplots"

    os.makedirs(output_folder, exist_ok=True)

    for col in numeric_data.columns:
        print(f"Plotting boxplot for: {col}")

        # Skip empty columns
        if df[col].dropna().empty:
            print(f"Skipping column '{col}' because it has no data.")
            continue

        plt.figure(figsize=(15, 8))

        if column_name is not None:
            # Grouped boxplot
            df.boxplot(column=col, by=column_name, vert=False, grid=False,
                       patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.7))
            plt.title(f"Boxplot of {col} by {column_name}")
            plt.ylabel(column_name)
        else:
            # Regular (ungrouped) boxplot
            plt.boxplot(df[col].dropna(), vert=False,
                        patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.7))
            plt.title(f"Boxplot of {col}")
            plt.ylabel('')

        plt.suptitle("")
        plt.xlabel(col)

        # Save plot
        sanitized_col = sanitize_filename(col)
        if column_name is not None:
            sanitized_category = sanitize_filename(column_name)
            filename = f"{sanitized_col}_by_{sanitized_category}_boxplot.png"
        else:
            filename = f"{sanitized_col}_boxplot.png"

        output_path = os.path.join(output_folder, filename)
        plt.savefig(output_path)
        plt.close()
        print(f"Saved boxplot to {output_path}")


# TODO Tha to kanoume gia sugkekrimena columns mallon
def calculate_feature_correlations(data):
    """
    Calculates correlations between all numeric features to identify redundant columns.
    Returns a correlation matrix and saves analysis files.
    Args:
        data (pl.DataFrame): The full dataset
    """
    print("Analyzing feature correlations...")

    # First, convert categorical columns to numeric
    data_prep = data.clone()

    # Convert Label to numeric (0 = Benign, 1 = Malicious)
    if 'Label' in data_prep.columns:
        data_prep = data_prep.with_columns(
            pl.when(pl.col('Label') == 'Benign')
            .then(0)
            .otherwise(1)
            .alias('Label')
        )
        print("Converted 'Label' to numeric: 0=Benign, 1=Malicious")

    # Convert Traffic Type to numeric if present
    if 'Traffic Type' in data_prep.columns:
        # Get unique values and map them to integers
        traffic_types = data_prep.select(
            'Traffic Type').unique().sort('Traffic Type')
        type_list = traffic_types.to_series().to_list()
        type_mapping = {t: i for i, t in enumerate(type_list)}

        # Create mapping function
        def map_traffic_type(x):
            return type_mapping.get(x, None)

        # Apply mapping using map_elements
        data_prep = data_prep.with_columns(
            pl.col('Traffic Type')
            .map_elements(map_traffic_type, return_dtype=pl.Int64)
            .alias('Traffic Type')
        )

        # Log the mapping for reference
        print("Converted 'Traffic Type' to numeric with mapping:")
        for k, v in type_mapping.items():
            print(f"  {k} -> {v}")

    # Make sure Label is included in numeric columns
    # Create list of columns to include (Label + numeric columns)
    columns_to_analyze = []

    # First add Label if it exists (ensure it's first in the matrix)
    if 'Label' in data_prep.columns:
        columns_to_analyze.append('Label')

    # Then add other numeric columns (except Label which is already added)
    for col, dtype in data_prep.schema.items():
        if dtype in (pl.Float64, pl.Int64) and col != 'Label':
            columns_to_analyze.append(col)

    # Select columns for correlation analysis
    numeric_data = data_prep.select(columns_to_analyze)

    print(
        f"Working with {len(numeric_data.columns)} numeric columns (including Label)")

    # Check for and remove zero variance columns
    columns_to_keep = []
    zero_var_cols = []

    for col in numeric_data.columns:
        # Calculate variance using nan-safe methods
        try:
            # Explicitly compute variance to check if close to zero
            variance = numeric_data.select(pl.col(col)).var().to_numpy().item()
            if pd.isna(variance) or abs(variance) < 1e-10:  # Near-zero variance
                print(f"Column '{col}' has zero/near-zero variance, skipping.")
                zero_var_cols.append(col)
            else:
                columns_to_keep.append(col)
        except Exception as e:
            print(f"Error checking variance for column '{col}': {e}")
            zero_var_cols.append(col)

    if zero_var_cols:
        print(
            f"Removed {len(zero_var_cols)} zero-variance columns: {', '.join(zero_var_cols)}")
        numeric_data = numeric_data.select(columns_to_keep)

    print(
        f"Proceeding with {len(columns_to_keep)} columns for correlation analysis")

    # Calculate correlation matrix with remaining columns
    corr_matrix = numeric_data.corr()

    # Create pandas DataFrame for easier analysis
    corr_df = corr_matrix.to_pandas()

    # Ensure column names are properly set
    corr_df.index = numeric_data.columns
    corr_df.columns = numeric_data.columns

    # Create output folder
    if not os.path.exists("feature_selection"):
        output_folder = "feature_selection"
        os.makedirs(output_folder, exist_ok=True)

    # Save full correlation matrix as CSV
    corr_df.to_csv(os.path.join(output_folder, "full_correlation_matrix.csv"))

    # Find highly correlated feature pairs (candidates for removal)
    high_corr_pairs = []

    # Use numpy for efficiency with large matrices
    corr_values = corr_df.values
    np.fill_diagonal(corr_values, 0)  # Remove self-correlations

    # Find pairs with high absolute correlation
    high_corr_indices = np.where(np.abs(corr_values) > 0.7)

    # Save high correlation pairs to CSV
    with open(os.path.join(output_folder, "high_correlation_pairs.csv"), "w") as f:
        f.write("Feature1,Feature2,Correlation\n")
        for i, j in zip(high_corr_indices[0], high_corr_indices[1]):
            # Only include each pair once (i < j)
            if i < j:
                feature1 = numeric_data.columns[i]
                feature2 = numeric_data.columns[j]
                correlation = corr_values[i, j]
                f.write(f"{feature1},{feature2},{correlation:.6f}\n")
                high_corr_pairs.append((feature1, feature2, correlation))

    print(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
    print("Analysis files saved to 'feature_selection' folder")

    return corr_matrix


def count_categories_in_column(data, column_name):
    """
    Counts the occurrences of each category in the specified column.
    """
    # Get the unique categories in the specified column
    categories = data.select(column_name).unique().to_series().to_list()
    print(categories)
    # Create a dictionary to store the counts
    category_counts = {}

    # Loop through each category and count the occurrences
    for category in categories:
        category_count = data.filter(pl.col(column_name) == category).height
        category_counts[category] = category_count
        print(f"Category '{category}' has {category_count} records.")

    return category_counts


def extract_significant_correlations(corr_matrix, threshold=0.15):
    label_correlations = corr_matrix[0]  # First row is Label
    significant_features = [i for i, corr in enumerate(label_correlations)
                            if abs(corr) > threshold and i != 0]

    print(f"Features with correlation > {threshold} with Label:")
    for i in significant_features:
        print(f"Feature {i}: {label_correlations[i]:.4f}")

    return significant_features


def calculate_correlation_matrix(data):
    # WORKING
    columns_to_drop_NO_VARIANCE = [
        "Bwd PSH Flags",
        "Bwd URG Flags",
        "Fwd Bytes/Bulk Avg",
        "Fwd Packet/Bulk Avg",
        "Fwd Bulk Rate Avg"
    ]

    # Drop from the main DataFrame
    try:
        data = data.drop(columns_to_drop_NO_VARIANCE)
    except Exception as e:
        print(f"Warning when dropping predefined columns: {e}")
        # Continue with available columns

    # Select numeric columns
    numeric_data = data.select(
        [col for col, dtype in data.schema.items() if dtype in (pl.Float64, pl.Int64)]
    )

    print(f"Working with {len(numeric_data.columns)} numeric columns")

    # Check for and remove zero variance columns
    columns_to_drop = []
    for col in numeric_data.columns:
        # Calculate variance for the column
        variance = numeric_data.select(pl.col(col)).var(
        ).to_numpy().item()  # Convert to scalar value
        # Check if variance is zero
        if variance == 0:
            print(f"Column {col} has zero variance.")
            columns_to_drop.append(col)

    # Drop columns with zero variance
    if columns_to_drop:
        numeric_data = numeric_data.drop(columns_to_drop)
        print(
            f"Dropped {len(columns_to_drop)} additional columns with zero variance.")

    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()

    # Convert to pandas for analysis and saving
    corr_matrix_pd = corr_matrix.to_pandas()

    # Print the shape to verify
    print(f"Correlation matrix shape: {corr_matrix_pd.shape}")

    # Debug: Print the first few column names
    print("First 5 column names:", corr_matrix_pd.columns[:5].tolist())

    # Make lists of column and index names - IMPORTANT: we use the actual names from the pandas DataFrame
    all_columns = corr_matrix_pd.columns.tolist()

    # DEBUG: Check if index and columns are the same
    if not all(corr_matrix_pd.index == corr_matrix_pd.columns):
        print("WARNING: Index and columns in correlation matrix do not match!")

    # 1. Find top positive correlations using numpy operations (more efficient)
    top_positive_corrs = []
    corr_np = corr_matrix_pd.to_numpy()

    # Create masks for the correlation matrix
    np.fill_diagonal(corr_np, 0)  # Mask self-correlations
    high_corr_mask = corr_np > 0.8

    if high_corr_mask.any():
        # Get indices of high correlations
        high_corr_indices = np.where(high_corr_mask)

        # Create list of (col1, col2, corr_value)
        for i, j in zip(high_corr_indices[0], high_corr_indices[1]):
            col1 = all_columns[i]
            col2 = all_columns[j]
            corr_value = corr_np[i, j]
            top_positive_corrs.append((col1, col2, corr_value))

        # Sort by correlation strength
        top_positive_corrs.sort(key=lambda x: x[2], reverse=True)

    # 2. Find top negative correlations (similar approach)
    top_negative_corrs = []
    low_corr_mask = corr_np < -0.8

    if low_corr_mask.any():
        # Get indices of low correlations
        low_corr_indices = np.where(low_corr_mask)

        # Create list of (col1, col2, corr_value)
        for i, j in zip(low_corr_indices[0], low_corr_indices[1]):
            col1 = all_columns[i]
            col2 = all_columns[j]
            corr_value = corr_np[i, j]
            top_negative_corrs.append((col1, col2, corr_value))

        # Sort by correlation strength
        top_negative_corrs.sort(key=lambda x: x[2])

    # 3. Identify feature clusters using numpy (more efficient)
    threshold = 0.9
    abs_corr_np = np.abs(corr_np)
    # Set diagonal to 0 to avoid self-correlations
    np.fill_diagonal(abs_corr_np, 0)

    feature_clusters = []
    processed_indices = set()

    for i in range(len(all_columns)):
        if i in processed_indices:
            continue

        # Find all columns highly correlated with column i
        correlated_indices = np.where(abs_corr_np[i] > threshold)[0]

        # If there are any correlations above threshold
        if len(correlated_indices) > 0:
            # Add current index to the cluster
            cluster_indices = [i] + correlated_indices.tolist()
            # Convert indices to feature names
            cluster = [all_columns[idx] for idx in cluster_indices]
            # Add to clusters
            feature_clusters.append(cluster)
            # Mark as processed
            processed_indices.update(cluster_indices)

    # 4. Save valuable information to separate files
    # Save top positive correlations
    if top_positive_corrs:
        with open('top_positive_correlations.txt', 'w') as f:
            f.write("Feature 1\tFeature 2\tCorrelation\n")
            for col1, col2, corr in top_positive_corrs[:20]:  # Save top 20
                f.write(f"{col1}\t{col2}\t{corr:.4f}\n")
        print("Top positive correlations saved to 'top_positive_correlations.txt'")

    # Save top negative correlations
    if top_negative_corrs:
        with open('top_negative_correlations.txt', 'w') as f:
            f.write("Feature 1\tFeature 2\tCorrelation\n")
            for col1, col2, corr in top_negative_corrs[:20]:  # Save top 20
                f.write(f"{col1}\t{col2}\t{corr:.4f}\n")
        print("Top negative correlations saved to 'top_negative_correlations.txt'")

    # Save feature clusters
    if feature_clusters:
        with open('feature_clusters.txt', 'w') as f:
            f.write("Features that are highly correlated (r > 0.9):\n")
            for i, cluster in enumerate(feature_clusters):
                f.write(f"Cluster {i+1}: {', '.join(cluster)}\n")
        print("Feature clusters saved to 'feature_clusters.txt'")

    # 5. Generate a simplified correlation matrix with only important features
    # Extract unique features from clusters (one per cluster)
    important_features = []
    clustered_features = set()

    # Add one feature from each cluster
    for cluster in feature_clusters:
        important_features.append(cluster[0])
        clustered_features.update(cluster)

    # Add features that aren't part of any cluster
    for feature in all_columns:
        if feature not in clustered_features:
            important_features.append(feature)

    # Create a simplified correlation matrix if we have a reasonable number of important features
    if len(important_features) < len(all_columns) and len(important_features) > 5:
        try:
            # Ensure all features are in the DataFrame
            valid_features = [
                f for f in important_features if f in all_columns and f in corr_matrix_pd.index]
            simplified_matrix = corr_matrix_pd.loc[valid_features,
                                                   valid_features]
            simplified_matrix.to_csv(
                'simplified_correlation_matrix.txt', sep='\t', header=True, index=True)
            print(
                f"Simplified correlation matrix with {len(valid_features)} features saved")
        except Exception as e:
            print(f"Error creating simplified matrix: {e}")
    else:
        print(
            "Skipping simplified matrix creation - too few features or not enough reduction")

    # Save the full correlation matrix to a .txt file using pandas
    corr_matrix_pd.to_csv('correlation_matrix.txt',
                          sep='\t', header=True, index=True)
    print("Full correlation matrix saved to 'correlation_matrix.txt'")

    return corr_matrix


def main():  # TODO REMOVE COMMENTS AND SOME PRINTS , CHECK FILE SAVING PATHS
    file_name = 'data'
    df = load_data_from_csv_parquet_format(file_name)

    if df is not None:

        # Analyze data
        analyze_data(df)

        # Ungrouped Histograms Plotting
        plot_histograms_grouped_by_column(df)

        # Grouped Histograms by column Label
        plot_histograms_grouped_by_column(df, 'Label')

        # Grouped Histograms by column Traffice Type
        plot_histograms_grouped_by_column(df, 'Traffic Type')

        # Ungrouped Boxplots
        plot_boxplots_grouped_by_column(df)

        # Grouped Boxplots by column Label
        plot_boxplots_grouped_by_column(df, 'Label')

        # Calculate correlations from whole dataset
        calculate_feature_correlations(df)


if __name__ == "__main__":
    main()
