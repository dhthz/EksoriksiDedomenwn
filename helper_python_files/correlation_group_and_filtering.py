import pandas as pd
import numpy as np
from scipy import stats
import polars as pl
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings  # To remove some warnings that dont cause an error
warnings.filterwarnings('ignore')


def analyze_variance_with_pca(original_df, reduced_df, sample_size=1000000):
    print(f"Analyzing variance with PCA (using {sample_size} samples)...")

    # Create output directory
    output_dir = "pca_variance_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Sample data to reduce memory usage - crucial for large datasets
    print(f"Sampling {sample_size} rows from {original_df.height} total rows")
    random_indices = np.random.choice(min(original_df.height, reduced_df.height),
                                      min(sample_size, original_df.height),
                                      replace=False)

    # Sample both datasets using the same indices
    # Use the same random seed for both datasets to ensure consistency
    orig_sample = original_df.sample(
        n=min(sample_size, original_df.height), seed=42)
    reduced_sample = reduced_df.sample(
        n=min(sample_size, reduced_df.height), seed=42)

    # Convert to pandas for sklearn compatibility - more efficient on sampled data
    orig_pd = orig_sample.to_pandas()
    reduced_pd = reduced_sample.to_pandas()

    # Get numeric columns only
    orig_numeric = orig_pd.select_dtypes(include=['number']).drop(
        columns=['Label', 'Traffic Type'], errors='ignore')
    reduced_numeric = reduced_pd.select_dtypes(include=['number']).drop(
        columns=['Label', 'Traffic Type'], errors='ignore')

    print(f"Original numeric features: {orig_numeric.shape[1]}")
    print(f"Reduced numeric features: {reduced_numeric.shape[1]}")

    # Fill any NaN values to avoid PCA issues
    orig_numeric.fillna(0, inplace=True)
    reduced_numeric.fillna(0, inplace=True)

    # Standardize data
    scaler = StandardScaler()
    orig_scaled = scaler.fit_transform(orig_numeric)
    reduced_scaled = scaler.fit_transform(reduced_numeric)

    # Configure how many components to analyze - using less improves performance
    max_components = min(50, orig_numeric.shape[1], reduced_numeric.shape[1])
    print(f"Analyzing up to {max_components} principal components")

    # Perform PCA on original data
    try:
        pca_orig = PCA(n_components=max_components)
        pca_orig.fit(orig_scaled)

        # For reduced dataset
        pca_reduced = PCA(n_components=min(
            max_components, reduced_numeric.shape[1]))
        pca_reduced.fit(reduced_scaled)

        # Get cumulative variance explained
        orig_cum_var = np.cumsum(pca_orig.explained_variance_ratio_)
        reduced_cum_var = np.cumsum(pca_reduced.explained_variance_ratio_)

        # Calculate key metrics
        reduced_feature_count = reduced_numeric.shape[1]

        # Variance preserved by same number of components
        min_components = min(len(orig_cum_var), len(reduced_cum_var))
        common_variance_orig = orig_cum_var[min_components-1] * 100
        common_variance_reduced = reduced_cum_var[min_components-1] * 100

        # Estimate variance preservation
        variance_at_reduced_count = orig_cum_var[min(reduced_feature_count-1, len(
            orig_cum_var)-1)] * 100 if reduced_feature_count <= len(orig_cum_var) else 100.0

        # Components needed for certain variance thresholds
        orig_80pct = np.argmax(orig_cum_var >= 0.8) + \
            1 if any(orig_cum_var >= 0.8) else float('nan')
        reduced_80pct = np.argmax(
            reduced_cum_var >= 0.8) + 1 if any(reduced_cum_var >= 0.8) else float('nan')

        # Save metrics
        metrics = {
            'original_features': orig_numeric.shape[1],
            'reduced_features': reduced_numeric.shape[1],
            'feature_reduction_pct': round((1 - reduced_numeric.shape[1]/orig_numeric.shape[1]) * 100, 1),
            'variance_at_reduced_count': round(variance_at_reduced_count, 2),
            'orig_components_for_80pct': orig_80pct,
            'reduced_components_for_80pct': reduced_80pct,
        }

        # Create simplified plot
        plt.figure(figsize=(10, 6))

        plt.plot(range(1, len(orig_cum_var) + 1), orig_cum_var * 100, 'b-',
                 label=f'Original ({orig_numeric.shape[1]} features)')

        plt.plot(range(1, len(reduced_cum_var) + 1), reduced_cum_var * 100, 'r-',
                 label=f'Reduced ({reduced_numeric.shape[1]} features)')

        # Add reference lines
        plt.axhline(y=80, color='gray', linestyle='--',
                    alpha=0.7, label='80% Variance')

        if reduced_feature_count <= len(orig_cum_var):
            plt.axvline(x=reduced_feature_count, color='green', linestyle='--', alpha=0.7,
                        label=f'Reduced count ({reduced_feature_count})')

        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance (%)')
        plt.title('PCA Variance Comparison: Original vs. Reduced Dataset')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_variance_comparison.png")
        plt.close()

        # Save results to text file
        with open(f"{output_dir}/pca_variance_analysis.txt", "w") as f:
            f.write("PCA Variance Analysis Results\n")
            f.write("============================\n\n")

            f.write("Dataset Dimensions:\n")
            f.write(f"  Original features: {orig_numeric.shape[1]}\n")
            f.write(f"  Reduced features: {reduced_numeric.shape[1]}\n")
            f.write(
                f"  Feature reduction: {metrics['feature_reduction_pct']}%\n\n")

            f.write("Variance Preservation:\n")
            f.write(
                f"  Variance explained by {reduced_feature_count} components in original data: {variance_at_reduced_count:.2f}%\n\n")

            f.write("Components Required:\n")
            f.write(
                f"  Components needed for 80% variance (original): {orig_80pct}\n")
            f.write(
                f"  Components needed for 80% variance (reduced): {reduced_80pct}\n\n")

        print(f"PCA analysis completed. Results saved to {output_dir}/")
        print(
            f"Variance explained by reduced feature count: {variance_at_reduced_count:.2f}%")

        return metrics

    except Exception as e:
        print(f"Error in PCA analysis: {e}")
        return {
            'original_features': orig_numeric.shape[1],
            'reduced_features': reduced_numeric.shape[1],
            'feature_reduction_pct': round((1 - reduced_numeric.shape[1]/orig_numeric.shape[1]) * 100, 1),
            'error': str(e)
        }


def select_features_by_target_significance(groups):
    print("Selecting features based on target significance...")
    selected_features = []

    # Load correlation matrix if it exists
    if os.path.exists('feature_selection_details/full_correlation_matrix.csv'):
        try:
            # Load correlation matrix with first row as header
            corr_matrix = pd.read_csv(
                'feature_selection_details/full_correlation_matrix.csv')

            feature_names = corr_matrix.columns.tolist()

            # Find index of Label and Traffic Type in column headers if they exist
            label_index = -1
            traffic_type_index = -1

            for i, name in enumerate(feature_names):
                if name == 'Label':
                    label_index = i
                elif name == 'Traffic Type':
                    traffic_type_index = i

            if label_index >= 0 and traffic_type_index >= 0:
                # Initialize dictionaries to store correlations
                label_corrs = {}
                traffic_type_corrs = {}

                # Extract correlations from the Label column
                label_column = corr_matrix.iloc[:, label_index]
                for i, feature_name in enumerate(feature_names):
                    if feature_name != 'Label' and feature_name != 'Traffic Type':
                        label_corrs[feature_name] = abs(
                            float(label_column.iloc[i]))

                # Extract correlations from the Traffic Type column
                traffic_type_column = corr_matrix.iloc[:, traffic_type_index]
                for i, feature_name in enumerate(feature_names):
                    if feature_name != 'Label' and feature_name != 'Traffic Type':
                        traffic_type_corrs[feature_name] = abs(
                            float(traffic_type_column.iloc[i]))

                # For each group, select the feature with highest combined correlation
                for group_id, features in groups.items():
                    best_feature = None
                    highest_correlation = -1

                    for feature in features:
                        # Skip Label and Traffic Type
                        if feature in ['Label', 'Traffic Type']:
                            continue

                        # Calculate combined significance score
                        label_corr = label_corrs.get(feature, 0)
                        traffic_corr = traffic_type_corrs.get(feature, 0)
                        combined_score = label_corr + traffic_corr

                        if combined_score > highest_correlation:
                            highest_correlation = combined_score
                            best_feature = feature

                    if best_feature is not None:
                        selected_features.append(best_feature)
                        label_corr = label_corrs.get(best_feature, 0)
                        traffic_corr = traffic_type_corrs.get(best_feature, 0)

                    else:
                        # Fallback to alphabetical selection if no correlation data
                        sorted_features = sorted(features)
                        selected_features.append(sorted_features[0])
                        print(
                            f"Group {group_id}: No correlation data, selected {sorted_features[0]}")

        except Exception as e:
            print(f"Error processing correlation file: {e}")
            # Fallback to alphabetical selection
            for group_id, features in groups.items():
                sorted_features = sorted(features)
                selected_features.append(sorted_features[0])
    else:
        print("No correlation matrix found. Using alphabetical selection.")
        # Fallback to original selection method - alphabetical
        for group_id, features in groups.items():
            sorted_features = sorted(features)
            selected_features.append(sorted_features[0])

    return selected_features


def simplified_feature_selection(data, correlation_pairs_file, threshold=0.7):
    columns_to_drop_NO_VARIANCE = [
        "Bwd PSH Flags",
        "Bwd URG Flags",
        "Fwd Bytes/Bulk Avg",
        "Fwd Packet/Bulk Avg",
        "Fwd Bulk Rate Avg"
    ]

    # Drop from the main DataFrame

    data = data.drop(columns_to_drop_NO_VARIANCE)
    print("Running simplified feature selection...")

    # Create output directory
    output_dir = "feature_selection_details"
    os.makedirs(output_dir, exist_ok=True)

    # Load correlation pairs
    corr_pairs = pd.read_csv(correlation_pairs_file)
    high_corr = corr_pairs[corr_pairs['Correlation'] >= threshold]

    # Build a simple map of correlated features
    feature_to_group = {}
    group_id = 0

    # Process each high correlation pair
    for _, row in high_corr.iterrows():
        f1, f2 = row['Feature1'], row['Feature2']

        # If both features are already in groups
        if f1 in feature_to_group and f2 in feature_to_group:
            # If they're in different groups, merge the groups
            if feature_to_group[f1] != feature_to_group[f2]:
                old_group = feature_to_group[f2]
                for f in feature_to_group:
                    if feature_to_group[f] == old_group:
                        feature_to_group[f] = feature_to_group[f1]
        # If only first feature is in a group
        elif f1 in feature_to_group:
            feature_to_group[f2] = feature_to_group[f1]
        # If only second feature is in a group
        elif f2 in feature_to_group:
            feature_to_group[f1] = feature_to_group[f2]
        # If neither feature is in a group
        else:
            feature_to_group[f1] = group_id
            feature_to_group[f2] = group_id
            group_id += 1

    # Convert to groups
    groups = {}
    for feature, group_id in feature_to_group.items():
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(feature)

    # Get all features mentioned in correlation file
    all_features = set()
    for _, row in corr_pairs.iterrows():
        all_features.add(row['Feature1'])
        all_features.add(row['Feature2'])

    # Save correlation groups to file
    with open(f"{output_dir}/correlation_groups.txt", "w") as f:
        f.write(f"Correlation groups with threshold {threshold}:\n")
        f.write("=============================================\n\n")
        for group_id, features in groups.items():
            f.write(f"Group {group_id+1}: {', '.join(sorted(features))}\n\n")

    # Simple approach: select first feature from each group
    selected_features = select_features_by_target_significance(groups)

    # Add features not in any group
    numeric_cols = [col for col in data.columns
                    if data.schema[col] in (pl.Float64, pl.Int64)
                    and col not in ['Label', 'Traffic Type']]

    for col in numeric_cols:
        if col not in feature_to_group and col in all_features:
            selected_features.append(col)

    # Also add features not mentioned in correlations file
    for col in numeric_cols:
        if col not in all_features:
            selected_features.append(col)

    # Make sure to include Label and Traffic Type
    final_features = selected_features.copy()
    if 'Label' in data.columns and 'Label' not in final_features:
        final_features.append('Label')

    if 'Traffic Type' in data.columns and 'Traffic Type' not in final_features:
        final_features.append('Traffic Type')

    # Save final selected features
    with open(f"{output_dir}/final_selected_features.txt", "w") as f:
        f.write(f"Selected features ({len(final_features)}):\n")
        f.write("===============================\n\n")
        for feature in sorted(final_features):
            f.write(f"{feature}\n")

    # Save dropped features
    dropped_features = [f for f in all_features if f not in selected_features
                        and f not in ['Label', 'Traffic Type']]

    with open(f"{output_dir}/dropped_features.txt", "w") as f:
        f.write(f"Dropped features ({len(dropped_features)}):\n")
        f.write("===============================\n\n")
        for feature in sorted(dropped_features):
            group_id = feature_to_group.get(feature)
            if group_id is not None:
                kept_feature = [f for f in groups[group_id]
                                if f in selected_features][0]
                f.write(f"{feature} (kept {kept_feature} from same group)\n")

    print(
        f"Selected {len(final_features)} features, dropped {len(dropped_features)}")
    return final_features


def build_correlation_groups(corr_pairs, threshold):
    high_corr = corr_pairs[corr_pairs['Correlation'] >= threshold]

    # Build adjacency list
    adj_list = {}
    for _, row in high_corr.iterrows():
        f1, f2 = row['Feature1'], row['Feature2']
        if f1 not in adj_list:
            adj_list[f1] = set()
        if f2 not in adj_list:
            adj_list[f2] = set()
        adj_list[f1].add(f2)
        adj_list[f2].add(f1)

    # Find connected components (groups)
    visited = set()
    groups = []

    for node in adj_list:
        if node not in visited:
            group = set()
            dfs(node, adj_list, visited, group)
            groups.append(group)

    return groups


def dfs(node, adj_list, visited, group):
    visited.add(node)
    group.add(node)

    for neighbor in adj_list.get(node, []):
        if neighbor not in visited:
            dfs(neighbor, adj_list, visited, group)

            # In your main function
