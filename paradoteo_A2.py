from svm import evaluate_binary_classification
from correlation_group_and_filtering import simplified_feature_selection, analyze_variance_with_pca
from paradoteo_A1 import load_data_from_csv_parquet_format
import os
import polars as pl
import math
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings  # To remove some warnings that dont cause an error
warnings.filterwarnings('ignore')


def stratified_sampling_multicolumn(data, columns, sample_size=100000, random_seed=42, mode=0):
    """
    Performs stratified sampling based on multiple columns.

    Args:
        data (pl.DataFrame): The input DataFrame
        columns (list): List of column names to stratify by
        sample_size (int): Target total sample size
        random_seed (int): Random seed for reproducibility

    Returns:
        pl.DataFrame: Stratified sample of the data
    """
    print(f"Performing stratified sampling by {columns}...")

    # Create a combined stratification column
    data = data.with_columns(
        pl.concat_str(
            [pl.col(col).cast(pl.Utf8) for col in columns],
            separator="_"
        ).alias("strat_group")
    )

    # Get class distribution for combined groups
    group_counts = data.group_by("strat_group").agg(pl.count().alias("count"))
    total_records = data.height

    # Calculate target count for each stratum
    group_counts = group_counts.with_columns(
        (pl.col("count") / total_records * sample_size).alias("target_count")
    )

    # Round to integers and ensure at least 1 sample per group
    group_counts = group_counts.with_columns(
        pl.max_horizontal(pl.col("target_count").round(), 1).cast(
            pl.Int64).alias("samples_needed")
    )

    # Make sure we're not asking for more samples than available
    group_counts = group_counts.with_columns(
        pl.min_horizontal(pl.col("samples_needed"),
                          pl.col("count")).alias("final_samples")
    )

    # Sample from each stratum
    sampled_dfs = []

    for row in group_counts.to_dicts():
        group_value = row["strat_group"]
        sample_count = row["final_samples"]

        # Filter to only this group
        group_data = data.filter(pl.col("strat_group") == group_value)

        # Random sample from this group
        try:
            group_sample = group_data.sample(n=sample_count, seed=random_seed)
            sampled_dfs.append(group_sample)
        except Exception as e:
            print(f"Error sampling from group {group_value}: {e}")

    # Combine all samples
    if sampled_dfs:
        stratified_sample = pl.concat(sampled_dfs)

        # Drop the temporary stratification column
        stratified_sample = stratified_sample.drop("strat_group")

        # This is for the stratified sampling and not the hybrid with clustering
        if mode == 0:
            stratified_sample = populate_dataset_with_rare_classes(
                data, stratified_sample)
            print(
                f"Created stratified sample with {stratified_sample.height} records")
            output_path = "sampled_data/stratified_sampled_data.csv"
            print(f"Saving results to {output_path}")
            stratified_sample = stratified_sample.sample(
                fraction=1.0, seed=42, shuffle=True)
            stratified_sample.write_csv(output_path)
        return stratified_sample
    else:
        print("Error: No samples were generated!")
        return None


def validate_dataset_quality(original_df, sampled_df):
    """
    Simplified validation of how well the sampled dataset preserves key characteristics.

    Args:
        original_df (pl.DataFrame): Original full dataset
        sampled_df (pl.DataFrame): Sampled smaller dataset

    Returns:
        dict: Dictionary with quality metrics
    """
    print("\nValidating dataset quality...")
    metrics = {}
    output_lines = ["Dataset Quality Validation Report", "=" * 50, ""]

    # Check class distribution preservation for Label and Traffic Type
    for col in ['Label', 'Traffic Type']:
        if col in original_df.columns and col in sampled_df.columns:
            header = f"\nDistribution comparison for '{col}':"
            print(header)
            output_lines.append(header.strip())

            # Get distributions with safe division
            try:
                orig_dist = original_df.group_by(
                    col).agg(pl.len().alias("count"))
                orig_total = original_df.height

                sample_dist = sampled_df.group_by(
                    col).agg(pl.len().alias("count"))
                sample_total = sampled_df.height

                # Calculate percentages with zero division protection
                if orig_total > 0:
                    orig_dist = orig_dist.with_columns(
                        (pl.col("count") / orig_total * 100).alias("percentage")
                    )
                else:
                    orig_dist = orig_dist.with_columns(
                        pl.lit(0.0).alias("percentage"))

                if sample_total > 0:
                    sample_dist = sample_dist.with_columns(
                        (pl.col("count") / sample_total * 100).alias("percentage")
                    )
                else:
                    sample_dist = sample_dist.with_columns(
                        pl.lit(0.0).alias("percentage"))

                # Join distributions
                comparison = orig_dist.join(
                    sample_dist,
                    on=col,
                    how="outer",
                    suffix="_sample"
                ).fill_null(0)

                # Calculate simple similarity score
                similarity_score = 0.0
                total_classes = comparison.height

                if total_classes > 0:
                    for row in comparison.to_dicts():
                        orig_pct = row.get("percentage", 0)
                        sample_pct = row.get("percentage_sample", 0)
                        # Simple difference metric (lower is better)
                        diff = abs(orig_pct - sample_pct)
                        similarity_score += max(0, 100 - diff)

                    similarity_score = similarity_score / total_classes
                else:
                    similarity_score = 100.0

                metrics[f'{col}_similarity'] = round(similarity_score, 2)

                similarity_line = f"  Distribution similarity: {similarity_score:.2f}%"
                print(similarity_line)
                output_lines.append(similarity_line)

                # Print simple comparison
                table_header = "  Class\t\tOriginal%\tSample%"
                table_separator = "  " + "-" * 40
                print(table_header)
                print(table_separator)
                output_lines.append(table_header)
                output_lines.append(table_separator)

                for row in comparison.to_dicts():
                    class_name = str(row[col])[:15]  # Truncate long names
                    orig_pct = row.get("percentage", 0)
                    sample_pct = row.get("percentage_sample", 0)
                    table_row = f"  {class_name:<15}\t{orig_pct:.1f}%\t\t{sample_pct:.1f}%"
                    print(table_row)
                    output_lines.append(table_row)

                output_lines.append("")  # Add blank line

            except Exception as e:
                error_line = f"  Error analyzing {col}: {e}"
                print(error_line)
                output_lines.append(error_line)
                metrics[f'{col}_similarity'] = 0.0

    # Overall quality assessment
    avg_similarity = 0.0
    similarity_count = 0

    for key, value in metrics.items():
        if '_similarity' in key:
            avg_similarity += value
            similarity_count += 1

    if similarity_count > 0:
        avg_similarity = avg_similarity / similarity_count
        metrics['overall_quality'] = round(avg_similarity, 2)

        overall_line = f"Overall Quality: {avg_similarity:.2f}%"
        print(f"\n{overall_line}")
        output_lines.append(overall_line)
    else:
        metrics['overall_quality'] = 0.0
        metrics['quality_rating'] = "UNKNOWN"
        overall_line = "Overall Quality: Could not assess"
        print(f"\n{overall_line}")
        output_lines.append(overall_line)

    # Basic size comparison
    reduction_ratio = 0.0
    if original_df.height > 0:
        reduction_ratio = (1 - sampled_df.height / original_df.height) * 100

    metrics['size_reduction'] = round(reduction_ratio, 2)
    size_line = f"Size reduction: {reduction_ratio:.1f}% (from {original_df.height} to {sampled_df.height} rows)"
    print(size_line)
    output_lines.append(size_line)

    # Add the captured output to metrics
    metrics['validation_report'] = "\n".join(output_lines)

    return metrics


def populate_dataset_with_rare_classes(original_df, sampled_df, random_seed=42):
    # Ensure at least 10 samples for each class
    if "Traffic Type" in sampled_df.columns:
        # Find classes with fewer than 10 samples
        class_counts = sampled_df.group_by(
            "Traffic Type").agg(pl.len().alias("count"))
        rare_classes = class_counts.filter(pl.col("count") < 30)[
            "Traffic Type"].to_list()

        if rare_classes:

            # Find the most common class
            most_common_row = class_counts.sort(
                "count", descending=True).row(0)
            # First element is Traffic Type
            most_common = most_common_row[0]
            # Second element is count
            most_common_count = most_common_row[1]

            print(
                f"Most common class: {most_common} with {most_common_count} samples")

            # Calculate how many samples we need to add
            samples_needed = {}
            total_needed = 0

            for cls in rare_classes:
                current_count = class_counts.filter(
                    pl.col("Traffic Type") == cls)["count"].item()
                needed = 30 - current_count
                samples_needed[cls] = needed
                total_needed += needed

            print(f"Need to add {total_needed} samples in total")

            # Get additional samples from original data
            extra_rows = []
            for cls in rare_classes:
                needed = samples_needed[cls]
                if needed <= 0:
                    continue

                # Get samples of this class from original data
                orig_samples = original_df.filter(
                    pl.col("Traffic Type") == cls)

                # Calculate how many samples we can take from original data
                available_originals = orig_samples.height

                if available_originals > needed:
                    # If we have enough original samples, use them
                    cols_to_select = sampled_df.columns
                    extra_sample = orig_samples.select(cols_to_select).sample(
                        n=needed, seed=random_seed+1)
                    extra_rows.append(extra_sample)
                else:
                    # If we don't have enough originals, use what we have
                    if available_originals > 0:
                        cols_to_select = sampled_df.columns
                        extra_sample = orig_samples.select(cols_to_select)
                        extra_rows.append(extra_sample)
                        needed -= available_originals

                    # For any remaining needed samples, duplicate existing ones with small variations
                    if needed > 0:
                        # Get the samples we already have in our stratified sample
                        existing_samples = sampled_df.filter(
                            pl.col("Traffic Type") == cls)

                        if existing_samples.height > 0:
                            for i in range(needed):
                                # Take a random sample from existing samples
                                seed_offset = i + 1  # Use different seed for each sample
                                duplicate = existing_samples.sample(
                                    n=1, seed=random_seed+seed_offset)
                                extra_rows.append(duplicate)
                        else:
                            print(
                                f"Warning: No samples available for class {cls}")

            # Combine all extra rows
            if extra_rows:
                extra_samples = pl.concat(extra_rows)
                total_added = extra_samples.height

                # Remove samples from most common class to maintain sample size
                if total_added > 0:
                    # Create a copy before adding indices to ensure we maintain original schema
                    filtered_sample = sampled_df.clone()

                    # Add row index for filtering
                    filtered_sample = filtered_sample.with_row_count("idx")

                    # Get indices of most common class
                    common_indices = filtered_sample.filter(
                        pl.col("Traffic Type") == most_common
                    ).select("idx").to_series().to_list()

                    # Keep all except the first n_to_remove rows of the most common class
                    n_to_remove = min(total_added, len(common_indices))

                    if len(common_indices) > n_to_remove:
                        to_remove = common_indices[:n_to_remove]
                        filtered_sample = filtered_sample.filter(
                            ~pl.col("idx").is_in(to_remove))

                        # Remove the temporary index column
                        filtered_sample = filtered_sample.drop("idx")

                        # Add the extra samples
                        final_sample = pl.concat(
                            [filtered_sample, extra_samples])

                        # Print final counts
                        final_counts = final_sample.group_by(
                            "Traffic Type").agg(pl.len().alias("count"))
                        min_count = final_counts["count"].min()
            return final_sample
        else:
            print("Error: No samples were generated!")
            return None


def calculate_new_data_set_preservation_statistics(df, dfAfterSampling, file_name):

    # Save a copy of the original dataset for validation
    df_before_sampling = df.clone()
    # Validate information preservation
    validation_metrics = validate_dataset_quality(
        df_before_sampling, dfAfterSampling)

    # Save validation metrics for reference
    with open(f"validation_metrics/validation_metrics_{file_name}.txt", "w") as f:
        f.write(validation_metrics['validation_report'])
    print("Validation metrics saved to validation_metrics.txt")


def kmeans_sampling(data, sample_size=10000, random_seed=42):
    print("Starting kmeans sampling...")

    # Create stratification column
    data = data.with_columns(pl.concat_str([pl.col('Label'), pl.col('Traffic Type')],
                                           separator="_").alias("strat_group"))

    # Get samples per group
    counts = data.group_by("strat_group").agg(pl.count())
    counts = counts.with_columns(
        (pl.col("count") / data.height * sample_size).round().cast(pl.Int64).alias("target"))
    counts = counts.with_columns(pl.max_horizontal(
        pl.col("target"), 1).alias("target"))

    # Both are used for catching really high counts of features and using stratified sampling to reduce them
    large_group_threshold = 70000
    pre_sample_size = 30000

    results = []
    for row in counts.to_dicts():
        group_data = data.filter(pl.col("strat_group") == row["strat_group"])
        target = min(row["target"], row["count"])

        # Check if this group is very large and needs pre-sampling
        if row["count"] > large_group_threshold:
            print(
                f"Large group detected: {row['strat_group']} with {row['count']} rows")
            print(
                f"Pre-sampling to {pre_sample_size} rows using stratified sampling")

            # Determine which columns to use for stratification
            strat_cols = []
            strat_cols.append("Label")
            strat_cols.append("Traffic Type")

            # Apply stratified sampling to reduce the size
            if strat_cols:
                print(f"Using stratification columns: {strat_cols}")
                # Use your existing stratified sampling function
                group_data = stratified_sampling_multicolumn(
                    group_data, strat_cols, sample_size=pre_sample_size, random_seed=random_seed, mode=1)

            print(f"Pre-sampled to {group_data.height} rows")

        numeric_cols = [col for col in group_data.columns
                        if col not in ['Label', 'Traffic Type', 'strat_group']
                        and group_data.schema[col] in (pl.Float64, pl.Int64)]

        # Extract and scale features
        X = group_data.select(numeric_cols).fill_null(0).to_numpy()
        X = StandardScaler().fit_transform(X)

        # Run K-means
        kmeans = KMeans(n_clusters=target, init='k-means++',
                        n_init=10, random_state=random_seed).fit(X)

        # Get points closest to centroids
        indices = []
        for i, centroid in enumerate(kmeans.cluster_centers_):
            cluster_points = np.where(kmeans.labels_ == i)[0]
            if len(cluster_points) == 0:
                continue
            distances = np.sum((X[cluster_points] - centroid)**2, axis=1)
            closest = cluster_points[np.argmin(distances)]
            indices.append(closest)

        results.append(group_data.sample(
            n=len(indices), seed=random_seed, with_replacement=False))

    if results:
        # Find common columns across all DataFrames
        all_columns = set(results[0].columns)
        for df in results[1:]:
            all_columns = all_columns.intersection(set(df.columns))

        # Select only common columns from each DataFrame
        aligned_results = []
        for df in results:
            # Drop strat_group if it exists, as we don't need it anymore
            if "strat_group" in df.columns and "strat_group" not in all_columns:
                df = df.drop("strat_group")

            # Select only common columns
            aligned_df = df.select([col for col in all_columns])
            aligned_results.append(aligned_df)

        # Now concat with aligned DataFrames
        result = pl.concat(aligned_results)

        result = populate_dataset_with_rare_classes(df, result)
        output_path = "sampled_data/sampled_data_kmeans_sample_data.csv"
        print(f"Saving results to {output_path}")
        result = result.sample(fraction=1.0, seed=42, shuffle=True)
        result.write_csv(output_path)

        print(f"Created {result.height} row kmeans sample")
        return result
    else:
        print("No results to concatenate")
        return None


def hdbscan_sampling(data, sample_size=10000, random_seed=42):
    print("Doing HDBSCAN sampling...")
    # Create stratification column
    data = data.with_columns(pl.concat_str([pl.col('Label'), pl.col('Traffic Type')],
                                           separator="_").alias("strat_group"))

    # Get samples per group
    counts = data.group_by("strat_group").agg(pl.count())
    counts = counts.with_columns(
        (pl.col("count") / data.height * sample_size).round().cast(pl.Int64).alias("target"))
    counts = counts.with_columns(pl.max_horizontal(
        pl.col("target"), 1).alias("target"))

    results = []
    # Define threshold for pre-sampling
    large_group_threshold = 70000  # 1 million
    pre_sample_size = 30000  # 100k

    for row in counts.to_dicts():
        group_data = data.filter(pl.col("strat_group") == row["strat_group"])
        target = min(row["target"], row["count"])
        # Check if this group is very large and needs pre-sampling
        if row["count"] > large_group_threshold:
            print(
                f"Large group detected: {row['strat_group']} with {row['count']} rows")
            print(
                f"Pre-sampling to {pre_sample_size} rows using stratified sampling")

            # Determine which columns to use for stratification
            strat_cols = []
            strat_cols.append("Label")
            strat_cols.append("Traffic Type")

            # Apply stratified sampling to reduce the size
            if strat_cols:
                group_data = stratified_sampling_multicolumn(
                    group_data, strat_cols, sample_size=pre_sample_size, random_seed=random_seed, mode=1)

            print(f"Pre-sampled to {group_data.height} rows")

        # Get numeric columns
        numeric_cols = [col for col in group_data.columns
                        if col not in ['Label', 'Traffic Type', 'strat_group']
                        and group_data.schema[col] in (pl.Float64, pl.Int64)]

        try:
            # Extract and scale features
            X = group_data.select(numeric_cols).fill_null(0).to_numpy()
            X = StandardScaler().fit_transform(X)

            # Run HDBSCAN
            # Adjust min_cluster_size based on group size
            min_cluster_size = max(5, min(50, len(X) // 100))
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size).fit(X)

            # Sample points from each cluster
            sample_indices = []

            # Get unique clusters (excluding noise points)
            clusters = np.unique(clusterer.labels_[clusterer.labels_ >= 0])
            print(f"Found {len(clusters)} clusters")

            # For each cluster, sample proportionally
            for cluster in clusters:
                cluster_points = np.where(clusterer.labels_ == cluster)[0]
                # Calculate how many to sample from this cluster
                cluster_target = max(
                    1, int(len(cluster_points) / len(X) * target))
                # Choose samples
                if len(cluster_points) <= cluster_target:
                    sample_indices.extend(cluster_points)
                else:
                    sample_indices.extend(np.random.choice(
                        cluster_points, cluster_target, replace=False))

            # Handle noise points (label -1) if not enough samples
            if len(sample_indices) < target:
                noise_points = np.where(clusterer.labels_ == -1)[0]
                if len(noise_points) > 0:
                    remaining = target - len(sample_indices)
                    noise_sample = np.random.choice(
                        noise_points, min(remaining, len(noise_points)), replace=False)
                    sample_indices.extend(noise_sample)

            # Take sample based on indices
            sample_indices = sample_indices[:target]
            results.append(group_data.sample(
                n=len(sample_indices), seed=random_seed))

        except Exception as e:
            print(f"HDBSCAN error: {e}. Using random sampling.")
            results.append(group_data.sample(n=target, seed=random_seed))

    # Combine and drop stratification column , concat didnt work had to do this
    if results:
        # Find common columns across all DataFrames
        all_columns = set(results[0].columns)
        for df in results[1:]:
            all_columns = all_columns.intersection(set(df.columns))

        # Select only common columns from each DataFrame
        aligned_results = []
        for df in results:
            # Drop strat_group if it exists, as we don't need it anymore
            if "strat_group" in df.columns and "strat_group" not in all_columns:
                df = df.drop("strat_group")

            # Select only common columns
            aligned_df = df.select([col for col in all_columns])
            aligned_results.append(aligned_df)

        # Now concat with aligned DataFrames
        result = pl.concat(aligned_results)

        result = populate_dataset_with_rare_classes(df, result)

        output_path = "sampled_data/hdbscan_sampled_data.csv"
        print(f"Saving results to {output_path}")
        result = result.sample(fraction=1.0, seed=42, shuffle=True)
        result.write_csv(output_path)

        print(f"Created {result.height} row HDBSCAN sample")
        return result
    else:
        print("No results to concatenate")
        return None


def main():
    file_name = 'data'
    df = load_data_from_csv_parquet_format(file_name)

    if not os.path.exists("sampled_data"):
        os.makedirs("sampled_data")

    if not os.path.exists("validation_metrics"):
        os.makedirs("validation_metrics")

    if df is not None:
        # Get the list of columns that will be kept from the original dataset
        selected_features = simplified_feature_selection(
            df,
            'feature_selection_details/high_correlation_pairs.csv',
            threshold=0.7
        )

        # Get the updated dataset with the lesser columns
        df_reduced_columns = df.select(selected_features)

        # Get statistics for the new dataset compared to the original by comparison using pca
        metrics = analyze_variance_with_pca(df, df_reduced_columns)

        # Perform stratified sampling on the reduced_columns dataset
        stratified_sample = stratified_sampling_multicolumn(
            df_reduced_columns, ['Label', 'Traffic Type'], 10000)

        # Calculate preservation statistics for the stratified sample based on the original dataset
        calculate_new_data_set_preservation_statistics(
            df_reduced_columns, stratified_sample, 'stratified_sample')

        # Perform Kmeans sampling
        kmeans_dataset = kmeans_sampling(df_reduced_columns)

        # Calculate preservation statistics for the kmeans sample based on the original dataset
        calculate_new_data_set_preservation_statistics(
            df, kmeans_dataset, 'kMeans')

        # Perform HDBSCAN sampling
        hdbscan_dataset = hdbscan_sampling(df_reduced_columns)

        # Calculate preservation statistics for the hdbscan sample based on the original dataset
        calculate_new_data_set_preservation_statistics(
            df, hdbscan_dataset, 'hdbscan')


if __name__ == "__main__":
    main()
