import os
import polars as pl
import math
import numpy as np
from paradoteo_A1 import load_data_from_csv_parquet_format
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def stratified_sampling_multicolumn(data, columns, sample_size, random_seed=42):
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

    print(f"Found {len(group_counts)} unique combinations")

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

        print(
            f"Created stratified sample with {stratified_sample.height} records")
        return stratified_sample
    else:
        print("Error: No samples were generated!")
        return None


def validate_dataset_quality(original_df, sampled_df):
    """
    Validates how well the sampled dataset preserves key characteristics of the original dataset.

    Args:
        original_df (pl.DataFrame): Original full dataset
        sampled_df (pl.DataFrame): Sampled smaller dataset

    Returns:
        dict: Dictionary with quality metrics
    """
    print("\nValidating dataset information preservation...")
    metrics = {}

    # 1. Check class distribution preservation
    for col in ['Label', 'Traffic Type']:
        if col in original_df.columns and col in sampled_df.columns:
            print(f"\nDistribution of '{col}':")

            orig_dist = original_df.group_by(col).agg(pl.count()).with_columns(
                (pl.col("count") / original_df.height * 100).alias("percentage")
            )

            sample_dist = sampled_df.group_by(col).agg(pl.count()).with_columns(
                (pl.col("count") / sampled_df.height * 100).alias("percentage")
            )

            # Join the distributions
            comparison = orig_dist.join(
                sample_dist,
                on=col,
                how="outer",
                suffix="_sample"
            ).sort("count", descending=True)

            # Calculate distribution similarity
            orig_percentages = comparison.select(
                "percentage").to_numpy().flatten()
            sample_percentages = comparison.select(
                pl.col("percentage_sample").fill_null(0)).to_numpy().flatten()

            # Jensen-Shannon divergence approximation (simpler than KL divergence) #TODO Check what this is
            diff = sum(abs(o - s)
                       for o, s in zip(orig_percentages, sample_percentages))
            similarity = 100 - (diff / 2)  # Convert to similarity percentage

            metrics[f"{col}_distribution_similarity"] = similarity
            print(f"  Distribution similarity: {similarity:.2f}%")

            # Print comparison
            for comp_row in comparison.to_dicts():
                orig_pct = comp_row["percentage"]
                sample_pct = comp_row.get(
                    "percentage_sample", 0) if "percentage_sample" in comp_row else 0
                print(
                    f"  {comp_row[col]}: {orig_pct:.2f}% in original, {sample_pct:.2f}% in sample")

    # 2. Check numeric column distribution preservation
    numeric_columns = [col for col in sampled_df.columns
                       if col not in ['Label', 'Traffic Type'] and
                       sampled_df.schema[col] in (pl.Float64, pl.Int64)]

    # Choose a few representative columns (for speed)
    sample_cols = numeric_columns[:5]  # First 5 numeric columns

    # Calculate basic statistics preservation
    stats_similarity = []
    for col in sample_cols:
        print(f"\nBasic statistics for column '{col}':")

        try:
            # Original stats
            orig_mean = original_df.select(pl.col(col).mean()).item()
            orig_std = original_df.select(pl.col(col).std()).item()
            orig_median = original_df.select(pl.col(col).median()).item()

            # Sample stats
            sample_mean = sampled_df.select(pl.col(col).mean()).item()
            sample_std = sampled_df.select(pl.col(col).std()).item()
            sample_median = sampled_df.select(pl.col(col).median()).item()

            # Calculate relative differences
            mean_diff_pct = 100 * \
                abs(orig_mean - sample_mean) / max(abs(orig_mean), 1e-10)
            std_diff_pct = 100 * \
                abs(orig_std - sample_std) / max(abs(orig_std), 1e-10)
            median_diff_pct = 100 * \
                abs(orig_median - sample_median) / max(abs(orig_median), 1e-10)

            # Average similarity
            avg_similarity = 100 - \
                ((mean_diff_pct + std_diff_pct + median_diff_pct) / 3)
            stats_similarity.append(avg_similarity)

            print(
                f"  Mean: {orig_mean:.3f} (original) vs {sample_mean:.3f} (sample), diff: {mean_diff_pct:.2f}%")
            print(
                f"  StdDev: {orig_std:.3f} (original) vs {sample_std:.3f} (sample), diff: {std_diff_pct:.2f}%")
            print(
                f"  Median: {orig_median:.3f} (original) vs {sample_median:.3f} (sample), diff: {median_diff_pct:.2f}%")
            print(f"  Column similarity: {avg_similarity:.2f}%")
        except Exception as e:
            print(f"  Error calculating stats: {e}")

    # Average numeric column similarity
    if stats_similarity:
        metrics['numeric_stats_similarity'] = sum(
            stats_similarity) / len(stats_similarity)
        print(
            f"\nOverall numeric distribution similarity: {metrics['numeric_stats_similarity']:.2f}%")

    # 3. Overall information preservation score
    all_scores = list(metrics.values())
    if all_scores:
        overall_score = sum(all_scores) / len(all_scores)
        metrics['overall_information_preservation'] = overall_score

        print(
            f"\nOVERALL INFORMATION PRESERVATION SCORE: {overall_score:.2f}%")

        if overall_score >= 95:  # TODO REMOVE Later
            print(
                "EXCELLENT: Sample preserves nearly all information from original dataset")
        elif overall_score >= 90:
            print("VERY GOOD: Sample preserves most important information")
        elif overall_score >= 80:
            print("GOOD: Sample preserves key information but with some differences")
        elif overall_score >= 70:
            print("FAIR: Sample shows notable differences from original data")
        else:
            print("POOR: Sample may not adequately represent original data")

    return metrics


def calculate_new_data_set_preservation_statistics(df, dfAfterSampling, fileName):
    print(f"Loaded dataset with {df.height:,} rows and {df.width} columns")

    # Save a copy of the original dataset for validation
    df_before_sampling = df.clone()
    # Drop columns with zero/low variance (optional)
    if df.width > 30:  # Only if we have many columns
        print("Removing low-variance columns...")
        # Calculate variances
        variances = {}
        for col in df.columns:
            if col not in ['Label', 'Traffic Type']:
                try:
                    var = df.select(pl.col(col).var()).item()
                    variances[col] = var
                except:
                    variances[col] = 0
        # Keep columns with non-zero variance and essential columns
        essential_cols = ['Label', 'Traffic Type']
        keep_cols = [col for col in df.columns if col in essential_cols or
                     (col in variances and variances[col] > 0)]
        df = df.select(keep_cols)
        print(f"Reduced to {df.width} columns")

        # Validate information preservation
        validation_metrics = validate_dataset_quality(
            df_before_sampling, dfAfterSampling)
        # Save results
        dfAfterSampling.write_parquet(f"sampled_data_10k_${fileName}.parquet")
        print(
            f"Saved {dfAfterSampling.height:,} samples to sampled_data_10k.parquet")
        # Save validation metrics for reference
        with open(f"validation_metrics_${fileName}.txt", "w") as f:
            f.write("Dataset Information Preservation Metrics:\n")
            f.write("=======================================\n\n")
            for key, value in validation_metrics.items():
                f.write(f"{key}: {value:.2f}%\n")
            f.write(
                f"\nReduction ratio: {df.height / dfAfterSampling.height:.0f}:1")
        print("Validation metrics saved to validation_metrics.txt")


def kmeans_sampling(data, sample_size=10000, random_seed=42):
    """Simple K-means sampling using scikit-learn."""
    print("RUNNING KMENANS  SAMPLING")

    # Create stratification column
    if 'Label' in data.columns:
        if 'Traffic Type' in data.columns:
            data = data.with_columns(pl.concat_str([pl.col('Label'), pl.col('Traffic Type')],
                                                   separator="_").alias("strat_group"))
        else:
            data = data.with_columns(pl.col('Label').alias("strat_group"))
    else:
        data = data.with_columns(pl.lit("all").alias("strat_group"))
    print("GETTING COUNTS")

    # Get samples per group
    counts = data.group_by("strat_group").agg(pl.count())
    counts = counts.with_columns(
        (pl.col("count") / data.height * sample_size).round().cast(pl.Int64).alias("target"))
    counts = counts.with_columns(pl.max_horizontal(
        pl.col("target"), 1).alias("target"))

    results = []
    print("FIRST FOR")
    for row in counts.to_dicts():
        group_data = data.filter(pl.col("strat_group") == row["strat_group"])
        target = min(row["target"], row["count"])

        # Get numeric columns
        numeric_cols = [col for col in group_data.columns
                        if col not in ['Label', 'Traffic Type', 'strat_group']
                        and group_data.schema[col] in (pl.Float64, pl.Int64)]

        # Extract and scale features
        X = group_data.select(numeric_cols).fill_null(0).to_numpy()
        X = StandardScaler().fit_transform(X)
        print("RUNNING KMENAS ")
        # Run K-means
        kmeans = KMeans(n_clusters=target, init='k-means++',
                        n_init=10, random_state=random_seed).fit(X)

        # Get points closest to centroids
        indices = []
        print("RUNNING KMENAS for")
        for i, centroid in enumerate(kmeans.cluster_centers_):
            cluster_points = np.where(kmeans.labels_ == i)[0]
            if len(cluster_points) == 0:
                continue
            distances = np.sum((X[cluster_points] - centroid)**2, axis=1)
            closest = cluster_points[np.argmin(distances)]
            indices.append(closest)

        results.append(group_data.sample(
            n=len(indices), seed=random_seed, with_replacement=False))
    print("RESULT")
    # Combine and drop stratification column
    result = pl.concat(results).drop("strat_group")
    print(f"Created {result.height} row K-means sample")
    return result


def hdbscan_sampling(data, sample_size=10000, random_seed=42):
    """HDBSCAN sampling with pre-sampling for very large strata."""
    print("DOING HDBSCAN SAMPLING")

    # Create stratification column
    if 'Label' in data.columns:
        if 'Traffic Type' in data.columns:
            data = data.with_columns(pl.concat_str([pl.col('Label'), pl.col('Traffic Type')],
                                                   separator="_").alias("strat_group"))
        else:
            data = data.with_columns(pl.col('Label').alias("strat_group"))
    else:
        data = data.with_columns(pl.lit("all").alias("strat_group"))

    # Get samples per group
    counts = data.group_by("strat_group").agg(pl.count())
    counts = counts.with_columns(
        (pl.col("count") / data.height * sample_size).round().cast(pl.Int64).alias("target"))
    counts = counts.with_columns(pl.max_horizontal(
        pl.col("target"), 1).alias("target"))

    print("ENTERING FOR:")
    results = []
    print(counts)
    # Define threshold for pre-sampling
    large_group_threshold = 70000  # 1 million
    pre_sample_size = 30000  # 100k

    for row in counts.to_dicts():
        group_data = data.filter(pl.col("strat_group") == row["strat_group"])
        target = min(row["target"], row["count"])
        print(
            f"Processing group: {row['strat_group']} with {row['count']} rows")
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
                    group_data, strat_cols, sample_size=pre_sample_size, random_seed=random_seed)
            else:
                # Fall back to random sampling if no stratification columns
                group_data = group_data.sample(
                    n=pre_sample_size, seed=random_seed)

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
            print("RUNNING SCAN")
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

    # Now concat will work since all DataFrames have the same columns
    # This should be OUTSIDE the for loop, with proper indentation
    result = result.drop("strat_group")
    result = pl.concat(results)

    # Drop the stratification column

    output_path = "hdbscan_sampled_data.csv"
    print(f"Saving results to {output_path}")
    result.write_csv(output_path)

    print(f"Created {result.height} row HDBSCAN sample")
    return result


def main():
    file_name = 'data'
    df = load_data_from_csv_parquet_format(file_name)

    if df is not None:  # TODO DONT RUN NEEDS CHANGES TO HIGH COUNTS , Tha kanoume stratified sampling sta megal prin ksekinisei to kmeans kai hdbscan
        # hdbscan_dataset = hdbscan_sampling(df)
        kmeans_dataset = kmeans_sampling(df)

        # calculate_new_data_set_preservation_statistics(
        #     df, kmeans_dataset, 'kMeans')
        # calculate_new_data_set_preservation_statistics(
        #     df, hdbscan_dataset, 'hdbscan')


if __name__ == "__main__":
    main()
