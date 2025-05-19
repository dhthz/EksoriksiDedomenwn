import os
import polars as pl
import math
import numpy as np
from paradoteo_A1 import load_data_from_csv_parquet_format


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

            # Jensen-Shannon divergence approximation (simpler than KL divergence)
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

        if overall_score >= 95:
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


def main():
    file_name = 'data'
    df = load_data_from_csv_parquet_format(file_name)

    if df is not None:
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

        # Perform stratified sampling
        sample_size = 10000
        columns_to_stratify = [
            'Label', 'Traffic Type'] if 'Traffic Type' in df.columns else ['Label']

        sampled_df = stratified_sampling_multicolumn(
            df,
            columns=columns_to_stratify,
            sample_size=sample_size
        )

        if sampled_df is not None:
            # Validate information preservation
            validation_metrics = validate_dataset_quality(
                df_before_sampling, sampled_df)

            # Save results
            sampled_df.write_parquet("sampled_data_10k.parquet")
            print(
                f"Saved {sampled_df.height:,} samples to sampled_data_10k.parquet")

            # Save validation metrics for reference
            with open("validation_metrics.txt", "w") as f:
                f.write("Dataset Information Preservation Metrics:\n")
                f.write("=======================================\n\n")
                for key, value in validation_metrics.items():
                    f.write(f"{key}: {value:.2f}%\n")
                f.write(
                    f"\nReduction ratio: {df.height / sampled_df.height:.0f}:1")
            print("Validation metrics saved to validation_metrics.txt")


if __name__ == "__main__":
    main()
