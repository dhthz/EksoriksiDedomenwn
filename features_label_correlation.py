import polars as pl
import numpy as np
from scipy.stats import pointbiserialr, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_problematic_features(file_path='data_description_analysis.csv'):
    """
    Identify features with high standard deviation or high maximum values
    
    Args:
        file_path (str): Path to the CSV file with feature statistics
        
    Returns:
        list: Features flagged with high std or high max values
    """
    try:
        # Using pandas to maintain index_col functionality
        import pandas as pd
        df = pd.read_csv(file_path, index_col='statistic')
        
        # Get features with 'yes' in either large_std or max_too_high rows
        problematic_features = []
        
        # Check if 'large_std' is in the index
        if 'large_std' in df.index:
            large_std_cols = df.loc['large_std']
            problematic_features.extend(large_std_cols[large_std_cols == 'yes'].index.tolist())
            
        # Check if 'max_too_high' is in the index
        if 'max_too_high' in df.index:
            max_too_high_cols = df.loc['max_too_high']
            problematic_features.extend(max_too_high_cols[max_too_high_cols == 'yes'].index.tolist())
        
        return list(set(problematic_features))  # Remove duplicates
        
    except Exception as e:
        logger.error(f"Error loading problematic features: {e}")
        return []


def analyze_feature_label_correlation(df, features, label_column, visualize=False, output_dir='plots'):
    """
    Analyze correlation between features and a binary label
    
    Args:
        df (pl.DataFrame): Input data
        features (list): List of feature names to analyze
        label_column (str): Name of the column containing binary labels
        visualize (bool): Whether to create visualizations
        output_dir (str): Directory for saving plots
        
    Returns:
        pl.DataFrame: Results of correlation analysis
    """
    results = []
    
    # Check if label column exists
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame")
    
    # Define explicit label mapping
    label_mapping = {'Benign': 0, 'Malicious': 1}
    
    # Verify all labels are either Benign or Malicious
    unique_labels = df.select(label_column).unique().to_series().to_list()
    unknown_labels = set(unique_labels) - set(label_mapping.keys())
    if unknown_labels:
        raise ValueError(f"Found unexpected labels: {unknown_labels}")
    
    # Encode labels using explicit mapping with Polars
    df = df.with_columns(
        pl.col(label_column).map_dict(label_mapping).alias('encoded_label')
    )
    
    # Create output directory for plots if needed
    if visualize:
        Path(output_dir).mkdir(exist_ok=True)
    
    logger.info(f"Analyzing {len(features)} features for correlation with '{label_column}'")
    logger.info(f"Label mapping: {label_mapping}")

    for col in features:
        if col not in df.columns:
            logger.warning(f"Feature '{col}' not found in DataFrame, skipping")
            continue

        try:
            # Filter to rows where both the feature and label are not null
            valid_data = df.filter(
                pl.col(col).is_not_null() & 
                pl.col('encoded_label').is_not_null()
            )
            
            if valid_data.height < 10:  # Require at least 10 valid samples
                logger.warning(f"Feature '{col}' has insufficient valid data ({valid_data.height} rows), skipping")
                continue

            # Extract numpy arrays for calculations
            feature_array = valid_data.select(pl.col(col)).to_numpy().flatten()
            label_array = valid_data.select(pl.col('encoded_label')).to_numpy().flatten()
            
            # Point-biserial correlation
            corr, corr_p = pointbiserialr(feature_array, label_array)

            # T-test between the two groups
            benign_values = valid_data.filter(pl.col(label_column) == 'Benign').select(pl.col(col))
            malicious_values = valid_data.filter(pl.col(label_column) == 'Malicious').select(pl.col(col))
            
            group0 = benign_values.to_numpy().flatten()
            group1 = malicious_values.to_numpy().flatten()
            
            if len(group0) == 0 or len(group1) == 0:
                logger.warning(f"Feature '{col}' has no data for one of the labels, skipping")
                continue
                
            t_stat, t_p = ttest_ind(group0, group1, equal_var=False)

            # Calculate mean difference and standard deviations
            mean_diff = abs(np.mean(group0) - np.mean(group1))
            std_benign = np.std(group0)
            std_malicious = np.std(group1)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(group0) - 1) * std_benign**2 + 
                                  (len(group1) - 1) * std_malicious**2) / 
                                 (len(group0) + len(group1) - 2))
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            if visualize:
                # Create a more informative plot
                plt.figure(figsize=(12, 8))
                
                # Create boxplot
                ax1 = plt.subplot(211)
                sns.boxplot(
                    x=valid_data.select(pl.col(label_column)).to_series(),
                    y=valid_data.select(pl.col(col)).to_series(),
                    ax=ax1
                )
                ax1.set_title(f'{col} Distribution by {label_column}')
                ax1.set_xlabel(label_column)
                ax1.set_ylabel(col)
                
                # Add statistical annotation
                ax1.annotate(
                    f"Cohen's d: {cohens_d:.2f}\np-value: {t_p:.4f}\nCorrelation: {corr:.4f}",
                    xy=(0.02, 0.95), 
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
                )
                
                # Create histogram
                ax2 = plt.subplot(212)
                # Plot benign distribution
                sns.histplot(
                    benign_values.to_series(),
                    color='green',
                    alpha=0.5,
                    label='Benign',
                    kde=True,
                    ax=ax2
                )
                # Plot malicious distribution
                sns.histplot(
                    malicious_values.to_series(),
                    color='red',
                    alpha=0.5,
                    label='Malicious',
                    kde=True,
                    ax=ax2
                )
                ax2.set_title(f'Histogram of {col} by {label_column}')
                ax2.set_xlabel(col)
                ax2.legend()
                
                plt.tight_layout()
                safe_filename = col.replace(' ', '_').replace('/', '_').replace('\\', '_')
                plt.savefig(f"{output_dir}/analysis_{safe_filename}.png", dpi=300)
                plt.close()

            results.append({
                'feature': col,
                'point_biserial_corr': round(corr, 4),
                'corr_p_value': round(corr_p, 6),
                't_test_p_value': round(t_p, 6),
                'mean_diff': round(mean_diff, 4),
                'cohens_d': round(cohens_d, 4),
                'benign_mean': round(np.mean(group0), 4),
                'malicious_mean': round(np.mean(group1), 4),
                'benign_std': round(std_benign, 4),
                'malicious_std': round(std_malicious, 4),
                'benign_count': len(group0),
                'malicious_count': len(group1)
            })

        except Exception as e:
            logger.error(f"Error processing {col}: {str(e)}")
            continue

    if not results:
        logger.warning("No valid feature-label correlations were found")
        return pl.DataFrame(
            schema={
                'feature': pl.Utf8,
                'point_biserial_corr': pl.Float64,
                'corr_p_value': pl.Float64, 
                't_test_p_value': pl.Float64,
                'mean_diff': pl.Float64,
                'cohens_d': pl.Float64,
                'benign_mean': pl.Float64,
                'malicious_mean': pl.Float64,
                'benign_std': pl.Float64,
                'malicious_std': pl.Float64,
                'benign_count': pl.Int64,
                'malicious_count': pl.Int64
            }
        )
    
    # Create Polars DataFrame with results
    result_df = pl.DataFrame(results)
    
    # Sort by absolute correlation value
    result_df = result_df.with_columns(
        pl.col('point_biserial_corr').abs().alias('abs_corr')
    ).sort('abs_corr', descending=True).drop('abs_corr')
    
    return result_df


def feature_importance_plot(result_df, top_n=15, output_path='feature_importance.png'):
    """
    Create a feature importance plot based on correlation results
    
    Args:
        result_df (pl.DataFrame): Results dataframe from analyze_feature_label_correlation
        top_n (int): Number of top features to display
        output_path (str): Path to save the plot
    """
    if result_df.height == 0:
        logger.warning("No data available for feature importance plot")
        return
        
    plt.figure(figsize=(12, 10))
    
    # Get top N features by absolute correlation
    top_features = result_df.with_columns(
        pl.col('point_biserial_corr').abs().alias('abs_corr')
    ).sort('abs_corr', descending=True).head(top_n)
    
    # Convert to pandas for easier plotting
    top_features_pd = top_features.to_pandas()
    
    # Create plot
    ax = sns.barplot(
        x='point_biserial_corr',
        y='feature',
        data=top_features_pd,
        palette=sns.color_palette("RdBu_r", n_colors=len(top_features_pd))
    )
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add p-value annotations
    for i, row in enumerate(top_features.to_dicts()):
        significant = "***" if row['corr_p_value'] < 0.001 else "**" if row['corr_p_value'] < 0.01 else "*" if row['corr_p_value'] < 0.05 else ""
        ax.text(
            0.01 if row['point_biserial_corr'] < 0 else -0.01,
            i,
            significant,
            ha='left' if row['point_biserial_corr'] < 0 else 'right',
            va='center',
            fontweight='bold'
        )
    
    plt.title(f'Top {top_n} Features by Point-Biserial Correlation with Label')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Feature importance plot saved to {output_path}")


def main():
    # Set up output directories
    output_dir = 'analysis_output'
    plots_dir = f'{output_dir}/plots'
    Path(output_dir).mkdir(exist_ok=True)
    Path(plots_dir).mkdir(exist_ok=True)
    
    # Get problematic features
    logger.info("Identifying problematic features")
    problematic_features = get_problematic_features()
    
    if not problematic_features:
        logger.error("No problematic features found. Check the data_description_analysis.csv file.")
        return
        
    logger.info(f"Found {len(problematic_features)} problematic features")
    
    # Load data
    logger.info("Loading dataset from data.parquet")
    try:
        necessary_columns = ['Label'] + problematic_features
        df = pl.read_parquet('data.parquet', columns=necessary_columns)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Identify numeric columns by trying to cast them
    numeric_cols = []
    for col in df.columns:
        if col != 'Label':
            try:
                # Test if column can be cast to numeric
                _ = df.select(pl.col(col).cast(pl.Float64))
                numeric_cols.append(col)
            except:
                logger.info(f"Skipping non-numeric column '{col}'")
    
    logger.info(f"Found {len(numeric_cols)} numeric columns")
    
    # Filter problematic features to only include numeric columns
    valid_features = [f for f in problematic_features if f in numeric_cols]
    logger.info(f"Analyzing {len(valid_features)} valid problematic features")
    
    # Cast only the numeric columns to Float64
    df = df.with_columns([
        pl.col(col).cast(pl.Float64) for col in numeric_cols
    ])
    
    # Analyze feature-label correlations
    result_df = analyze_feature_label_correlation(
        df, 
        valid_features, 
        label_column='Label', 
        visualize=True,
        output_dir=plots_dir
    )
    
    # Save results
    result_path = f'{output_dir}/feature_label_correlations.csv'
    result_df.write_csv(result_path)
    logger.info(f"Results saved to '{result_path}'")
    
    # Show top correlated features
    logger.info("\nTop 10 most label-correlated features:")
    print(result_df.head(10))

    # Display additional insights
    significant_features = result_df.filter(pl.col('corr_p_value') < 0.05)
    logger.info(f"\nNumber of statistically significant features (p < 0.05): {significant_features.height}")
    
    # Create feature importance plot
    feature_importance_plot(
        result_df, 
        top_n=20, 
        output_path=f'{output_dir}/feature_importance.png'
    )
    
    # Create metrics summary
    strong_corr_features = result_df.filter(pl.col('point_biserial_corr').abs() > 0.3).height
    moderate_corr_features = result_df.filter(
        (pl.col('point_biserial_corr').abs() > 0.1) & 
        (pl.col('point_biserial_corr').abs() <= 0.3)
    ).height
    
    logger.info("\nSummary:")
    logger.info(f"- Features with strong correlation (|r| > 0.3): {strong_corr_features}")
    logger.info(f"- Features with moderate correlation (0.1 < |r| <= 0.3): {moderate_corr_features}")
    logger.info(f"- Statistically significant features (p < 0.05): {significant_features.height}")
    
    # Export a comprehensive report
    with open(f'{output_dir}/analysis_report.txt', 'w') as f:
        f.write("Feature-Label Correlation Analysis Report\n")
        f.write("========================================\n\n")
        f.write(f"Total features analyzed: {len(valid_features)}\n")
        f.write(f"Features with strong correlation (|r| > 0.3): {strong_corr_features}\n")
        f.write(f"Features with moderate correlation (0.1 < |r| <= 0.3): {moderate_corr_features}\n")
        f.write(f"Statistically significant features (p < 0.05): {significant_features.height}\n\n")
        
        f.write("Top 10 features by correlation strength:\n")
        for row in result_df.head(10).to_dicts():
            f.write(f"- {row['feature']}: r={row['point_biserial_corr']}, p={row['corr_p_value']}, Cohen's d={row['cohens_d']}\n")
            
    logger.info(f"Analysis complete. Full report saved to '{output_dir}/analysis_report.txt'")


if __name__ == "__main__":
    main()