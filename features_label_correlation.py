import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

def get_problematic_features():
    """
    Identify features with high standard deviation or high maximum values
    Returns two lists: features with high std and features with high max
    """
    df = pd.read_csv('data_description_analysis.csv', index_col='statistic')
    
     # Get features with 'yes' in either large_std or max_too_high rows
    problematic_features = df.columns[
        (df.loc['large_std'] == 'yes') | 
        (df.loc['max_too_high'] == 'yes')
    ].unique().tolist()
    
    return problematic_features


def analyze_feature_label_correlation(df, features, label_column, visualize=False):

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
    label_array = (df.select(pl.col(label_column))
                    .to_series()
                    .replace(label_mapping)  # Using replace instead of map_dict
                    .to_numpy())
    print(f"Label mapping: {label_mapping}")

    for col in features:
        if col not in df.columns:
            print(f"Warning: Feature '{col}' not found in DataFrame, skipping")
            continue

        try:
            # Extract feature as Float64
            feature_array = df.select(pl.col(col).cast(pl.Float64)).to_series().to_numpy()
            
            # Handle NaN values
            valid_indices = ~np.isnan(feature_array)
            clean_feature = feature_array[valid_indices]
            clean_label = label_array[valid_indices]
            
            if len(clean_feature) < 2:
                print(f"Warning: Feature '{col}' has insufficient valid data, skipping")
                continue

            # Point-biserial correlation
            corr, corr_p = pointbiserialr(clean_feature, clean_label)

            # T-test
            group0 = clean_feature[clean_label == unique_labels[0]]
            group1 = clean_feature[clean_label == unique_labels[1]]
            
            if len(group0) == 0 or len(group1) == 0:
                print(f"Warning: Feature '{col}' has no data for one of the labels, skipping")
                continue
                
            t_stat, t_p = ttest_ind(group0, group1, equal_var=False)

            # Mean difference
            mean_diff = abs(np.mean(group0) - np.mean(group1))

            if visualize:
                plt.figure(figsize=(10, 6))
                # Create a DataFrame for seaborn with just the needed data
                plot_data = {
                    'label': clean_label,
                    'feature': clean_feature
                }
                sns.boxplot(x='label', y='feature', data=plot_data)
                plt.title(f'{col} vs {label_column}')
                plt.xlabel(label_column)
                plt.ylabel(col)
                plt.savefig(f"boxplot_{col.replace(' ', '_').replace('/', '_')}.png")
                plt.close()

            results.append({
                'feature': col,
                'point_biserial_corr': round(corr, 4),
                'corr_p_value': round(corr_p, 4),
                't_test_p_value': round(t_p, 4),
                'mean_diff': round(mean_diff, 4)
            })

        except Exception as e:
            print(f"Error processing {col}: {str(e)}")
            continue

    if not results:
        print("No valid feature-label correlations were found")
        return pl.DataFrame(
            schema={
                'feature': pl.Utf8,
                'point_biserial_corr': pl.Float64,
                'corr_p_value': pl.Float64, 
                't_test_p_value': pl.Float64,
                'mean_diff': pl.Float64
            }
        )
    
    # Create Polars DataFrame with results
    result_df = pl.DataFrame(results)
    
    # Sort by absolute correlation value
    result_df = result_df.with_columns(
        pl.col('point_biserial_corr').abs().alias('abs_corr')
    ).sort('abs_corr', descending=True).drop('abs_corr')
    
    return result_df

def main():
    problematic_features = get_problematic_features()
    # print("\nProblematic features (high std or high max):")
    # for feature in problematic_features:
    #     print(f"- {feature}")

     # Load data and handle numeric columns
    df = pl.read_parquet('data.parquet')
    
    # Identify numeric columns by trying to cast them
    numeric_cols = []
    for col in df.columns:
        if col != 'Label':
            try:
                # Test if column can be cast to numeric
                _ = df.select(pl.col(col).cast(pl.Float64))
                numeric_cols.append(col)
            except:
                print(f"Warning: Skipping non-numeric column '{col}'")
    
    # Cast only the numeric columns to Float64
    df = df.with_columns([
        pl.col(col).cast(pl.Float64) for col in numeric_cols
    ])
    
    result_df = analyze_feature_label_correlation(df, problematic_features, label_column='Label', visualize=True)
    result_df.write_csv('feature_label_correlations.csv')
    print("\nResults saved to 'feature_label_correlations.csv'")
   
    print(result_df.head(10))  # Top 10 most label-correlated features

    # Additional insights
    # significant_features = result_df.filter(pl.col('corr_p_value') < 0.05)
    # print(f"\nNumber of statistically significant features (p < 0.05): {significant_features.height}")

if __name__ == "__main__":
    main()