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


def analyze_feature_label_correlation(df, features, label_column='label', visualize=False):
    results = []

    for col in df.select_dtypes(include=[np.number]).columns:
        if col == features:
            continue

        try:
            # Extract feature and labels
            feature = df[col].dropna()
            labels = df.loc[feature.index, label_column]

            # Point-biserial correlation
            corr, corr_p = pointbiserialr(feature, labels)

            # T-test between benign (0) and attack (1)
            group0 = feature[labels == 0]
            group1 = feature[labels == 1]
            t_stat, t_p = ttest_ind(group0, group1, equal_var=False)

            # Mean difference
            mean_diff = abs(group0.mean() - group1.mean())

            results.append({
                'feature': col,
                'point_biserial_corr': round(corr, 4),
                'corr_p_value': round(corr_p, 4),
                't_test_p_value': round(t_p, 4),
                'mean_diff': round(mean_diff, 4)
            })

            # Optional: show plot
            if visualize:
                sns.boxplot(x=labels, y=feature)
                plt.title(f"{col} vs. {label_column}")
                plt.xlabel("Label (0 = benign, 1 = attack)")
                plt.ylabel(col)
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Skipping {col} due to error: {e}")
            continue

    # Convert results to DataFrame and sort by correlation
    result_df = pd.DataFrame(results)
    return result_df.sort_values(by='point_biserial_corr', key=abs, ascending=False)

def main():
    problematic_features = get_problematic_features()
    print("\nProblematic features (high std or high max):")
    for feature in problematic_features:
        print(f"- {feature}")

    df = pl.read_csv('data.csv')
    result_df = analyze_feature_label_correlation(df, problematic_features, label_column='label', visualize=True)

    print(result_df.head(10))  # Top 10 most label-correlated features


if __name__ == "__main__":
    main()