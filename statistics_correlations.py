import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# CORRECT AND WORKS
def plot_statistics_correlation_heatmap():
    os.makedirs('statistics_correlations', exist_ok=True)

    file = 'data_analysis/data_description.csv'
    df = pd.read_csv(file)
    df = df.set_index('statistic') # Set 'statistic' column as index
    df = df.select_dtypes(include=['float64', 'int64']) # Select only numeric columns
    df = df.loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']] # row selection
    df = df.iloc[1:]  # Remove the first row

    correlation_between_stats = df.T.corr() # T: Transpose to get correlation between statistics

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_between_stats, 
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                square=True,
                linewidths=0.5)

    plt.title(f'Correlation Heatmap of Statistics')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Save the plot
    output_path = f'statistics_correlations/stats_correlations_heatmap.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()


# CORRECT AND WORKS
#COMPUTES THE CORRELATION BETWEEN THE AGGREGATE OF STATISTICS AND IS PROBABLY UNECESSARY
def aggregate_statistics_correlations():
    data = 'data_analysis/data_description.csv'
    df = pd.read_csv(data)

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    # Calculate aggregate statistics for each column
    aggregates = pd.DataFrame({
        'count': numeric_df.count(), #or count average ?
        'mean': numeric_df.mean(),
        'std': numeric_df.std(),
        'min': numeric_df.min(),
        'max': numeric_df.max(),
    })

    # Calculate correlation between the aggregate statistics
    aggr_stat_corr = aggregates.corr()

    # Plot the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(aggr_stat_corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Between Aggregate Statistics')
    plt.savefig('aggregate_statistics_correlations.png', bbox_inches='tight', dpi=300)


# SIMILAR TO plot_correlation_heatmap() AND PROBABLY NOT NEEDED
#DOES NOT WORK, ERROR: Unable to allocate 4.90 GiB for an array with shape (76, 8656767) and data type float64
def correlations_heatmap_by_group():
    data = 'data.csv'
    df = pd.read_csv(data)
    with open('grouping.json', 'r') as f:
        groups = json.load(f)

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    os.makedirs('correlations_by_group', exist_ok=True)

    # Create correlation heatmap for each group
    for group_name, metrics in groups.items():

        correlation_matrix = numeric_df[metrics].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, 
                    annot=True,
                    cmap='coolwarm',
                    fmt='.2f',
                    square=True,
                    linewidths=0.5)
        
        plt.title(f'Correlation Heatmap of {group_name}')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Save the plot
        output_path = f'correlations_by_group/correlations_heatmap_of_{group_name}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Also save the correlation values to a text file
        text_output = f'correlations_by_group/correlations_matrix_of_{group_name}.txt'
        with open(text_output, 'w') as f:
            f.write(f"Correlation Matrix for {group_name}\n\n")
            f.write(correlation_matrix.to_string())
        
        print(f"Created heatmap and matrix for {group_name}")
