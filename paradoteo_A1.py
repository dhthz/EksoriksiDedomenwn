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
    except pl.errors.EmptyDataError:
        print(f"No data: {file_name} is empty")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
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


# Den thumamai ti kanei auth mallon den xreiazetai alla thn afhnw edw gia na uparxei
def export_plot_findings(data, column_name, output_folder):
    """
    Export statistical findings from plots to text file
    """
    findings_path = os.path.join(output_folder, "findings.txt")

    with open(findings_path, 'w') as f:
        f.write(f"Analysis of distributions grouped by {column_name}\n")
        f.write("=" * 50 + "\n\n")

        # For each numeric column
        numeric_data = data.select(
            [col for col, dtype in data.schema.items()
             if dtype in (pl.Float64, pl.Int64)]
        )

        for col in numeric_data.columns:
            f.write(f"\nColumn: {col}\n")
            f.write("-" * 30 + "\n")

            # Calculate statistics per category
            for category in data.select(column_name).unique().to_series().to_list():
                category_data = data.filter(
                    pl.col(column_name) == category).select(col)

                if category_data.is_empty:
                    f.write(f"\nNo data for category {category}\n")
                    continue

                # Basic statistics
                stats = category_data.describe()
                f.write(f"\n{category} statistics:\n")
                f.write(f"Mean: {stats['mean'][0]:.2f}\n")
                f.write(f"Std: {stats['std'][0]:.2f}\n")
                f.write(f"Min: {stats['min'][0]:.2f}\n")
                f.write(f"Max: {stats['max'][0]:.2f}\n")
                f.write(f"Count: {stats['count'][0]:.2f}\n")

                # Density calculation
                benign_data = data.filter(
                    (pl.col(column_name) == category) & (pl.col('Label') == 0))
                malicious_data = data.filter(
                    (pl.col(column_name) == category) & (pl.col('Label') == 1))

                if not benign_data.is_empty and not malicious_data.is_empty:
                    benign_density = len(benign_data) / \
                        (len(benign_data) + len(malicious_data))
                    malicious_density = len(
                        malicious_data) / (len(benign_data) + len(malicious_data))
                    density_ratio = malicious_density / \
                        benign_density if benign_density != 0 else float('inf')

                    f.write(
                        f"Density ratio (Malicious/Benign): {density_ratio:.2f}\n")

            f.write("\n" + "=" * 50 + "\n")


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
def plot_correlation_heatmap(data, category):
    """
    Creates and saves correlation heatmap including Label correlations.
    Args: data (pl.DataFrame): Dataset with Label column
    """
    print(f"Input columns: {data.columns}")

    # Convert Label to numeric
    if 'Label' in data.columns:
        data = data.with_columns(
            pl.when(pl.col('Label') == 'Benign')
            .then(0)
            .otherwise(1)
            .alias('Label')
        )

    # Calculate correlation matrix and convert to pandas with names
    corr_matrix = data.corr().to_pandas()

    # Ensure index and columns match
    corr_matrix.index = data.columns
    corr_matrix.columns = data.columns

    # Create heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                square=True,
                linewidths=0.5,
                xticklabels=data.columns,  # Use column names
                yticklabels=data.columns)  # Use column names

    plt.title(f'Feature Correlation Heatmap - {category}', pad=20, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Save plot
    output_folder = "correlations"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(
        output_folder, f"correlation_heatmap_{category}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {output_path}")

    # Add text output
    corr_matrix = data.corr().to_pandas()

    # Save correlation values to text file
    output_folder = "correlations"
    os.makedirs(output_folder, exist_ok=True)

    text_output = os.path.join(
        output_folder, f"correlation_matrix_{category}.txt")
    with open(text_output, 'w') as f:
        f.write(f"Correlation Matrix for {category}\n")
        f.write("Label: 0=Benign, 1=Malicious\n\n")
        f.write(corr_matrix.to_string())

    print(f"Saved correlation matrix to {text_output}")

    """
    Plots improved histograms for numeric columns with log scale option.
    Args: data (pl.DataFrame): Dataset
    """
    # Select numeric columns
    numeric_data = data.select(
        [col for col, dtype in data.schema.items() if dtype in (pl.Float64, pl.Int64)]
    )

    # Create output folder
    output_folder = "histograms"
    os.makedirs(output_folder, exist_ok=True)

    # Convert to pandas
    df = numeric_data.to_pandas()

    # Loop over numeric columns
    for col in df.columns:
        print(f"Plotting histogram for: {col}")

        # Skip if empty
        if df[col].dropna().empty:
            print(f"Skipping {col} - no data")
            continue

        # Create figure with two subplots - normal and log scale
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # Calculate range information
        data_range = df[col].max() - df[col].min()

        # Adjust bins based on data distribution
        if data_range > 0:
            bins = min(50, max(10, int(np.sqrt(len(df[col])))))
        else:
            bins = 30

        # Plot normal scale
        ax1.hist(df[col], bins=bins, color='blue', alpha=0.7)
        ax1.set_title(f"Distribution of {col} (Linear Scale)")
        ax1.grid(True, alpha=0.3)

        # Add statistical annotations
        mean = df[col].mean()
        median = df[col].median()
        ax1.axvline(mean, color='red', linestyle='dashed', linewidth=1,
                    label=f'Mean: {mean:.2f}')
        ax1.axvline(median, color='green', linestyle='dashed', linewidth=1,
                    label=f'Median: {median:.2f}')
        ax1.legend()

        plt.tight_layout()

        # Save plot
        sanitized_col = sanitize_filename(col)
        output_path = os.path.join(
            output_folder, f"{sanitized_col}_histogram.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300,
                    pad_inches=0.3)  # Added padding for annotations
        plt.close()
        print(f"Saved histogram to {output_path}")


# HELPER FUNCTION gia ta counts twn label klp
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


def plot_flag_distributions(data):  # TODO mallon thelei diagrafh auto
    """
    Creates bar plots for flag columns showing counts grouped by Label
    """
    # Select flag columns
    flag_columns = [
        'FIN Flag Count', 'SYN Flag Count', 'PSH Flag Count',
        'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count',
        'ECE Flag Count'
    ]

    # Create output folder
    output_folder = "flag_distributions"
    os.makedirs(output_folder, exist_ok=True)

    # Convert to pandas
    df = data.to_pandas()

    for flag in flag_columns:
        plt.figure(figsize=(12, 6))

        # Create grouped bar plot
        df_grouped = df.groupby(['Label', flag])[flag].count().unstack()
        df_grouped.plot(kind='bar', stacked=True)

        plt.title(f'Distribution of {flag} by Label')
        plt.xlabel('Label')
        plt.ylabel('Count')

        # Add value labels on bars
        for container in plt.gca().containers:
            plt.bar_label(container)

        plt.xticks(rotation=45)
        plt.legend(title=flag)
        plt.tight_layout()

        # Save plot
        output_path = os.path.join(
            output_folder, f"{sanitize_filename(flag)}_distribution.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved flag distribution to {output_path}")


def calculate_correlation_matrix(data):  # WORKING
    columns_to_drop_NO_VARIANCE = [
        "Bwd PSH Flags",
        "Bwd URG Flags",
        "Fwd Bytes/Bulk Avg",
        "Fwd Packet/Bulk Avg",
        "Fwd Bulk Rate Avg"
    ]

    # Drop from the main DataFrame
    data = data.drop(columns_to_drop_NO_VARIANCE)

    numeric_data = data.select(
        [col for col, dtype in data.schema.items() if dtype in (pl.Float64, pl.Int64)]
    )
    for col in numeric_data.columns:
        # Calculate variance for the column
        variance = numeric_data.select(pl.col(col)).var(
        ).to_numpy().item()  # Convert to scalar value

    # Check if variance is zero
        if variance == 0:
            print(f"Column {col} has zero variance.")

    corr_matrix = numeric_data.corr()

    # Save the correlation matrix to a .txt file
    # Convert the Polars DataFrame to a Pandas DataFrame
    corr_matrix_pd = corr_matrix.to_pandas()
    # Save the correlation matrix to a .txt file using pandas
    corr_matrix_pd.to_csv('correlation_matrix.txt',
                          sep='\t', header=True, index=True)

    print("Correlation matrix saved to 'correlation_matrix.txt'")


def main():
    file_name = 'data'
    df = load_data_from_csv_parquet_format(file_name)

    if df is not None:
        # 1. Analyze data
        # count_categories_in_column(df, 'Label')
        # plot_histograms_grouped_by_column(df, 'Label') # Grouped Histograms by column Label
        # plot_histograms_grouped_by_column(df) # Ungrouped Histograms
        # calculate_correlation_matrix(df)

        # Grouped Boxplots by column Label
        #plot_boxplots_grouped_by_column(df, 'Label')

        # Ungrouped Boxplots
        plot_boxplots_grouped_by_column(df)


if __name__ == "__main__":
    main()
