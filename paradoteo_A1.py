import os
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def load_data_from_csv_parquet_format(file_name):
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


def analyze_data(data):
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


def plot_histograms_grouped_by_column(data, column_name):
    """
    Plots overlapping histograms for numeric columns grouped by category.
    Args:
        data (pl.DataFrame): Dataset
        column_name (str): Column to group by
    """
    # Select numeric columns
    numeric_data = data.select(
        [col for col, dtype in data.schema.items() if dtype in (pl.Float64, pl.Int64)]
    )

    # Convert to Pandas
    df = data.to_pandas()

    # Create output folder
    output_folder = f"grouped_histograms_by_{sanitize_filename(column_name)}"
    os.makedirs(output_folder, exist_ok=True)

    # Get unique categories
    categories = df[column_name].unique()
    colors = ['blue', 'red', 'green', 'orange', 'purple']  # Add more if needed

    # Loop over numeric columns
    for col in numeric_data.columns:
        print(f"Plotting grouped histogram for: {col}")

        plt.figure(figsize=(12, 6))

        # Plot histogram for each category
        for idx, category in enumerate(categories):
            category_data = df[df[column_name] == category][col]
            if len(category_data) > 0:
                plt.hist(category_data, bins=50, alpha=0.5,
                         label=str(category), color=colors[idx % len(colors)],
                         density=True)  # Use density for better comparison

        plt.title(f"Distribution of {col} by {column_name}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()

        # Save plot with sanitized names
        sanitized_col = sanitize_filename(col)
        output_path = os.path.join(
            output_folder, f"{sanitized_col}_grouped_histogram.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved grouped histogram to {output_path}")


def plot_boxplots_grouped_by_column(data, column_name):
    """
    Plots boxplots for all numeric columns grouped by the column_name.
    Args:
        - data (pl.DataFrame): Dataset
        - column_name (str): Column to group by
    """
    # Select numeric columns
    numeric_data = data.select(
        [col for col, dtype in data.schema.items() if dtype in (pl.Float64, pl.Int64)]
    )

    # Convert to Pandas
    df = data.to_pandas()

    # Create sanitized output folder name
    output_folder = f"boxplots_by_{sanitize_filename(column_name)}"
    os.makedirs(output_folder, exist_ok=True)

    # Loop over numeric columns
    for col in numeric_data.columns:
        print(f"Plotting boxplot for: {col}")

        # Skip empty columns
        if df[col].dropna().empty:
            print(f"Skipping column '{col}' because it has no data.")
            continue

        # Create plot
        plt.figure(figsize=(10, 5))
        df.boxplot(column=col, by=column_name, vert=False, grid=False,
                   patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.7))
        plt.title(f"Boxplot of {col} by {column_name}")
        plt.suptitle("")
        plt.xlabel(col)
        plt.ylabel(column_name)

        # Save plot with sanitized names
        sanitized_col = sanitize_filename(col)
        sanitized_category = sanitize_filename(column_name)
        output_path = os.path.join(
            output_folder, f"{sanitized_col}_by_{sanitized_category}_boxplot.png"
        )
        plt.savefig(output_path)
        plt.close()
        print(f"Saved boxplot to {output_path}")


def plot_correlation_heatmap(data):
    return


def main():
    file_name = 'data'  # Name of CSV
    df = load_data_from_csv_parquet_format(file_name)  # Load the Data
    grouping_columns = [
        'Label',
        'Traffic Type',
        'Traffic Subtype',
        'Protocol',
        'FIN Flag Count',
        'SYN Flag Count',
        'PSH Flag Count',
        'ACK Flag Count',
        'URG Flag Count'
    ]
    if df is not None:
        # analyze_data(df)
        # plot_all_histograms(df)
        for col in grouping_columns:
            plot_boxplots_grouped_by_column(df, col)
            plot_histograms_grouped_by_column(df, col)
        # plot_correlation_heatmap(df)


if __name__ == "__main__":
    main()
