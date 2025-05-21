import pandas as pd
import shutil

# Create a copy of the original file
original_file = 'data_analysis/data_description.csv'
analysis_file = 'data_description_analysis.csv'  # This will save in the current EksoriksiDedomenwn folder

# Create a copy of the original file
shutil.copy2(original_file, analysis_file)

# Read the copied file
df = pd.read_csv(analysis_file)
n_rows = len(df)

mean_max_min_equal = {'statistic' : 'mean_max_min_equal'}
mean_median_gap = {'statistic' : 'mean_median_gap'}
large_std = {'statistic' : 'large_std'}
max_too_high = {'statistic' : 'max_too_high'}
long_tail_check = {'statistic' : 'long_tail_check'}

report = []

for col in df.columns[1:]:
    if pd.api.types.is_numeric_dtype(df[col]):
        try:
            print(f"Processing column: {col}")
            #median_value = df.loc[df['statistic'] == 'median', col].values[0]
            mean_value = df.loc[df['statistic'] == 'mean', col].values[0]
            std_value = df.loc[df['statistic'] == 'std', col].values[0]
            min_value = df.loc[df['statistic'] == 'min', col].values[0]
            max_value = df.loc[df['statistic'] == 'max', col].values[0]
            value_of_25 = df.loc[df['statistic'] == '25%', col].values[0]
            value_of_50 = df.loc[df['statistic'] == '50%', col].values[0]
            value_of_75 = df.loc[df['statistic'] == '75%', col].values[0]
            
            # Check if mean, max, and min are equal
            if mean_value == max_value and mean_value == min_value:
                mean_max_min_equal[col] = 'yes'
            else:
                mean_max_min_equal[col] = 'no'

            gap = abs(mean_value - value_of_50)
            if gap > std_value:
                mean_median_gap[col] = 'large gap'
            else:
                mean_median_gap[col] = 'small gap'
                
            if std_value >= 0.5 * mean_value:
                large_std[col] = 'yes'
            else:
                large_std[col] = 'no'

            iqr = value_of_75 - value_of_25
            if max_value > 5 * value_of_50:
                max_too_high[col] = 'yes'
            else:
                max_too_high[col] = 'no'

            if max_value > value_of_75 + 1.5 * iqr and min_value < value_of_25 - 1.5*iqr:
                long_tail_check[col] = 'Both tails'
            elif max_value > value_of_75 + 1.5 * iqr:
                long_tail_check[col] = 'Right long-tailed'
            elif min_value < value_of_25 - 1.5*iqr:
                long_tail_check[col] = 'Left long-tailed'
            else:
                long_tail_check[col] = 'Not long-tailed'

        except (ValueError, TypeError):
            # Handle non-numeric or invalid values
            mean_max_min_equal[col] = ''
            large_std[col] = ''   
            max_too_high[col] = ''  
            long_tail_check[col] = ''
        
# Add the new row to the dataframe
df = pd.concat([df, pd.DataFrame([mean_max_min_equal])], ignore_index=True)
df = pd.concat([df, pd.DataFrame([mean_median_gap])], ignore_index=True)
df = pd.concat([df, pd.DataFrame([large_std])], ignore_index=True)
df = pd.concat([df, pd.DataFrame([max_too_high])], ignore_index=True)
df = pd.concat([df, pd.DataFrame([long_tail_check])], ignore_index=True)

print('analysis file created')
df.to_csv(analysis_file, index=False)
