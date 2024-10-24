import pandas as pd

# Load the dataset
file_path = 'healthcare-dataset-stroke-data.csv'
data = pd.read_csv(file_path)


#  Shape of the dataset
print("Shape of the dataset:", data.shape)



# Data types and info
print("\nData Info:")
print(data.info())


# Descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe(include='object'))


# Find unique values, null values, and percentage of null values
# Unique values in each column
unique_values = {column: data[column].unique() for column in data.columns}
print("\nUnique Values in Each Column:")
for column, values in unique_values.items():
    print(f"{column}: {values[:10]}")  # Show only first 10 unique values


# Null values
null_values = data.isnull().sum()
print("\nNull Values in Each Column:")
print(null_values)

#Percentage of null values
percentage_nulls = (null_values / len(data)) * 100
print("\nPercentage of Null Values in Each Column:")
print(percentage_nulls)


