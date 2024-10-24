import pandas as pd

# Load the Stroke Dataset file (healthcare-dataset-stroke-data.csv)....
file_path = 'healthcare-dataset-stroke-data.csv'
data = pd.read_csv(file_path)


#  Shape of the dataset
print("Shape of the dataset:", data.shape)



# Data types and info
print("\nData Info:")
print(data.info())


# Describe
print("\nDescribe:")
print(data.describe(include='object'))




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


#how are you going to deal with null values?
#In This We will add Median Value of that Column in the Null Cell....
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    median_value = data[column].median()
    data[column].fillna(median_value, inplace=True)

print("\nResults After Dealing with Null Value")
print(data.isnull().sum())
