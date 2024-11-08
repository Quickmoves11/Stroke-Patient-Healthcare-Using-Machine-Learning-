import pandas as pd

# Loading the Data
data = pd.read_csv('healthcare-dataset-stroke-data.csv')


# 1: Convert 'Residence_type' column to binary (0 = Rural, 1 = Urban)
data['Urban/Rural'] = data['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)

# 2: Convert 'work_type' column to binary columns
data['work_type_Never_worked'] = data['work_type'].apply(lambda x: 1 if x == 'Never_worked' else 0)
data['work_type_Private'] = data['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
data['work_type_Self_employed'] = data['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)


# 3: Convert 'smoking_status' column to binary columns
data['smoking_status_formerly_smoked'] = data['smoking_status'].apply(lambda x: 1 if x == 'formerly smoked' else 0)
data['smoking_status_never_smoked'] = data['smoking_status'].apply(lambda x: 1 if x == 'never smoked' else 0)
data['smoking_status_smokes'] = data['smoking_status'].apply(lambda x: 1 if x == 'smokes' else 0)

# 4: Create a new variable to store the dataset for data model
# Drop original categorical columns that were encoded
model_data = data.drop(['Residence_type', 'work_type', 'smoking_status'], axis=1)

# Save the encoded dataset to a new CSV file for model building
model_data.to_csv('model_data.csv', index=False)

# Print the final transformed dataset to verify
print(model_data.head())

