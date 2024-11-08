import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset....
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Frame your questions:
# 1. What is the distribution of age in stroke cases?
# 2. How does average glucose level vary with stroke cases?
# 3. How does BMI vary between stroke and non-stroke cases?
# 4. What is the stroke count by residence type?
# 5. What is the stroke count by work type?

# 1. Age Distribution with Stroke Cases
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='age', hue='stroke', multiple='stack', bins=30)
plt.title('Age Distribution with Stroke Cases')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 2. Average Glucose Level Distribution by Stroke Status
plt.figure(figsize=(10, 6))
sns.kdeplot(data=data, x='avg_glucose_level', hue='stroke', fill=True)
plt.title('Average Glucose Level Distribution by Stroke Status')
plt.xlabel('Average Glucose Level')
plt.ylabel('Density')
plt.show()

# 3. BMI Distribution with Stroke Cases
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='bmi', hue='stroke', multiple='stack', bins=30)
plt.title('BMI Distribution with Stroke Cases')
plt.xlabel('BMI')
plt.ylabel('Count')
plt.show()

# 4. Stroke Count by Residence Type
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='Residence_type', hue='stroke')
plt.title('Stroke Count by Residence Type')
plt.xlabel('Residence Type')
plt.ylabel('Count')
plt.show()

# 5. Stroke Count by Work Type
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='work_type', hue='stroke')
plt.title('Stroke Count by Work Type')
plt.xlabel('Work Type')
plt.ylabel('Count')
plt.show()

# Observations:
# - Stroke cases are more common among older age groups.
# - Higher average glucose levels are more associated with stroke cases.
# - There is a variation in BMI between stroke and non-stroke cases.
# - There is a difference in stroke counts based on residence type.
# - Certain work types (like 'Private') have a higher count of stroke cases.
