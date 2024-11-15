
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Task 2 - **Preprocessing: Handle missing values and encode categorical variables**
# Handling missing values for the 'bmi' column by assigning the result back
df['bmi'] = df['bmi'].fillna(df['bmi'].median())  # Replace missing BMI values with the median BMI

# Convert categorical variables into numerical format (encoding)
df['Residence_type'] = df['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)  # Urban = 1, Rural = 0

# One-hot encoding for work_type and smoking_status (binary encoding)
df['work_type_Never_worked'] = df['work_type'].apply(lambda x: 1 if x == 'Never_worked' else 0)
df['work_type_Private'] = df['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
df['work_type_Self_employed'] = df['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)

df['smoking_status_formerly_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'formerly smoked' else 0)
df['smoking_status_never_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'never smoked' else 0)
df['smoking_status_smokes'] = df['smoking_status'].apply(lambda x: 1 if x == 'smokes' else 0)

# Drop unnecessary columns from the dataset that are not needed for model training
df_model = df.drop(['Residence_type', 'work_type', 'smoking_status', 'gender', 'ever_married'], axis=1)

# Define features (X) and target (y)
X = df_model.drop('stroke', axis=1)  # independent variables
y = df_model['stroke']  # dependent variable 

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 1 - **Building and training the models**

# Linear Regression Model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)  # Train the model
linear_reg_predictions = linear_reg.predict(X_test)  # Predict on test data
linear_reg_rmse = np.sqrt(mean_squared_error(y_test, linear_reg_predictions))  # Calculate RMSE
linear_reg_rmse = linear_reg_rmse * 100  # RMSE in percentage

# Lasso Regression Model 
lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)
lasso_reg_predictions = lasso_reg.predict(X_test)
lasso_reg_rmse = np.sqrt(mean_squared_error(y_test, lasso_reg_predictions))
lasso_reg_rmse = lasso_reg_rmse * 100

# Ridge Regression Model
ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)
ridge_reg_predictions = ridge_reg.predict(X_test)
ridge_reg_rmse = np.sqrt(mean_squared_error(y_test, ridge_reg_predictions))
ridge_reg_rmse = ridge_reg_rmse * 100

# Logistic Regression Model 
clf=LogisticRegression()
clf.fit(X_train, y_train)
Logistic_pred = clf.predict(X_test)
Log_reg_rmse = np.sqrt(mean_squared_error(y_test, Logistic_pred))
Log_reg_rmse=Log_reg_rmse*100



# Task 2 - **Comparing the models based on RMSE and Accuracy**

# Display RMSE for regression models and Accuracy for Logistic Regression
print(f"Linear Regression RMSE: {linear_reg_rmse:.2f}")  # RMSE for Linear Regression
print(f"Lasso Regression RMSE: {lasso_reg_rmse:.2f}")  # RMSE for Lasso Regression
print(f"Ridge Regression RMSE: {ridge_reg_rmse:.2f}")  # RMSE for Ridge Regression
print(f"Logistic Regression RMSE: {Log_reg_rmse:.2f}") # RMSE for Logistic Regression


#Accuracy of each Model 
linear_reg_accuracy =linear_reg.score(X_test, y_test)*100
lasso_reg_accuracy =lasso_reg.score(X_test, y_test)*100
ridge_reg_accuracy =ridge_reg.score(X_test, y_test)*100
log_reg_accuracy = clf.score(X_test, y_test)*100

# Calculate accuracy score

# Logistic Regression accuracy is in percentage, as it's a classification task
print(f"Linear Regression Accuracy (R² score): {linear_reg_accuracy:.2f}%")
print(f"Lasso Regression Accuracy (R² score): {lasso_reg_accuracy:.2f}%")
print(f"Ridge Regression Accuracy (R² score): {ridge_reg_accuracy:.2f}%")
print(f"Logistic Regression Accuracy: {log_reg_accuracy:.2f}%")# Accuracy for Logistic Regression

# Task 2 - **Visualizations**

# Plotting the RMSE for Linear, Lasso, and Ridge Regression Models, and Accuracy for Logistic Regression
plt.figure(figsize=(12, 6))

# Bar plot for RMSE of regression models and accuracy for Logistic Regression
plt.bar(['Linear Regression', 'Lasso Regression', 'Ridge Regression'], 
        [linear_reg_rmse, lasso_reg_rmse, ridge_reg_rmse], color=['blue', 'green', 'red'], alpha=0.7, label='RMSE')
plt.bar(['Logistic Regression'], [log_reg_accuracy], color='purple', alpha=0.7, label='Accuracy')

plt.title('Model Comparison: RMSE vs Accuracy')
plt.ylabel('Score (RMSE/Accuracy)')
plt.tight_layout()  # Adjust layout for better visualization
plt.legend(loc='upper left')
plt.show()

# Task 2 - **Visualizing Predictions of Logistic Regression and Linear Regression**

# Visualize the predicted vs actual values for Logistic Regression (Classification task)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, Logistic_pred, color='purple')
plt.plot([0, 1], [0, 1], 'r--', lw=2)  # Line representing perfect prediction (ideal case)
plt.title('Logistic Regression: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Visualize the predicted vs actual values for Linear Regression (Regression task)
plt.subplot(1, 2, 2)
plt.scatter(y_test, linear_reg_predictions, color='blue')
plt.plot([0, 1], [0, 1], 'r--', lw=2)  # Line representing perfect prediction (ideal case)
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()

