# ML Model Comparison: Task 1 & Task 2

This repository contains a project that builds and evaluates different machine learning models on a healthcare dataset for stroke prediction. The tasks are divided into two parts:

## Task 1: Building and Training Models
In Task 1, we build and train four different models:
- Linear Regression
- Lasso Regression
- Ridge Regression
- Logistic Regression (for classification)

The models are trained on the **healthcare stroke dataset** and evaluated based on RMSE (Root Mean Squared Error) and Accuracy.

## Task 2: Preprocessing, Evaluation, and Visualizations
In Task 2, the dataset undergoes preprocessing, including handling missing values, encoding categorical features, and feature selection. After preprocessing, the models are evaluated and compared based on RMSE and Accuracy. Additionally, visualizations are created to compare the performance of the models.

## Steps:
1. **Preprocessing**: Handle missing values, encode categorical variables.
2. **Model Building**: Train multiple models (Linear, Lasso, Ridge, Logistic Regression).
3. **Evaluation**: Compare models based on RMSE and accuracy.
4. **Visualizations**: Plot RMSE and Accuracy comparisons and visualize model predictions.

## Technologies Used:
- Python
- `scikit-learn` for machine learning models
- `pandas` for data manipulation
- `numpy` for numerical operations
- `matplotlib` and `seaborn` for data visualization

### RMSE Comparison Plot
Here is the RMSE comparison of all models:

![Figure_1](https://github.com/user-attachments/assets/3d103381-f0d5-4ca1-ab1d-a447fcd00c09)


### Accuracy Comparison Plot
Here is the accuracy comparison of all models:

![Figure_2](https://github.com/user-attachments/assets/92e87bba-a48a-4d9c-8744-021eb283de43)


### Predictions Comparison
Below is the comparison of the actual vs predicted values for Logistic and Linear Regression:

![Figure_3](https://github.com/user-attachments/assets/f7d80c9f-1de6-4663-9fa0-d27ef68d317b)


...
