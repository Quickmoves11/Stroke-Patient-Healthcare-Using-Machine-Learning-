# ML Model Comparison: Task 1 & Task 2

This repository contains a project that builds and evaluates different machine learning models on a healthcare dataset for stroke prediction. The tasks are divided into two parts:

## Task 1: Building and Training Models
In Task 1, we build and train four different models:
- Linear Regression
- Lasso Regression
- Ridge Regression
- Logistic Regression (for classification)

The models are trained on the **healthcare stroke dataset** and evaluated based on RMSE (Root Mean Squared Error) and Accuracy.

## Task 2: Preprocessing, Evaluation, Visualizations, and Observations
In Task 2, the dataset undergoes preprocessing, including handling missing values, encoding categorical features, and feature selection. After preprocessing, the models are evaluated and compared based on RMSE and Accuracy. Additionally, visualizations are created to compare the performance of the models.

## Steps:
1. **Preprocessing**: Handle missing values, encode categorical variables.
2. **Model Building**: Train multiple models (Linear, Lasso, Ridge, Logistic Regression).
3. **Evaluation**: Compare models based on RMSE and accuracy.
4. **Visualizations**: Plot RMSE and Accuracy comparisons and visualize model predictions.

### Observations: Why Certain Models Have Good Accuracy and Others Don't

The performance of each model varies due to the following reasons:

1. **Model Type and Purpose:**
   - **Linear Regression** is designed for continuous outcomes and may not work well for binary classification tasks like stroke prediction. 
   - **Logistic Regression**, however, is specifically designed for classification tasks and works well for binary outcomes, making it the most suitable model for this problem.
   - **Ridge and Lasso Regression** use regularization techniques that prevent overfitting, making them potentially more robust than Linear Regression when dealing with noisy or high-dimensional data.

2. **Overfitting and Underfitting:**
   - **Regularization** in **Ridge** and **Lasso** helps prevent overfitting, improving generalization and performance on unseen data. This often results in better accuracy than **Linear Regression**, which is more prone to overfitting when the number of features is large or complex.

3. **Data Characteristics:**
   - Models like **Logistic Regression** tend to perform well when the data has a linear relationship with the target variable, as in this stroke prediction case. If the data has a more complex, non-linear structure, more advanced models would be required to achieve better performance.

4. **Feature Engineering:**
   - The preprocessing steps such as handling missing values, encoding categorical variables, and feature selection can significantly impact model performance. For example, **missing BMI values** were imputed with the median, and categorical variables were encoded into numeric formats.

5. **Class Imbalance:**
   - If the dataset has an imbalanced class distribution (e.g., more people did not have a stroke), models may show bias toward predicting the majority class. This can lead to higher accuracy but poor predictive performance for the minority class (stroke). In such cases, metrics like **precision**, **recall**, and **F1-score** should be considered.


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
