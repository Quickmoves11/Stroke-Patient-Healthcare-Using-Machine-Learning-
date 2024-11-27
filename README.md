# Infosys_Stroke-Patient-Healthcare-Using-Deep-Learning

This project leverages a dataset of stroke patient data to build various machine learning models to predict stroke risk based on medical history, lifestyle factors, and health metrics. The dataset includes information such as gender, age, hypertension status, heart disease status, marital status, work type, residence type, glucose levels, BMI, and smoking status.

## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Problem Statement](#problem-statement)
- [Libraries Used](#libraries-used)
- [Dataset Loading](#dataset-loading)
- [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
- [Data Visualization](#data-visualization)
- [Data Encoding](#data-encoding)
- [Model Building](#model-building)
  - [Linear Regression](#linear-regression)
  - [Lasso Regression](#lasso-regression)
  - [Ridge Regression](#ridge-regression)
  - [Logistic Regression](#logistic-regression)
- [Model Evaluation](#model-evaluation)
- [Results and Visualizations](#results-and-visualizations)
- [Conclusion](#conclusion)

## Introduction

Stroke prediction is a critical task in healthcare, and this project aims to predict the likelihood of a patient suffering a stroke based on factors such as age, hypertension, heart disease, BMI, and smoking habits. Using machine learning, we can analyze these relationships and predict stroke occurrences, which could aid healthcare professionals in early diagnosis and prevention.

## Dataset Description

The dataset used in this project contains patient records with the following features:

- **id**: Unique identifier for each patient
- **gender**: Gender of the patient (Male/Female/Other)
- **age**: Age of the patient
- **hypertension**: Hypertension status (0 = No, 1 = Yes)
- **heart_disease**: Heart disease status (0 = No, 1 = Yes)
- **ever_married**: Marital status (Yes/No)
- **work_type**: Type of work (e.g., Private, Self-employed, etc.)
- **residence_type**: Type of residence (Urban/Rural)
- **avg_glucose_level**: Average glucose level in the patient's blood
- **bmi**: Body Mass Index of the patient
- **smoking_status**: Smoking status (e.g., formerly smoked, never smoked)
- **stroke**: Stroke occurrence status (1 = Yes, 0 = No)

## Problem Statement

The goal of this project is to predict whether a patient will experience a stroke or not, based on the available health metrics and lifestyle factors. This is a binary classification problem, where the target variable is "stroke."

## Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib** and **seaborn**: Data visualization
- **sklearn**: Machine learning algorithms and metrics
- **warnings**: Suppress warnings

## Dataset Loading

The dataset is loaded from a CSV file, containing medical history, lifestyle choices, and health metrics of patients. We clean the dataset by handling missing values and preparing it for analysis and modeling.

## Data Exploration and Preprocessing

### Shape of the Dataset
We start by exploring the structure of the dataset, checking its size and number of features.

### Data Types and Null Values
We analyze the data types and check for missing values.

### Handle Missing Values
For the `bmi` column, we impute the missing values using the median of the column.

## Data Visualization

We visualize several aspects of the data to gain insights into the relationships between different features and the target variable (stroke).

- **Age Distribution with Stroke Cases**  
- **Average Glucose Level vs Stroke Status**
- **BMI vs Stroke Status**
- **Stroke Count by Residence Type and Work Type**

## Data Encoding

We convert categorical variables into binary format for model training. This includes:
- **Residence Type** (Urban/Rural)
- **Work Type** (Private, Self-employed, Never worked)
- **Smoking Status** (Formerly smoked, Never smoked, Smokes)

We drop the original categorical columns and save the processed data to a new CSV file.

## Model Building

We train multiple machine learning models to predict stroke occurrence:

1. **Linear Regression**
2. **Lasso Regression**
3. **Ridge Regression**
4. **Logistic Regression**

### Linear Regression

A linear model is used to predict the stroke risk based on continuous features such as age, glucose levels, and BMI.

### Lasso Regression
Lasso Regression is a variant of linear regression that includes regularization to penalize large coefficients. This helps in improving the model's generalization by reducing overfitting.

### Ridge Regression
Ridge Regression is another variant of linear regression with regularization, similar to Lasso but using L2 regularization. This helps in controlling multicollinearity and overfitting.

### Logistic Regression

Logistic Regression is used for binary classification, where the goal is to predict the likelihood of a stroke based on the features.

## Model Evaluation

The models are evaluated using **Root Mean Squared Error (RMSE)** for regression models and **accuracy** for classification models.

### Comparison of Models

We compare the RMSE and accuracy of all models visually to determine the best performing model.

## Results and Visualizations

### RMSE Comparison

We visualize the RMSE values of all models to compare their performance.

### Accuracy Comparison

We visualize the accuracy of the models and determine the best one based on classification metrics.

## Conclusion

In this project, we used various machine learning algorithms to predict stroke occurrences based on patient data. Logistic Regression outperformed other regression models in terms of accuracy, providing an effective solution for stroke prediction.

The project demonstrates the utility of machine learning in healthcare to predict health conditions based on historical data, helping healthcare professionals make informed decisions.
