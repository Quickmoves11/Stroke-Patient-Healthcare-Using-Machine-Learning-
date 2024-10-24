# Infosys_Stroke-Patient-Healthcare-Using-Deep-Learning

**<ins>Dataset Description:</ins>**       
This dataset contains critical information such as medical history, lifestyle choices, and health metrics. It includes features like gender, age, hypertension status, heart disease status, marital status, work type, residence type, average glucose level, body mass index (BMI), and smoking status. The dataset serves as a valuable resource for understanding the relationships between these variables and stroke incidence, facilitating research and predictive modeling in healthcare.

**<ins>Dataset:</ins>**  
The dataset provided consists of patient records aimed at analyzing stroke occurrences and related factors:
* **id:** Unique identifier for each patient
* **gender:** Gender of the patient (Male/Female)
* **age:** Age of the patient
* **hypertension:** Hypertension status (0: No, 1: Yes)
* **heart_disease:** Heart disease status (0: No, 1: Yes)
* **ever_married:** Marital status (Yes/No)
* **work_type:** Type of work (e.g., Private, Self-employed)
* **Residence_type:** Type of residence (Urban/Rural)
* **avg_glucose_level:** Average glucose level in the patient's blood
* **bmi:** Body Mass Index of the patient
* **smoking_status:** Smoking status (e.g., formerly smoked, never smoked)
* **stroke:** Stroke occurrence status (1: Yes, 0: No)

# **(1) Defining Problem Statement and Analyzing Basic Information**
***

In this section, we will define the problem statement and analyze the basic metrics of the healthcare stroke dataset to understand its structure and characteristics:

1. **Shape of the Dataset:** 
   - Provides the number of rows and columns, giving insight into the dataset's size.

2. **Data Information:**
   - Using the `info()` function to display data types and the count of non-null entries, which helps identify potential issues with missing values and data types.

3. **Describe:**
   - Utilizing the `describe()` function to summarize numerical features of the dataset, providing insights into central tendency, dispersion, and shape of the dataset’s distribution.


# **(2) Analyzing Unique Values and Null Values**
***

In this section, we will analyze the unique values, null values, and their percentages within the healthcare stroke dataset:

1. **Finding Unique Values:**
   - Identifying unique values in each column to understand the variability and quality of categorical data, which can help spot potential data entry issues.

2. **Finding Null Values:**
   - Counting the number of null values in each column to assess the completeness of the dataset. This step is crucial for understanding which features may require imputation or further cleaning.

3. **Calculating Percentage of Null Values:**
   - Computing the percentage of null values for each column relative to the total number of entries. This metric helps prioritize which columns may need attention based on the extent of missing data.

This analysis is essential for ensuring data quality and informing subsequent preprocessing steps.


# **(3) Handling Null Values**
***

In this section, we will outline the approach for dealing with null values in the healthcare stroke dataset:

1. **Identifying Null Values:**
   - First, we will identify columns with null values and assess their frequency to understand the extent of missing data.

2. **Imputation Strategy:**
   - For numerical columns, we will use the **median** to fill null values. The median is robust to outliers and provides a better measure of central tendency when the data is skewed.

3. **Checking for Remaining Null Values:**
   - After imputation, we will check for any remaining null values to ensure that all missing data has been addressed.

This approach aims to maintain data integrity and usability for subsequent analysis and modeling.


