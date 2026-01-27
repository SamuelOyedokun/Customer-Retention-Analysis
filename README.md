# Customer Retention Analysis (Telco Churn)


## Overview
This project analyzes **customer churn** for a telecom company to identify factors driving churn and quantify **revenue at risk**. Using the **Telco Customer Churn dataset**, we performed data cleaning, exploratory data analysis (EDA), and built a **simple, explainable logistic regression model**.

---

## Business Problem
Customer churn is costly. Retaining existing customers is far cheaper than acquiring new ones.  

**Objectives:**
- Identify which customers are most likely to churn.
- Determine the features driving churn.
- Quantify monthly revenue at risk.
- Provide actionable recommendations for retention.

---

## Data
The dataset contains **7043 customer records** with 21 features including:

- `tenure` – how long the customer has stayed  
- `MonthlyCharges` – monthly subscription fees  
- `Contract` – type of contract (month-to-month, yearly)  
- `PaymentMethod` – payment method  
- `InternetService` – type of internet service  
- `Churn` – target variable (1 = churned, 0 = stayed)  

Data source: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## Data Cleaning
- Converted `TotalCharges` to numeric and handled missing values.  
- Converted categorical columns to `category` type.  
- Converted target variable `Churn` to numeric (1/0).

---

## Exploratory Data Analysis (EDA)
- **Churn Distribution:** ~26% of customers churned.  
- **Churn by Contract Type:** Month-to-month contracts have the highest churn.  
- **Monthly Revenue at Risk:** `$7,874.00` (sum of monthly charges for churned customers).  

### Example Plots
#### Churn Distribution
![Churn Distribution](plots/Churn_distribution.png)

#### Churn by Contract Type
![Churn by Contract](plots/Churn_by_contract.png)


---

## Model
- **Algorithm:** Logistic Regression (explainable)  
- **Features:** All original features, one-hot encoded categorical variables  
- **Train/Test Split:** 80/20  

**Performance:**
- Accuracy: 0.79  
- ROC-AUC: 0.83  
- F1-score (churn): 0.56  

### Top 10 Features Driving Churn
| Feature | Coefficient | Interpretation |
|---------|------------|----------------|
| InternetService_Fiber optic | 1.08 | Fiber customers more likely to churn |
| StreamingMovies_Yes | 0.36 | Streaming movies slightly increases churn |
| StreamingTV_Yes | 0.34 | Streaming TV slightly increases churn |
| MultipleLines_No phone service | 0.34 | No phone service customers have higher churn |
| MultipleLines_Yes | 0.28 | Multiple lines slightly increases churn |
| PaperlessBilling_Yes | 0.28 | Paperless billing slightly increases churn |
| SeniorCitizen | 0.26 | Older customers more likely to churn |
| PaymentMethod_Electronic check | 0.26 | Electronic check payers more likely to churn |
| Partner_Yes | 0.05 | Partner reduces churn slightly |
| DeviceProtection_Yes | 0.04 | Device protection slightly reduces churn |

---

## Business Insights
- Fiber internet users and month-to-month contract customers are at **highest churn risk**.  
- Streaming services (TV & movies) slightly increase churn.  
- Paperless billing and payment method influence churn.  
- Older and multi-line customers may need **targeted retention campaigns**.  

**Recommendation:** Focus retention efforts on high-risk customer segments to **reduce revenue loss**.

---

## Next Steps
- Test other models (Random Forest, XGBoost) for comparison.  
- Implement a **simple dashboard** to visualize churn risk for business users.  
- Explore **customer lifetime value (CLV)** to prioritize retention efforts.  

---

## Contact
Connect with me on [LinkedIn](https://www.linkedin.com/in/samuel_oyedokun/)  

---

