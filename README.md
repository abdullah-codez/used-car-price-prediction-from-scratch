# **🚗 Used Car Price Prediction – Linear Regression from Scratch**

A deep-dive machine learning project where I **manually implemented multiple linear regression** to predict used car prices from structured data, with full preprocessing, feature engineering, and optimization — no Scikit-learn model training shortcuts used.

---

## **🧠 Project Goal**

To **understand linear regression inside-out** by applying it to a real-world dataset building the entire ML pipeline from scratch and manually implementing:

* Cost computation

* Gradient descent

* Weight updates

* Feature engineering

* Evaluation metrics

---

## **🔧 Dataset**

**Used Car Price Prediction (Cardekho version)**  
 Source: [Kaggle](https://www.kaggle.com/datasets)  
 Features include:

* `Unnamed: 0, car_name, brand, model, vehicle_age, km_driven, seller_type, fuel_type, transmission_type, mileage, engine, max_power, seats`  
* Target: `selling_price`

---

## **🔍 Problem Statement**

Predict the selling price of a used car based on its specs, brand/model, engine power, and other characteristics , using only linear regression (from scratch) while handling real-world data imperfections.

---

## **🛠 Pipeline Overview**

### **🔹 1\. Data Cleaning & Exploration**

* Dropped ID and  redundant column (`Unnamed: 0`, `car_name`)

* Handled missing values

* Identified heavily skewed numeric features

### **🔹 2\. Feature Transformation**

* Applied `log1p()` or `sqrt()` on skewed features

* Capped extreme outliers

### **🔹 3\. Categorical Encoding**

* Used **mean target encoding** for high-cardinality columns (`brand`, `model`) *after* train-test split to avoid leakage

* One-hot encoding for features like `fuel_type`, `seller_type`

* Label encoding for categorical data

### **🔹 4\. Feature Engineering**

Manually added meaningful interaction/polynomial features:

* `engine × max_power`

* `Power_squared, engine_squared, vehicle_age_squared`

This massively boosted performance (R² from 0.37 → 0.54 on train set).

### **🔹 5\. Feature Scaling**

* Standardized all numeric features **before** train-test split

* Applied same scaling on engineered features

### **🔹 6\. Manual Model Training**

* Implemented custom **gradient descent** (no Scikit-learn models)

* Printed cost every 100 epochs

* Tuned learning rate and epochs for convergence

### **🔹 7\. Evaluation**

* Transformed predictions back using `expm1()` to compare in real price scale

* Used:

  * **RMSE**

  * **R²**

  * **Cost vs. Iteration plot**

---

## **📊 Final Results**

| Set | RMSE (Rs.) | R² Score |
| ----- | ----- | ----- |
| **Train** | 606,416 | 0.547 |
| **Test** | 616,343 | 0.495 |

 ✅ Model generalizes well with minimal overfitting  
 ✅ Predicts \~50% of variance in selling price using linear model only  
 ✅ Feature engineering had a major impact on accuracy

---

## 

## **🔥 What I Learned**

* How to detect and fix **underfitting** in linear models

* Why **target encoding \+ careful scaling** matters

* The power of **feature engineering** in boosting linear models

* Deep understanding of **gradient descent**, loss landscape, and R² intuition

* How to build **a full ML pipeline** from raw CSV to model evaluation

---

## **📌 Future Improvements**

* Add Ridge/Lasso regularization

* Try cross-validation

* Compare with Random Forest or XGBoost baselines

* Deploy with Streamlit or Flask

---

## **🧑‍💻 Author**

**Muhammad Abdullah Iftikhar**  
 Computer Science @ PIEAS  
 ML enthusiast, learning by building.

---

## **⭐ If You Like This**

* Give this repo a ⭐ on GitHub

