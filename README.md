# 🧪 Predicting Tank Failure Pressure Using Machine Learning

## 📌 Problem Statement

The goal of this project is to predict the **pressure at which a tank will fail** using supervised machine learning techniques. A dataset is provided containing multiple features relevant to tank conditions and structure. The objective is to train a model that accurately forecasts the failure pressure based on these features.

The project workflow includes **data preprocessing**, **feature engineering**, **model development**, **hyperparameter tuning**, **evaluation**, and optionally **deployment**. Performance will be evaluated using metrics such as:

* **MAPE (Mean Absolute Percentage Error)**
* **R² Score**
* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**

The final solution will include model ensembling using **Support Vector Regression (SVR)** and **Random Forest Regressor**, optimized via **GridSearchCV**, and deployed for use on unseen instances.

---

## 📚 Table of Contents

### 🔧 Task One: Data Preprocessing <a id="1"></a>

* ✅ [Load the dataset](#2)
* ✅ [Handle missing values](#3)
* ✅ [Detect and remove outliers](#4)
* ✅ [Check for duplicate records](#5)
* ✅ [Select relevant features using correlation analysis](#6)
* ✅ [Perform feature engineering (e.g., interaction terms)](#7)
* ✅ [Convert data types where needed](#8)
* ✅ [Apply feature scaling (StandardScaler/MinMax)](#9)
* ✅ [Data augmentation (if applicable)](#10)
* ✅ [Encode categorical columns (One-Hot or Label Encoding)](#11)
* ✅ [Perform final data preprocessing pipeline](#12)
* ✅ [Split the dataset into training and testing sets](#13)

---

### 🤖 Task Two: Model Development <a id="14"></a>

* 📊 [Visualize data distributions and relationships](#15)
* ⚙️ [Train baseline models and evaluate](#16)
* 🛠️ [Perform hyperparameter tuning using GridSearchCV](#18)
* 🔮 [Make predictions on test data](#17)
* 📈 [Visualize model performance using graphs](#19)
* 🧩 [Model ensembling using SVR and Random Forest](#20)

