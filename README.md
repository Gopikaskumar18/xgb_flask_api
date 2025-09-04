# Large-Scale Product Demand Forecasting with XGBoost

This project is an **end-to-end product demand forecasting system** built using **XGBoost**, with **data preprocessing, feature engineering, model training, hyperparameter tuning**, and **deployment via a Flask API**. It predicts product demand for retail stores based on historical sales and other features.

---

## **Table of Contents**

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Features](#features)  
4. [Modeling](#modeling)  
5. [Hyperparameter Tuning](#hyperparameter-tuning)  
6. [Deployment](#deployment)  
7. [Project Structure](#project-structure)  
8. [Installation & Setup](#installation--setup)  
9. [API Usage](#api-usage)  
10. [Evaluation](#evaluation)  
11. [Future Work](#future-work)  
12. [Author](#author)  

---

## **Project Overview**

The goal of this project is to **predict the demand for products** across multiple stores and time periods. Accurate demand forecasting helps in inventory management, reducing overstock and stockouts, and optimizing supply chain operations.

- **Problem type:** Regression  
- **Algorithm:** XGBoost Regressor  
- **Evaluation metric:** RMSE (Root Mean Squared Error)  

---

## **Dataset**

- Based on the **M5 Forecasting dataset** (sample or full dataset can be used).  
- Contains historical sales data, calendar information, and price data.  
- Data includes columns like:
  - `item_id`, `store_id`, `dept_id`, `sell_price`, `date`, etc.

---

## **Features**

The model uses a mix of **categorical, numerical, and temporal features**, including:

- **Time features:**  
  - `d` (day index), `weekday`, `month`, `year`
- **Price features:**  
  - `sell_price`, lag features, rolling means  
- **Event features:**  
  - `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2`  
- **Store & item identifiers:**  
  - `item_id`, `store_id`, `dept_id`, `cat_id`, `state_id`  
- **Snap flags:** Indicate if a state/store has special sales events  

---

## **Modeling**

- **Model:** XGBoost Regressor (`reg:squarederror`)  
- **Data preprocessing:**  
  - Missing value imputation (median)  
  - Label encoding for categorical features  
- **Training:** Subset of data for faster experimentation  
- **Hyperparameter tuning:** RandomizedSearchCV  

**Best parameters found:**

| Parameter            | Value  |
|---------------------|-------|
| `subsample`          | 0.8   |
| `n_estimators`       | 200   |
| `min_child_weight`   | 3     |
| `max_depth`          | 4     |
| `learning_rate`      | 0.05  |
| `gamma`              | 0     |
| `colsample_bytree`   | 0.7   |

---

## **Hyperparameter Tuning**

- **Tool:** `RandomizedSearchCV` with 3-fold cross-validation  
- **Scoring metric:** RMSE  
- **Process:**  
  1. Define parameter grid  
  2. Encode categorical features using LabelEncoder  
  3. Fit RandomizedSearchCV to find optimal hyperparameters  
- **Result:** Best RMSE achieved: `~2.53` on validation subset  

---

## **Deployment**

- **Framework:** Flask  
- **Endpoint:** `/predict`  
- **Method:** POST  
- **Input:** JSON object with feature values  
- **Output:** Predicted demand in JSON  

**Example request:**

```json
{
  "item_id": "HOBBIES_1_001",
  "store_id": "CA_1",
  "dept_id": "HOBBIES_1",
  "sell_price": 15.99,
  "d": 1913,
  "weekday": 5,
  "event_name_1": "Sports",
  "event_type_1": "Sporting",
  "snap_CA": 1
}
