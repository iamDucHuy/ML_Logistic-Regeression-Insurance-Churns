# 🛡️Insurance Churn Prediction Project (Logistic Regression)

## 🌟 Overview
This project focuses on building a machine learning model to predict the likelihood of an insurance customer *churning* (canceling their policy). The primary goal is to help the insurance company proactively identify high-risk customers, allowing them to implement suitable retention strategies.

The model is built using the **Logistic Regression** algorithm and has demonstrated **exceptionally strong performance**, providing high-confidence classification of potential churners.

## 📊 Data
The input data is sourced from the file `randomdata.csv`. Key analyzed data fields include:
* `Customer Name`, `Customer_Address`, `Company Name`
* `Claim Reason`
* `Data confidentiality`
* `Claim Amount`
* `Category Premium`
* `Premium/Amount Ratio`
* `BMI` (Body Mass Index)
* `Churn` (Target Variable: **Yes** or **No**)

## 🛠️ Methodology and Technology
The project is implemented in Python within a Jupyter Notebook environment, utilizing the following core libraries:
* **Modeling:** `scikit-learn` (Logistic Regression, Cross-Validation)
* **Data Analysis:** `pandas`, `numpy`
* **Visualization:** `matplotlib.pyplot`, `seaborn`, `plotly.express`
* **Others:** `pycountry-convert` (for geographical/country data processing)

## ✅ Model Evaluation Results
The model was rigorously evaluated using Cross-Validation and achieved impressive metrics:

| Evaluation Metric | Average Value | Description |
| :--- | :--- | :--- |
| **Accuracy** | **0.978** | The model correctly predicts 97.8% of cases. |
| **Macro F1 Score** | **0.977** | A high score indicating good class balance, meaning the model performs well in classifying both majority and minority classes (Churn/Non-Churn). |
| **AUC** (Area Under the Curve) | **0.9984** | An **exceptionally high** score, demonstrating the model's ability to distinguish between churners and non-churners is **near-perfect (99.84%)** across all decision thresholds. |

## 🔑 Key Insight
The notebook includes a specific analysis of the **BMI (Body Mass Index)** effect on the predicted probability of customer churn:
* The plot titled **Effect of BMI on Predicted Probability of Churn** illustrates a sigmoid-like relationship between BMI and Churn probability. This strongly suggests that BMI is one of the most critical factors influencing a customer's decision to leave.

## 🚀 How to Run the Project
### Installation Requirements
To run this notebook, you need to install the following Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn pycountry-convert
