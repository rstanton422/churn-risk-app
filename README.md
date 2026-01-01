# churn-risk-app
A machine learning app to predict customer churn and calculate potential revenue loss

## Model Selection: Why Random Forest?
The Random Forest Classifier was selected over other algorithms (such as Logistic Regression) for three specific reasons tied to the dataset's characteristics:

1.  **Handling Non-Linearity:** EDA revealed non-linear patterns, specifically in the "Age" feature, where risk peaked in the 40-50 age bracket but dropped for older customers. Random Forest captures these non-linear boundaries effectively without manual feature transformation.
2.  **Imbalanced Interaction Effects:** The interaction between "Geography" and "Balance" (specifically German customers with high balances) was a strong predictor. Tree-based ensembles capture these conditional dependencies naturally.
3.  **Interpretability:** In a financial context, "black box" predictions are risky. Random Forest provides `feature_importance_` metrics, allowing us to explain *why* a customer is flagged (e.g., verifying that "Product Depth" is a critical retention factor).

# Customer Retention & Financial Risk Dashboard

## Project Overview
A full-stack data science application designed to predict bank customer churn and quantify financial risk. By analyzing demographic and behavioral data, this tool allows relationship managers to identify high-risk high-value customers before they leave.

## Tech Stack
* **Python 3.10**
* **Machine Learning:** Scikit-Learn (Random Forest Classifier - 86% Accuracy)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Streamlit, Seaborn

## Key Features
* **Real-time Prediction:** Instant churn probability assessment based on live inputs.
* **Financial Impact Analysis:** meaningful "Revenue at Risk" calculation ($) triggers for high-risk profiles.
* **Interactive Dashboard:** User-friendly interface for non-technical stakeholders.

## How to Run
1.  Clone the repository: `git clone [YOUR_REPO_LINK_HERE]`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the app: `streamlit run app.py`