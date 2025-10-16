# Bank Customer Churn Prediction

This project aims to predict whether a bank customer will churn (i.e., leave the bank) using machine learning. The goal is to analyze customer data to identify the key drivers of churn and build a predictive model that can help the bank take proactive steps to retain customers.

## Project Workflow

The project is structured into a series of Jupyter notebooks that cover the end-to-end machine learning pipeline:

1.  **`1_EDA.ipynb`**: Exploratory Data Analysis to understand the dataset, visualize distributions, and identify initial patterns.
2.  **`2_Preprocessing.ipynb`**: Data cleaning, feature engineering, encoding categorical variables, scaling numerical features, and handling class imbalance with SMOTE.
3.  **`3_Modeling.ipynb`**: Training a baseline Logistic Regression model and several advanced models (Random Forest, Gradient Boosting, XGBoost), followed by hyperparameter tuning.
4.  **`4_Model_Evaluation.ipynb`**: Evaluating the models using various metrics (Accuracy, Precision, Recall, F1-Score, AUC), plotting confusion matrices, and ROC curves.
5.  **`5_Model_Explainability.ipynb`**: Using SHAP to interpret the best model's predictions and extract actionable business insights.

## Results

The final model, a **Tuned Random Forest Classifier**, achieved the following performance on the test set:
- **Accuracy**: ~86%
- **Precision (for churn)**: ~74%
- **Recall (for churn)**: ~62%
- **F1-Score (for churn)**: ~67%
- **AUC**: ~0.85

The model explainability analysis using SHAP revealed the following key drivers of churn:
- **Age**: Older customers are more likely to churn.
- **NumOfProducts**: Customers with more than 2 products have a very high churn rate.
- **Balance**: Customers with a higher balance are more likely to churn.
- **IsActiveMember**: Non-active members are at a higher risk of churning.
- **Geography**: Customers from Germany show a higher churn rate.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd ChurnSense-Bank
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Jupyter notebooks:**
    Launch Jupyter Notebook and run the notebooks in sequential order (from `1_EDA.ipynb` to `5_Model_Explainability.ipynb`) to reproduce the analysis and results.
    ```bash
    jupyter notebook
    ```

## Saved Models

The trained models and the preprocessor are saved in the root directory:
- `best_random_forest_model.joblib`: The final, tuned model.
- `preprocessor.joblib`: The scikit-learn pipeline for data preprocessing.

## Future Work

-   Automate the model retraining pipeline using MLflow or Kubeflow.
-   Deploy the final model as a REST API using Flask or FastAPI.
-   Build an interactive dashboard with Streamlit to visualize predictions and insights.
-   Experiment with deep learning models (e.g., ANNs) to potentially improve prediction accuracy.
