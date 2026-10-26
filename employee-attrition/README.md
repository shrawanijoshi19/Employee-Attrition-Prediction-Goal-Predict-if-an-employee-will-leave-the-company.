# Employee Attrition Prediction

**Author:** Shrawani Joshi
**Subject:** Machine Learning Minor Project
**Year:** 2025-26

---

## Problem Statement
Predict whether an employee will leave the company (Attrition: Yes/No) using HR data including personal, job, compensation, and satisfaction-related features.

## Objective
- Analyze key factors driving employee attrition
- Build and compare multiple ML classification models
- Deploy an interactive HR dashboard using Streamlit

## Dataset Features (27 Features)
| Category       | Features |
|----------------|----------|
| Personal       | Age, Gender, MaritalStatus, DistanceFromHome, Education |
| Job            | Department, JobRole, JobLevel, OverTime, BusinessTravel |
| Compensation   | MonthlyIncome, PercentSalaryHike, StockOptionLevel |
| Experience     | TotalWorkingYears, YearsAtCompany, NumCompaniesWorked |
| Satisfaction   | JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance |

**Target:** Attrition (Yes = 1, No = 0)

## Models Used
| Model                 | Type           |
|-----------------------|----------------|
| Logistic Regression   | Linear         |
| Decision Tree         | Tree-based     |
| Random Forest         | Ensemble       |
| Gradient Boosting     | Ensemble       |
| AdaBoost              | Ensemble       |
| K-Nearest Neighbors   | Instance-based |
| SVM                   | Kernel-based   |

## Project Structure
```
employee-attrition/
|-- generate_dataset.py     # Create synthetic HR dataset
|-- train_model.py          # Train all models + save plots
|-- app.py                  # Streamlit web application
|-- requirements.txt        # Python dependencies
|-- data/
|   |-- employee_attrition.csv
|-- models/
|   |-- attrition_model.pkl
|   |-- scaler.pkl
|   |-- encoders.pkl
|   |-- feature_importance.pkl
|-- plots/
    |-- 01_attrition_distribution.png
    |-- 02_attrition_by_dept.png
    |-- ...
```

## How to Run
```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Generate dataset
python generate_dataset.py

# Step 3: Train models
python train_model.py

# Step 4: Launch app
streamlit run app.py
```
Open http://localhost:8501 in your browser.