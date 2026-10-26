# coding: utf-8
"""
Generate synthetic Employee Attrition dataset (IBM HR Analytics style)
Author : Shrawani Joshi
Project: ML Minor Project - Employee Attrition Prediction
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1470  # IBM HR dataset standard size

def generate_dataset(n=N):
    departments    = np.random.choice(["Sales","Research & Development","Human Resources"], n, p=[0.35,0.55,0.10])
    job_roles      = []
    for d in departments:
        if d == "Sales":
            job_roles.append(np.random.choice(["Sales Executive","Sales Representative","Manager"], p=[0.55,0.30,0.15]))
        elif d == "Research & Development":
            job_roles.append(np.random.choice(["Research Scientist","Laboratory Technician","Manufacturing Director","Healthcare Representative","Research Director","Manager"], p=[0.25,0.25,0.15,0.15,0.10,0.10]))
        else:
            job_roles.append(np.random.choice(["Human Resources","Manager"], p=[0.80,0.20]))

    age                    = np.random.randint(18, 61, n)
    gender                 = np.random.choice(["Male","Female"], n, p=[0.60,0.40])
    marital_status         = np.random.choice(["Single","Married","Divorced"], n, p=[0.32,0.46,0.22])
    education              = np.random.randint(1, 6, n)
    education_field        = np.random.choice(["Life Sciences","Medical","Marketing","Technical Degree","Other","Human Resources"], n, p=[0.41,0.27,0.11,0.09,0.08,0.04])
    distance_from_home     = np.random.randint(1, 30, n)
    monthly_income         = np.random.randint(1009, 20000, n)
    percent_salary_hike    = np.random.randint(11, 26, n)
    stock_option_level     = np.random.randint(0, 4, n)
    num_companies_worked   = np.random.randint(0, 10, n)
    total_working_years    = np.clip(np.random.randint(0, 41, n), 0, 40)
    years_at_company       = np.clip(np.random.randint(0, 41, n), 0, total_working_years)
    years_in_current_role  = np.clip(np.random.randint(0, 19, n), 0, years_at_company)
    years_last_promotion   = np.clip(np.random.randint(0, 16, n), 0, years_at_company)
    years_with_manager     = np.clip(np.random.randint(0, 18, n), 0, years_at_company)
    training_times         = np.random.randint(0, 7, n)
    overtime               = np.random.choice(["Yes","No"], n, p=[0.28,0.72])
    job_level              = np.random.randint(1, 6, n)
    job_involvement        = np.random.randint(1, 5, n)
    job_satisfaction       = np.random.randint(1, 5, n)
    env_satisfaction       = np.random.randint(1, 5, n)
    relationship_sat       = np.random.randint(1, 5, n)
    work_life_balance      = np.random.randint(1, 5, n)
    performance_rating     = np.random.choice([3,4], n, p=[0.85,0.15])
    business_travel        = np.random.choice(["Non-Travel","Travel_Rarely","Travel_Frequently"], n, p=[0.19,0.71,0.10])

    # Attrition probability based on realistic factors
    prob = np.full(n, 0.12)
    prob += (overtime == "Yes")            * 0.18
    prob += (job_satisfaction <= 2)        * 0.10
    prob += (env_satisfaction <= 2)        * 0.08
    prob += (years_at_company <= 2)        * 0.12
    prob += (distance_from_home > 20)      * 0.06
    prob += (marital_status == "Single")   * 0.06
    prob += (job_level == 1)               * 0.08
    prob += (work_life_balance == 1)       * 0.10
    prob += (business_travel == "Travel_Frequently") * 0.07
    prob -= (years_at_company > 10)        * 0.05
    prob -= (job_satisfaction >= 4)        * 0.06
    prob -= (monthly_income > 10000)       * 0.04
    prob = np.clip(prob, 0.02, 0.95)

    attrition = (np.random.rand(n) < prob).astype(int)
    attrition_label = np.where(attrition == 1, "Yes", "No")

    df = pd.DataFrame({
        "Age":                    age,
        "Attrition":              attrition_label,
        "BusinessTravel":         business_travel,
        "Department":             departments,
        "DistanceFromHome":       distance_from_home,
        "Education":              education,
        "EducationField":         education_field,
        "EnvironmentSatisfaction":env_satisfaction,
        "Gender":                 gender,
        "JobInvolvement":         job_involvement,
        "JobLevel":               job_level,
        "JobRole":                job_roles,
        "JobSatisfaction":        job_satisfaction,
        "MaritalStatus":          marital_status,
        "MonthlyIncome":          monthly_income,
        "NumCompaniesWorked":     num_companies_worked,
        "OverTime":               overtime,
        "PercentSalaryHike":      percent_salary_hike,
        "PerformanceRating":      performance_rating,
        "RelationshipSatisfaction":relationship_sat,
        "StockOptionLevel":       stock_option_level,
        "TotalWorkingYears":      total_working_years,
        "TrainingTimesLastYear":  training_times,
        "WorkLifeBalance":        work_life_balance,
        "YearsAtCompany":         years_at_company,
        "YearsInCurrentRole":     years_in_current_role,
        "YearsSinceLastPromotion":years_last_promotion,
        "YearsWithCurrManager":   years_with_manager,
    })
    return df

if __name__ == "__main__":
    df = generate_dataset()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "employee_attrition.csv")
    df.to_csv(out, index=False)
    print(f"Dataset saved -> {out}")
    print(f"Shape: {df.shape}")
    print(df["Attrition"].value_counts())
    print(df.head())