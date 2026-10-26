# coding: utf-8
"""
Streamlit Web App - Employee Attrition Prediction
Author : Shrawani Joshi
Project: ML Minor Project
"""

import os, pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE   = os.path.dirname(os.path.abspath(__file__))
MODEL  = os.path.join(BASE, "models")
DATA   = os.path.join(BASE, "data", "employee_attrition.csv")
PLOTS  = os.path.join(BASE, "plots")

@st.cache_resource
def load_models():
    model    = pickle.load(open(os.path.join(MODEL,"attrition_model.pkl"),    "rb"))
    scaler   = pickle.load(open(os.path.join(MODEL,"scaler.pkl"),             "rb"))
    encoders = pickle.load(open(os.path.join(MODEL,"encoders.pkl"),           "rb"))
    fi       = pickle.load(open(os.path.join(MODEL,"feature_importance.pkl"), "rb"))
    return model, scaler, encoders, fi

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

try:
    model, scaler, encoders, fi = load_models()
    df = load_data()
    ready = True
except:
    ready = False

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1a237e,#4a148c);padding:2rem;
border-radius:15px;margin-bottom:1.5rem;text-align:center;}
.main-header h1{color:white;font-size:2.3rem;margin:0;}
.main-header p{color:#e1bee7;margin:0.4rem 0 0;}
.risk-card{border-radius:15px;padding:1.8rem;text-align:center;
margin:0.5rem 0;box-shadow:0 4px 15px rgba(0,0,0,0.12);}
.risk-value{font-size:3rem;font-weight:900;}
.stButton>button{background:linear-gradient(135deg,#1a237e,#4a148c);
color:white;border:none;padding:0.7rem 2rem;border-radius:10px;
font-size:1rem;font-weight:700;width:100%;}
.footer{text-align:center;color:#9e9e9e;padding:1rem;font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>👔 Employee Attrition Predictor</h1>
    <p>Machine Learning Minor Project &nbsp;|&nbsp; Author: <strong>Shrawani Joshi</strong></p>
</div>
""", unsafe_allow_html=True)

if not ready:
    st.error("Models not found! Run generate_dataset.py then train_model.py first.")
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Employee Profile")
    st.markdown("---")
    st.markdown("### Personal Info")
    age            = st.slider("Age", 18, 60, 30)
    gender         = st.selectbox("Gender", ["Male","Female"])
    marital_status = st.selectbox("Marital Status", ["Single","Married","Divorced"])
    distance       = st.slider("Distance from Home (km)", 1, 30, 10)

    st.markdown("---")
    st.markdown("### Job Details")
    department     = st.selectbox("Department", ["Sales","Research & Development","Human Resources"])
    job_role_opts  = {
        "Sales":                    ["Sales Executive","Sales Representative","Manager"],
        "Research & Development":   ["Research Scientist","Laboratory Technician","Manufacturing Director","Healthcare Representative","Research Director","Manager"],
        "Human Resources":          ["Human Resources","Manager"],
    }
    job_role       = st.selectbox("Job Role", job_role_opts[department])
    job_level      = st.slider("Job Level (1=Entry, 5=Senior)", 1, 5, 2)
    overtime       = st.selectbox("OverTime", ["No","Yes"])
    business_travel= st.selectbox("Business Travel", ["Non-Travel","Travel_Rarely","Travel_Frequently"])

    st.markdown("---")
    st.markdown("### Compensation & Growth")
    monthly_income  = st.slider("Monthly Income", 1000, 20000, 5000, step=500)
    salary_hike     = st.slider("Percent Salary Hike (%)", 11, 25, 15)
    stock_option    = st.slider("Stock Option Level", 0, 3, 1)
    training_times  = st.slider("Training Times Last Year", 0, 6, 3)

    st.markdown("---")
    st.markdown("### Experience")
    total_yrs       = st.slider("Total Working Years", 0, 40, 8)
    yrs_company     = st.slider("Years at Company", 0, int(total_yrs) if total_yrs > 0 else 1, min(5, int(total_yrs)))
    yrs_role        = st.slider("Years in Current Role", 0, int(yrs_company) if yrs_company > 0 else 1, min(3, int(yrs_company)))
    yrs_promotion   = st.slider("Years Since Last Promotion", 0, int(yrs_company) if yrs_company > 0 else 1, min(2, int(yrs_company)))
    yrs_manager     = st.slider("Years with Current Manager", 0, int(yrs_company) if yrs_company > 0 else 1, min(3, int(yrs_company)))
    num_companies   = st.slider("No. of Companies Worked", 0, 9, 2)
    education       = st.slider("Education Level (1-5)", 1, 5, 3)
    education_field = st.selectbox("Education Field", ["Life Sciences","Medical","Marketing","Technical Degree","Other","Human Resources"])
    perf_rating     = st.selectbox("Performance Rating", [3, 4])

    st.markdown("---")
    st.markdown("### Satisfaction Scores (1=Low, 4=High)")
    env_sat         = st.slider("Environment Satisfaction", 1, 4, 3)
    job_sat         = st.slider("Job Satisfaction", 1, 4, 3)
    rel_sat         = st.slider("Relationship Satisfaction", 1, 4, 3)
    wlb             = st.slider("Work Life Balance", 1, 4, 3)
    job_inv         = st.slider("Job Involvement", 1, 4, 3)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Attrition Risk")

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Data Analysis", "🤖 Model Info"])

# ── TAB 1: Prediction ─────────────────────────────────────────────────────────
with tab1:
    def build_input():
        row = {
            "Age": age, "DistanceFromHome": distance, "Education": education,
            "EnvironmentSatisfaction": env_sat, "JobInvolvement": job_inv,
            "JobLevel": job_level, "JobSatisfaction": job_sat,
            "MonthlyIncome": monthly_income, "NumCompaniesWorked": num_companies,
            "PercentSalaryHike": salary_hike, "PerformanceRating": perf_rating,
            "RelationshipSatisfaction": rel_sat, "StockOptionLevel": stock_option,
            "TotalWorkingYears": total_yrs, "TrainingTimesLastYear": training_times,
            "WorkLifeBalance": wlb, "YearsAtCompany": yrs_company,
            "YearsInCurrentRole": yrs_role, "YearsSinceLastPromotion": yrs_promotion,
            "YearsWithCurrManager": yrs_manager,
            "BusinessTravel_enc":   encoders["BusinessTravel"].transform([business_travel])[0],
            "Department_enc":       encoders["Department"].transform([department])[0],
            "EducationField_enc":   encoders["EducationField"].transform([education_field])[0],
            "Gender_enc":           encoders["Gender"].transform([gender])[0],
            "JobRole_enc":          encoders["JobRole"].transform([job_role])[0],
            "MaritalStatus_enc":    encoders["MaritalStatus"].transform([marital_status])[0],
            "OverTime_enc":         encoders["OverTime"].transform([overtime])[0],
        }
        return np.array(list(row.values())).reshape(1, -1)

    X_input = scaler.transform(build_input())
    prob    = model.predict_proba(X_input)[0][1]
    pred    = model.predict(X_input)[0]

    if prob < 0.30:
        risk, bg, color, icon, msg = "LOW RISK", "#E8F5E9","#2E7D32","✅","Employee is likely to stay. Engagement is good."
    elif prob < 0.60:
        risk, bg, color, icon, msg = "MEDIUM RISK","#FFF3E0","#E65100","⚠️","Monitor this employee. Some risk factors present."
    else:
        risk, bg, color, icon, msg = "HIGH RISK",  "#FFEBEE","#B71C1C","🚨","High chance of leaving. Immediate action recommended."

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"""
        <div class="risk-card" style="background:{bg};border:2px solid {color};">
            <div style="font-size:1rem;color:#555;">Attrition Probability</div>
            <div class="risk-value" style="color:{color};">{prob*100:.1f}%</div>
            <div style="font-size:1.3rem;font-weight:700;color:{color};margin-top:0.3rem;">
                {icon} {risk}
            </div>
            <div style="color:{color};margin-top:0.5rem;font-size:0.95rem;">{msg}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction", "Will Leave" if pred==1 else "Will Stay",
                delta="⚠️ Risk" if pred==1 else "Safe")
    col2.metric("Attrition Probability", f"{prob*100:.1f}%")
    col3.metric("Confidence", f"{max(prob, 1-prob)*100:.1f}%")

    st.markdown("---")
    # Risk Gauge
    st.markdown("### 📉 Risk Meter")
    fig, ax = plt.subplots(figsize=(10, 1.6))
    segments = [(0,0.30,"#2E7D32","Low Risk"),(0.30,0.60,"#E65100","Medium Risk"),(0.60,1.0,"#B71C1C","High Risk")]
    for lo, hi, col, lbl in segments:
        ax.barh(0, hi-lo, left=lo, height=0.5, color=col, alpha=0.85)
        ax.text((lo+hi)/2, 0, lbl, ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
    ax.axvline(prob, color="black", lw=3, zorder=5)
    ax.annotate(f"  {prob*100:.1f}%", xy=(prob, 0.26),
                fontsize=12, fontweight="bold", color="black")
    ax.set_xlim(0,1); ax.set_ylim(-0.4, 0.4); ax.axis("off")
    ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#f8f9fa")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### 💡 HR Recommendations")
    recs = []
    if overtime == "Yes":         recs.append("Reduce overtime workload or offer compensation")
    if job_sat <= 2:              recs.append("Conduct 1-on-1 to address job dissatisfaction")
    if env_sat <= 2:              recs.append("Improve workplace environment and culture")
    if wlb <= 2:                  recs.append("Offer flexible working hours or remote options")
    if yrs_promotion >= 4:        recs.append("Consider promotion or new responsibilities")
    if monthly_income < 4000:     recs.append("Review compensation against market standards")
    if business_travel == "Travel_Frequently": recs.append("Reduce frequent travel requirements")
    if marital_status == "Single" and distance > 20: recs.append("Offer relocation assistance or WFH")
    if not recs:                  recs.append("Employee profile looks healthy - keep it up!")
    for r in recs:
        st.write(f"- {r}")

# ── TAB 2: Data Analysis ──────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Dataset Overview")
    c1,c2,c3,c4 = st.columns(4)
    total  = len(df)
    left   = (df["Attrition"]=="Yes").sum()
    stayed = total - left
    rate   = left/total*100
    c1.metric("Total Employees", f"{total:,}")
    c2.metric("Stayed", f"{stayed:,}")
    c3.metric("Left", f"{left:,}")
    c4.metric("Attrition Rate", f"{rate:.1f}%")

    st.markdown("---")
    if os.path.exists(PLOTS):
        plot_files = sorted([f for f in os.listdir(PLOTS) if f.endswith(".png")])
        titles = [
            "Attrition Distribution","Attrition by Department",
            "Age Distribution by Attrition","Monthly Income vs Attrition",
            "OverTime vs Attrition","Feature Importance",
            "Model Comparison","ROC Curve","Confusion Matrix","Correlation Heatmap"
        ]
        cols = st.columns(2)
        for i, pf in enumerate(plot_files):
            title = titles[i] if i < len(titles) else pf
            cols[i%2].markdown(f"#### {title}")
            cols[i%2].image(os.path.join(PLOTS, pf), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Sample Data")
    st.dataframe(df.head(15), use_container_width=True)

# ── TAB 3: Model Info ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🤖 ML Pipeline")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Models Trained")
        for m in ["Logistic Regression","Decision Tree","Random Forest",
                  "Gradient Boosting","AdaBoost","K-Nearest Neighbors","SVM"]:
            st.write(f"- {m}")
    with col2:
        st.markdown("#### Best Model Used")
        st.info(f"**{type(model).__name__}**\n\nSelected based on F1 Score (handles class imbalance better than accuracy)")

    st.markdown("---")
    st.markdown("#### Input Features (27 features)")
    feat_cats = {
        "Personal":      ["Age","Gender","MaritalStatus","DistanceFromHome","Education","EducationField"],
        "Job":           ["Department","JobRole","JobLevel","OverTime","BusinessTravel","JobInvolvement"],
        "Compensation":  ["MonthlyIncome","PercentSalaryHike","StockOptionLevel"],
        "Experience":    ["TotalWorkingYears","YearsAtCompany","NumCompaniesWorked","TrainingTimesLastYear","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"],
        "Satisfaction":  ["EnvironmentSatisfaction","JobSatisfaction","RelationshipSatisfaction","WorkLifeBalance","PerformanceRating"],
    }
    for cat, feats in feat_cats.items():
        st.write(f"**{cat}:** {', '.join(feats)}")

    st.markdown("---")
    st.markdown("#### Class Imbalance Handling")
    st.markdown("""
    The dataset is imbalanced (~16%% attrition rate).
    We used **upsampling (oversampling)** of the minority class to balance the training set.
    Evaluation uses **F1 Score** and **AUC-ROC** instead of plain accuracy.
    """)

    st.markdown("---")
    st.markdown("#### Pipeline")
    st.code("""
Input Features (27)
     |
Label Encoding (Categorical -> Numeric)
     |
Upsample Minority Class (Balance dataset)
     |
Standard Scaler (Normalize features)
     |
Train 7 ML Models
     |
Select Best by F1 Score
     |
Predict: Will Employee Leave? (Yes / No)
    """)

st.markdown("---")
st.markdown("""
<div class="footer">
    Employee Attrition Predictor | ML Minor Project |
    <strong>Shrawani Joshi</strong> | 2025-26
</div>
""", unsafe_allow_html=True)