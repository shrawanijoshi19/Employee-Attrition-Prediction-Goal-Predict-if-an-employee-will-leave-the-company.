# coding: utf-8
"""
Train ML models to predict Employee Attrition (Yes/No).
Author : Shrawani Joshi
Project: ML Minor Project - Employee Attrition Prediction
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree             import DecisionTreeClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.svm              import SVC
from sklearn.metrics          import (accuracy_score, classification_report,
                                      confusion_matrix, roc_auc_score, roc_curve,
                                      precision_recall_curve, f1_score)
from sklearn.utils            import resample

warnings.filterwarnings("ignore")

BASE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(BASE, "data",   "employee_attrition.csv")
MODEL = os.path.join(BASE, "models")
PLOTS = os.path.join(BASE, "plots")
os.makedirs(MODEL, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

print("=" * 65)
print("   ML MINOR PROJECT - EMPLOYEE ATTRITION PREDICTION")
print("   Author : Shrawani Joshi")
print("=" * 65)

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
print(f"\nDataset shape : {df.shape}")
print(f"Attrition split:\n{df['Attrition'].value_counts()}")

# ── Encode Target ──────────────────────────────────────────────────────────────
df["Attrition_enc"] = (df["Attrition"] == "Yes").astype(int)

# ── Encode Categorical Features ────────────────────────────────────────────────
cat_cols = ["BusinessTravel","Department","EducationField",
            "Gender","JobRole","MaritalStatus","OverTime"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

feature_cols = [
    "Age","DistanceFromHome","Education","EnvironmentSatisfaction",
    "JobInvolvement","JobLevel","JobSatisfaction","MonthlyIncome",
    "NumCompaniesWorked","PercentSalaryHike","PerformanceRating",
    "RelationshipSatisfaction","StockOptionLevel","TotalWorkingYears",
    "TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany",
    "YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager",
] + [c + "_enc" for c in cat_cols]

X = df[feature_cols]
y = df["Attrition_enc"]

# ── Handle Class Imbalance (upsample minority) ─────────────────────────────────
df_model   = X.copy(); df_model["target"] = y.values
df_maj     = df_model[df_model["target"] == 0]
df_min     = df_model[df_model["target"] == 1]
df_min_up  = resample(df_min, replace=True, n_samples=len(df_maj), random_state=42)
df_bal     = pd.concat([df_maj, df_min_up])
X_bal      = df_bal[feature_cols]
y_bal      = df_bal["target"]

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_bal)

X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_bal, test_size=0.2,
                                           random_state=42, stratify=y_bal)

# Original (unbalanced) test set for realistic evaluation
X_orig_sc = scaler.transform(X)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_orig_sc, y, test_size=0.2,
                                                random_state=42, stratify=y)

# ── Models ─────────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression":      LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree":            DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":            RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "Gradient Boosting":        GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    "AdaBoost":                 AdaBoostClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors":      KNeighborsClassifier(n_neighbors=7),
    "SVM":                      SVC(probability=True, random_state=42),
}

print("\n" + "-"*65)
print("MODEL TRAINING & EVALUATION")
print("-"*65)

results   = []
best_name, best_model, best_f1 = "", None, 0

for name, model in models.items():
    model.fit(X_tr, y_tr)
    preds    = model.predict(X_te)
    probs    = model.predict_proba(X_te)[:, 1]
    acc      = accuracy_score(y_te, preds)
    f1       = f1_score(y_te, preds)
    auc      = roc_auc_score(y_te, probs)
    results.append({"Model": name, "Accuracy": round(acc*100,2),
                    "F1 Score": round(f1,4), "AUC-ROC": round(auc,4)})
    print(f"\n{name}")
    print(f"  Accuracy={acc*100:.2f}%  F1={f1:.4f}  AUC={auc:.4f}")
    if f1 > best_f1:
        best_f1, best_name, best_model = f1, name, model

print(f"\nBest Model : {best_name}  (F1={best_f1:.4f})")
print("\nDetailed Report (Best Model):")
print(classification_report(y_te, best_model.predict(X_te),
                             target_names=["Stay","Leave"]))

# ── Feature Importance ─────────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_tr, y_tr)
fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

# ── Save ──────────────────────────────────────────────────────────────────────
pickle.dump(best_model, open(os.path.join(MODEL, "attrition_model.pkl"),  "wb"))
pickle.dump(scaler,     open(os.path.join(MODEL, "scaler.pkl"),           "wb"))
pickle.dump(encoders,   open(os.path.join(MODEL, "encoders.pkl"),         "wb"))
pickle.dump(fi,         open(os.path.join(MODEL, "feature_importance.pkl"),"wb"))
print("\nModels saved to /models/")

# ── PLOTS ──────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"Stay": "#2196F3", "Leave": "#F44336"}

# 1. Attrition Distribution
fig, ax = plt.subplots(figsize=(7, 5))
counts = df["Attrition"].value_counts()
bars   = ax.bar(["Stay (No)", "Leave (Yes)"], counts.values,
                color=["#2196F3","#F44336"], width=0.5, edgecolor="white")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
            f"{val}\n({val/len(df)*100:.1f}%)", ha="center", fontweight="bold")
ax.set_title("Employee Attrition Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Employees")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"01_attrition_distribution.png"), dpi=150)
plt.close()

# 2. Attrition by Department
fig, ax = plt.subplots(figsize=(9, 5))
dept_attr = df.groupby("Department")["Attrition_enc"].mean().sort_values(ascending=False)*100
dept_attr.plot(kind="bar", color="#EF5350", ax=ax, edgecolor="white")
ax.set_title("Attrition Rate by Department (%)", fontsize=14, fontweight="bold")
ax.set_ylabel("Attrition Rate (%)"); ax.set_xlabel("")
plt.xticks(rotation=20, ha="right")
for i, v in enumerate(dept_attr):
    ax.text(i, v+0.3, f"{v:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"02_attrition_by_dept.png"), dpi=150)
plt.close()

# 3. Age Distribution by Attrition
fig, ax = plt.subplots(figsize=(10, 5))
for label, color in [("Yes","#F44336"),("No","#2196F3")]:
    subset = df[df["Attrition"]==label]["Age"]
    sns.kdeplot(subset, ax=ax, color=color, fill=True, alpha=0.4,
                label=f"{'Leave' if label=='Yes' else 'Stay'}")
ax.set_title("Age Distribution by Attrition", fontsize=14, fontweight="bold")
ax.set_xlabel("Age"); ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"03_age_distribution.png"), dpi=150)
plt.close()

# 4. Monthly Income vs Attrition
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Attrition", y="MonthlyIncome",
            palette={"No":"#2196F3","Yes":"#F44336"}, ax=ax)
ax.set_xticklabels(["Stay","Leave"])
ax.set_title("Monthly Income vs Attrition", fontsize=14, fontweight="bold")
ax.set_xlabel(""); ax.set_ylabel("Monthly Income")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"04_income_vs_attrition.png"), dpi=150)
plt.close()

# 5. OverTime vs Attrition
fig, ax = plt.subplots(figsize=(8, 5))
ot = df.groupby(["OverTime","Attrition"]).size().unstack()
ot.plot(kind="bar", ax=ax, color=["#2196F3","#F44336"], edgecolor="white")
ax.set_title("OverTime vs Attrition", fontsize=14, fontweight="bold")
ax.set_xlabel("OverTime"); ax.set_ylabel("Count")
ax.set_xticklabels(["No OverTime","OverTime"], rotation=0)
ax.legend(["Stay","Leave"])
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"05_overtime_attrition.png"), dpi=150)
plt.close()

# 6. Feature Importance
fig, ax = plt.subplots(figsize=(10, 7))
fi.head(15).sort_values().plot(kind="barh", color="#5C6BC0", ax=ax)
ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"06_feature_importance.png"), dpi=150)
plt.close()

# 7. Model Comparison
res_df = pd.DataFrame(results)
fig, ax = plt.subplots(figsize=(11, 5))
x      = np.arange(len(res_df))
w      = 0.25
b1 = ax.bar(x - w,   res_df["Accuracy"], w, label="Accuracy (%)", color="#42A5F5")
b2 = ax.bar(x,       res_df["F1 Score"]*100, w, label="F1 Score x100", color="#66BB6A")
b3 = ax.bar(x + w,   res_df["AUC-ROC"]*100, w, label="AUC-ROC x100", color="#FFA726")
ax.set_xticks(x); ax.set_xticklabels(res_df["Model"], rotation=20, ha="right")
ax.set_ylim(0, 115); ax.set_ylabel("Score")
ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"07_model_comparison.png"), dpi=150)
plt.close()

# 8. ROC Curve (best model)
probs_best = best_model.predict_proba(X_te)[:,1]
fpr, tpr, _ = roc_curve(y_te, probs_best)
auc_val     = roc_auc_score(y_te, probs_best)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#E53935", lw=2, label=f"ROC Curve (AUC = {auc_val:.3f})")
ax.plot([0,1],[0,1],"k--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve - {best_name}", fontsize=13, fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"08_roc_curve.png"), dpi=150)
plt.close()

# 9. Confusion Matrix
cm = confusion_matrix(y_te, best_model.predict(X_te))
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Stay","Leave"], yticklabels=["Stay","Leave"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"09_confusion_matrix.png"), dpi=150)
plt.close()

# 10. Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 9))
num_cols = ["Age","MonthlyIncome","YearsAtCompany","TotalWorkingYears",
            "DistanceFromHome","JobSatisfaction","EnvironmentSatisfaction",
            "WorkLifeBalance","YearsInCurrentRole","Attrition_enc"]
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, annot_kws={"size":8})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"10_correlation_heatmap.png"), dpi=150)
plt.close()

print("\nAll plots saved to /plots/")
print("\nTraining complete!")