import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
MODEL_DIR = os.path.join(BASE, 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(42)
N = 4119

jobs = ['admin.', 'blue-collar', 'technician', 'services', 'management',
        'retired', 'entrepreneur', 'self-employed', 'housemaid', 'unemployed', 'student', 'unknown']
marital = ['married', 'single', 'divorced', 'unknown']
education = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
             'professional.course', 'university.degree', 'unknown']
contact_types = ['cellular', 'telephone']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
days = ['mon', 'tue', 'wed', 'thu', 'fri']
poutcomes = ['failure', 'nonexistent', 'success']

age = np.random.normal(40, 12, N).clip(18, 95).astype(int)
job_col = np.random.choice(jobs, N, p=[0.25, 0.22, 0.17, 0.09, 0.07,
                                        0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01])
marital_col = np.random.choice(marital, N, p=[0.61, 0.28, 0.10, 0.01])
education_col = np.random.choice(education, N, p=[0.10, 0.06, 0.14, 0.23,
                                                    0.004, 0.13, 0.30, 0.036])
default_col = np.random.choice(['no', 'yes', 'unknown'], N, p=[0.79, 0.02, 0.19])
housing_col = np.random.choice(['no', 'yes', 'unknown'], N, p=[0.45, 0.53, 0.02])
loan_col = np.random.choice(['no', 'yes', 'unknown'], N, p=[0.83, 0.15, 0.02])
contact_col = np.random.choice(contact_types, N, p=[0.64, 0.36])
month_col = np.random.choice(months, N, p=[0.01, 0.01, 0.02, 0.03, 0.33,
                                            0.05, 0.18, 0.16, 0.05, 0.03, 0.10, 0.03])
day_col = np.random.choice(days, N)
duration = np.random.exponential(260, N).clip(0, 4918).astype(int)
campaign = np.random.geometric(0.35, N).clip(1, 56)
pdays = np.where(np.random.random(N) < 0.14, np.random.randint(1, 999, N), 999)
previous = np.where(pdays < 999, np.random.randint(1, 10, N), 0)
poutcome_col = np.where(previous > 0,
                         np.random.choice(['failure', 'success'], N, p=[0.65, 0.35]),
                         'nonexistent')
emp_var_rate = np.random.choice([-3.4, -3.0, -2.9, -1.8, -1.7, -1.1, 1.1, 1.4], N)
cons_price_idx = np.random.normal(93.5, 0.6, N).round(3)
cons_conf_idx = np.random.normal(-40, 4.6, N).round(1)
euribor3m = np.random.normal(3.6, 1.7, N).clip(0.6, 5.0).round(3)
nr_employed = np.random.choice([4963.6, 5008.7, 5017.5, 5023.5, 5099.1,
                                  5176.3, 5191.0, 5195.8, 5228.1], N)

logit = (
    -2.0
    + 0.01 * (age - 40)
    - 0.2 * (campaign > 5)
    + 1.5 * (poutcome_col == 'success').astype(float)
    - 0.3 * (poutcome_col == 'failure').astype(float)
    + 0.5 * np.isin(month_col, ['mar', 'oct', 'sep', 'dec']).astype(float)
    + 0.4 * (job_col == 'student').astype(float)
    + 0.3 * (job_col == 'retired').astype(float)
    - 0.2 * (housing_col == 'yes').astype(float)
    + 0.003 * duration
    - 0.3 * emp_var_rate
    + 0.1 * cons_conf_idx / 10
    + np.random.normal(0, 0.3, N)
)
prob = 1 / (1 + np.exp(-logit))
target = (np.random.random(N) < prob).astype(int)

df = pd.DataFrame({
    'age': age, 'job': job_col, 'marital': marital_col, 'education': education_col,
    'default': default_col, 'housing': housing_col, 'loan': loan_col,
    'contact': contact_col, 'month': month_col, 'day_of_week': day_col,
    'duration': duration, 'campaign': campaign, 'pdays': pdays,
    'previous': previous, 'poutcome': poutcome_col,
    'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
    'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m,
    'nr.employed': nr_employed, 'y': target
})

df.to_csv(os.path.join(DATA_DIR, 'bank_data.csv'), index=False)
print(f"Dataset generated: {df.shape}, conversion rate: {df['y'].mean():.1%}")

df_enc = df.copy()
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'day_of_week', 'poutcome']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, os.path.join(MODEL_DIR, 'encoders.pkl'))

feature_cols = [c for c in df_enc.columns if c != 'y']
X = df_enc[feature_cols]
y = df_enc['y']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'feature_cols.pkl'))

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

lr = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5, random_state=42)
lr.fit(X_train, y_train)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
print(f"Logistic Regression AUC: {lr_auc:.4f}")
joblib.dump(lr, os.path.join(MODEL_DIR, 'lr_model.pkl'))

rf = RandomForestClassifier(n_estimators=50, max_depth=8, class_weight='balanced',
                             random_state=42, n_jobs=1)
rf.fit(X_train, y_train)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"Random Forest AUC: {rf_auc:.4f}")
joblib.dump(rf, os.path.join(MODEL_DIR, 'rf_model.pkl'))

fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
joblib.dump(fi, os.path.join(MODEL_DIR, 'feature_importance.pkl'))

seg_features = ['age', 'campaign', 'duration', 'previous', 'cons.conf.idx', 'euribor3m']
X_seg = df[seg_features].copy()
seg_scaler = StandardScaler()
X_seg_scaled = seg_scaler.fit_transform(X_seg)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_seg_scaled)

joblib.dump(kmeans, os.path.join(MODEL_DIR, 'kmeans.pkl'))
joblib.dump(seg_scaler, os.path.join(MODEL_DIR, 'seg_scaler.pkl'))
joblib.dump(seg_features, os.path.join(MODEL_DIR, 'seg_features.pkl'))

cluster_profiles = df.groupby('cluster').agg(
    size=('y', 'count'),
    conversion_rate=('y', 'mean'),
    avg_age=('age', 'mean'),
    avg_duration=('duration', 'mean'),
    avg_campaign=('campaign', 'mean'),
    avg_previous=('previous', 'mean'),
).round(2)
cluster_profiles['label'] = ['Skeptics', 'Loyalists', 'Prospects', 'Champions']
joblib.dump(cluster_profiles, os.path.join(MODEL_DIR, 'cluster_profiles.pkl'))
print(f"K-Means clustering done: 4 segments")

test_results = {
    'lr_auc': lr_auc,
    'rf_auc': rf_auc,
    'y_test': y_test.values,
    'lr_pred_proba': lr.predict_proba(X_test)[:, 1],
    'rf_pred_proba': rf.predict_proba(X_test)[:, 1],
    'lr_pred': lr.predict(X_test),
    'rf_pred': rf.predict(X_test),
    'conversion_rate': df['y'].mean(),
    'n_samples': len(df),
}
joblib.dump(test_results, os.path.join(MODEL_DIR, 'test_results.pkl'))

df.to_csv(os.path.join(DATA_DIR, 'bank_data.csv'), index=False)
print("All models trained and saved successfully!")
