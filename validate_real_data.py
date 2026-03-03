import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# ══════════════════════════════════════════════════════
# REAL DATA VALIDATION
# Tests ML model on actual truss analysis snapshots
# This is the most important validation —
# model trained on generated data, tested on real data
# ══════════════════════════════════════════════════════

print("="*55)
print("  REAL DATA VALIDATION")
print("  ML model vs actual truss analysis results")
print("="*55)

# Load model and encoders
with open('models/bridge_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('models/le_use.pkl', 'rb') as f:
    le_use = pickle.load(f)
with open('models/le_env.pkl', 'rb') as f:
    le_env = pickle.load(f)
with open('models/le_maint.pkl', 'rb') as f:
    le_maint = pickle.load(f)

# Load real data
if not os.path.exists('data/bridge_real_snapshots.csv'):
    print("❌ Run process_real_data.py first")
    exit()

df = pd.read_csv('data/bridge_real_snapshots.csv')
print(f"\nReal snapshots loaded: {df.shape}")
print(f"Conditions: {df['Condition'].value_counts().to_dict()}")

# Encode
df['MemberType_enc']  = le.transform(df['MemberType'])
df['UseCase_enc']     = le_use.transform(df['UseCase'])
df['Environment_enc'] = le_env.transform(df['Environment'])
df['Maintenance_enc'] = le_maint.transform(df['Maintenance'])

ML_FEATURES = [
    'MemberID', 'MemberType_enc', 'Length_m',
    'Area_original_mm2', 'Area_effective_mm2',
    'CorrosionFactor', 'YieldStrength_MPa',
    'DeadLoad_kN', 'LiveLoad_kN',
    'AgeYears', 'UseCase_enc',
    'Environment_enc', 'Maintenance_enc',
]

X_real = df[ML_FEATURES]
y_real = df['Condition']

y_pred  = model.predict(X_real)
probas  = model.predict_proba(X_real)
accuracy= (y_pred == y_real).mean() * 100

print(f"\n{'='*55}")
print(f"  VALIDATION RESULTS")
print(f"{'='*55}")
print(f"  Rows tested : {len(X_real)} "
      f"(3 real snapshots × 29 members)")
print(f"  Accuracy    : {accuracy:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_real, y_pred))

# Per snapshot accuracy
print("Per Snapshot Accuracy:")
for age in df['AgeYears'].unique():
    mask   = df['AgeYears'] == age
    acc    = (y_pred[mask] == y_real[mask]).mean() * 100
    n_corr = (y_pred[mask] == y_real[mask]).sum()
    print(f"  Year {age:>2} : {acc:.0f}% "
          f"({n_corr}/{mask.sum()} correct)")

# Confusion matrix
os.makedirs('outputs', exist_ok=True)
cm = confusion_matrix(
    y_real, y_pred,
    labels=['Safe','At-Risk','Critical']
)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Safe','At-Risk','Critical'],
            yticklabels=['Safe','At-Risk','Critical'])
plt.title(
    'ML Model vs Real Truss Analysis Data\n'
    '3 Snapshots (Year 5, 25, 40) × 29 Members = 87 rows'
)
plt.ylabel('Actual (from real analysis)')
plt.xlabel('Predicted (ML model)')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix_realdata.png', dpi=150)
plt.close()
print(f"\n✅ outputs/confusion_matrix_realdata.png")
print(f"\n🎉 Real data validation complete!")
