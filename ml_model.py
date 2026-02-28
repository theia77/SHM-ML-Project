import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# ── LOAD DATA ──
df = pd.read_csv('data/bridge_database.csv')
print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nCondition split:\n{df['Condition'].value_counts()}")

# ── LABEL ENCODING ──
le = LabelEncoder()
df['MemberType_enc'] = le.fit_transform(df['MemberType'])

# ── FEATURES & LABEL ──
# ⚠️ IMPORTANT: Removed AxialForce_kN, Stress_MPa, StressRatio
# These directly define the label → causes data leakage
# ML must now learn structural behavior from physical inputs only

features = [
    'MemberID',          # which member
    'MemberType_enc',    # type (chord/vertical/diagonal)
    'Length_m',          # member length
    'Area_mm2',          # cross section size
    'YieldStrength_MPa', # material strength
    'DeadLoad_kN',       # permanent load
    'LiveLoad_kN',       # traffic load
]

X = df[features]
y = df['Condition']  # Safe / At-Risk / Critical

print(f"\n✅ Features (no leakage): {features}")
print(f"✅ Target: Condition")

# ── TRAIN TEST SPLIT ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y        # keeps class balance in both sets
)
print(f"\n✅ Train: {len(X_train)} | Test: {len(X_test)}")

# ── TRAIN MODEL ──
print(f"\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'  # handles At-Risk being smaller class
)
model.fit(X_train, y_train)
print(f"✅ Model trained!")

# ── EVALUATE ──
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean() * 100

print(f"\n{'='*55}")
print(f"  MODEL PERFORMANCE (No Data Leakage)")
print(f"{'='*55}")
print(f"  Accuracy : {accuracy:.2f}%")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred))

# ── CONFUSION MATRIX ──
os.makedirs('outputs', exist_ok=True)
cm = confusion_matrix(
    y_test, y_pred,
    labels=['Safe', 'At-Risk', 'Critical']
)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Safe', 'At-Risk', 'Critical'],
    yticklabels=['Safe', 'At-Risk', 'Critical']
)
plt.title('Confusion Matrix — Bridge Member Condition Prediction\n(Physical Features Only — No Leakage)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150)
plt.close()
print(f"✅ Confusion matrix → outputs/confusion_matrix.png")

# ── FEATURE IMPORTANCE ──
importances = model.feature_importances_
feat_df = pd.DataFrame({
    'Feature':    features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(f"\n  Feature Importance (what drives failure?):")
print(feat_df.to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feat_df,
    x='Importance',
    y='Feature',
    palette='viridis'
)
plt.title('Feature Importance — What Physical Parameters Drive Bridge Failure?\n(Random Forest — SHM Project)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150)
plt.close()
print(f"✅ Feature importance → outputs/feature_importance.png")

# ── SAMPLE PREDICTION ──
# Simulate new bridge scenario using physical inputs only
# No stress/force values — model predicts from geometry + loads
print(f"\n{'='*55}")
print(f"  SAMPLE PREDICTIONS — New Scenarios")
print(f"{'='*55}")

scenarios = [
    {
        'name': 'Bottom Chord — Light Load',
        'data': {
            'MemberID': 5, 'MemberType_enc': le.transform(['Bottom Chord'])[0],
            'Length_m': 3.0, 'Area_mm2': 1903.0, 'YieldStrength_MPa': 250,
            'DeadLoad_kN': 80, 'LiveLoad_kN': 0
        }
    },
    {
        'name': 'Bottom Chord — Extreme Overload',
        'data': {
            'MemberID': 5, 'MemberType_enc': le.transform(['Bottom Chord'])[0],
            'Length_m': 3.0, 'Area_mm2': 1200.0, 'YieldStrength_MPa': 250,
            'DeadLoad_kN': 150, 'LiveLoad_kN': 150
        }
    },
    {
        'name': 'Diagonal — Full Service Load',
        'data': {
            'MemberID': 24, 'MemberType_enc': le.transform(['Diagonal'])[0],
            'Length_m': 5.0, 'Area_mm2': 1903.0, 'YieldStrength_MPa': 345,
            'DeadLoad_kN': 100, 'LiveLoad_kN': 75
        }
    },
    {
        'name': 'Vertical — Weak Section Overload',
        'data': {
            'MemberID': 18, 'MemberType_enc': le.transform(['Vertical'])[0],
            'Length_m': 4.0, 'Area_mm2': 1200.0, 'YieldStrength_MPa': 250,
            'DeadLoad_kN': 120, 'LiveLoad_kN': 112
        }
    }
]

for s in scenarios:
    inp = pd.DataFrame([s['data']])
    pred = model.predict(inp)[0]
    proba = model.predict_proba(inp)[0]
    print(f"\n  Scenario : {s['name']}")
    print(f"  Result   : {pred}")
    for cls, p in zip(model.classes_, proba):
        bar = '█' * int(p * 25)
        print(f"    {cls:<10}: {bar} {p*100:.1f}%")

# ── SAVE MODEL ──
os.makedirs('models', exist_ok=True)
with open('models/bridge_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(f"\n✅ Model saved   → models/bridge_rf_model.pkl")
print(f"✅ Encoder saved → models/label_encoder.pkl")
print(f"\n🎉 ML Pipeline complete — No leakage, research valid!")
