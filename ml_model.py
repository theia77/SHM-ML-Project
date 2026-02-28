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

# ── ENCODE MEMBER TYPE ──
le = LabelEncoder()
df['MemberType_enc'] = le.fit_transform(df['MemberType'])

# ── FEATURES — NO LEAKAGE ──
# Removed: AxialForce_kN, Stress_MPa, StressRatio, FOS
# Model learns from physical inputs only
# Now includes degradation factors — real SHM behavior

features = [
    'MemberID',             # position in truss
    'MemberType_enc',       # structural role
    'Length_m',             # geometry
    'Area_original_mm2',    # design section
    'Area_effective_mm2',   # actual section after corrosion
    'CorrosionFactor',      # how corroded (1.0=new, 0.7=30% corroded)
    'E_effective_GPa',      # stiffness after degradation
    'YieldStrength_MPa',    # material grade
    'DeadLoad_kN',          # permanent load
    'LiveLoad_kN',          # traffic load
]

X = df[features]
y = df['Condition']

print(f"\n✅ Features (no leakage, includes degradation):")
for f in features:
    print(f"   - {f}")

# ── TRAIN TEST SPLIT ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"\n✅ Train: {len(X_train)} | Test: {len(X_test)}")

# ── TRAIN MODEL ──
print(f"\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
model.fit(X_train, y_train)
print(f"✅ Model trained!")

# ── EVALUATE ──
y_pred   = model.predict(X_test)
accuracy = (y_pred == y_test).mean() * 100

print(f"\n{'='*55}")
print(f"  MODEL PERFORMANCE")
print(f"  (Physical + Degradation Features — No Leakage)")
print(f"{'='*55}")
print(f"  Accuracy : {accuracy:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── CONFUSION MATRIX ──
os.makedirs('outputs', exist_ok=True)
cm = confusion_matrix(y_test, y_pred, labels=['Safe','At-Risk','Critical'])
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Safe','At-Risk','Critical'],
            yticklabels=['Safe','At-Risk','Critical'])
plt.title('Confusion Matrix\nBridge SHM — With Corrosion & Degradation')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150)
plt.close()
print(f"✅ Confusion matrix → outputs/confusion_matrix.png")

# ── FEATURE IMPORTANCE ──
feat_df = pd.DataFrame({
    'Feature':    features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nFeature Importance:")
print(feat_df.to_string(index=False))

plt.figure(figsize=(10,6))
sns.barplot(data=feat_df, x='Importance', y='Feature',
            hue='Feature', palette='viridis', legend=False)
plt.title('What Physical Parameters Drive Bridge Failure?\n(Random Forest Feature Importance)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150)
plt.close()
print(f"✅ Feature importance → outputs/feature_importance.png")

# ── SAMPLE PREDICTIONS ──
print(f"\n{'='*55}")
print(f"  SAMPLE PREDICTIONS — SHM Scenarios")
print(f"{'='*55}")

scenarios = [
    {
        'name': 'New bridge — Light Load',
        'data': {
            'MemberID':1, 'MemberType_enc':le.transform(['Bottom Chord'])[0],
            'Length_m':3.0, 'Area_original_mm2':1903.0,
            'Area_effective_mm2':1903.0, 'CorrosionFactor':1.0,
            'E_effective_GPa':200.0, 'YieldStrength_MPa':250,
            'DeadLoad_kN':80, 'LiveLoad_kN':0
        }
    },
    {
        'name': '30% Corroded — Full Load',
        'data': {
            'MemberID':5, 'MemberType_enc':le.transform(['Bottom Chord'])[0],
            'Length_m':3.0, 'Area_original_mm2':1903.0,
            'Area_effective_mm2':1903*0.70, 'CorrosionFactor':0.70,
            'E_effective_GPa':180.0, 'YieldStrength_MPa':250,
            'DeadLoad_kN':100, 'LiveLoad_kN':75
        }
    },
    {
        'name': '20% Corroded Diagonal — Overload',
        'data': {
            'MemberID':24, 'MemberType_enc':le.transform(['Diagonal'])[0],
            'Length_m':5.0, 'Area_original_mm2':1500.0,
            'Area_effective_mm2':1500*0.80, 'CorrosionFactor':0.80,
            'E_effective_GPa':190.0, 'YieldStrength_MPa':345,
            'DeadLoad_kN':120, 'LiveLoad_kN':112
        }
    },
    {
        'name': 'Severely Degraded Vertical — Critical Load',
        'data': {
            'MemberID':18, 'MemberType_enc':le.transform(['Vertical'])[0],
            'Length_m':4.0, 'Area_original_mm2':1200.0,
            'Area_effective_mm2':1200*0.72, 'CorrosionFactor':0.72,
            'E_effective_GPa':165.0, 'YieldStrength_MPa':250,
            'DeadLoad_kN':150, 'LiveLoad_kN':150
        }
    }
]

for s in scenarios:
    inp   = pd.DataFrame([s['data']])
    pred  = model.predict(inp)[0]
    proba = model.predict_proba(inp)[0]
    print(f"\n  Scenario : {s['name']}")
    print(f"  Result   : ⚡ {pred}")
    for cls, p in zip(model.classes_, proba):
        bar = '█' * int(p * 30)
        print(f"    {cls:<10}: {bar} {p*100:.1f}%")

# ── SAVE ──
os.makedirs('models', exist_ok=True)
with open('models/bridge_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(f"\n✅ Model  → models/bridge_rf_model.pkl")
print(f"✅ Encoder → models/label_encoder.pkl")
print(f"\n🎉 Research-level SHM pipeline complete!")
print(f"   No leakage | Corrosion | Degradation | Load noise")
