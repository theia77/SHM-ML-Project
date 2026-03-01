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

# ══════════════════════════════════════════════════════
# IS CODE REFERENCES:
# IS 2062:2011  — E=200 GPa constant (not a feature)
#                 Fy = 250/345/410 MPa (IS 2062 grades)
# IS 9077:1979  — CorrosionFactor drives area loss
# IRC 6:2017    — DeadLoad, LiveLoad values
# IS 800:2007   — FOS condition labels:
#                 Safe     → StressRatio < 0.60 (FOS > 1.67)
#                 At-Risk  → StressRatio 0.60-0.80 (FOS 1.25-1.67)
#                 Critical → StressRatio > 0.80 (FOS < 1.25)
# ══════════════════════════════════════════════════════

# ── LOAD DATA ──
df = pd.read_csv('data/bridge_database.csv')
print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nCondition split (IS 800:2007 FOS basis):")
print(df['Condition'].value_counts())

# ── ENCODE MEMBER TYPE ──
le = LabelEncoder()
df['MemberType_enc'] = le.fit_transform(df['MemberType'])

# ── FEATURES — NO LEAKAGE, IS CODE ALIGNED ──
# Removed: AxialForce_kN, Stress_MPa, StressRatio, FOS
#          → these directly define label = data leakage
# Removed: E_effective_GPa
#          → IS 2062:2011 constant = 200 GPa always
#          → zero variance = no predictive value
# Kept   : geometry + IS 9077 corrosion + IS 2062 material
#          + IRC 6 loads = real physical inputs

features = [
    'MemberID',             # member position in truss
    'MemberType_enc',       # structural role (IS 800 member types)
    'Length_m',             # member geometry (m)
    'Area_original_mm2',    # design area — IS 808 section
    'Area_effective_mm2',   # corroded area — IS 9077
    'CorrosionFactor',      # IS 9077 C3: area retention ratio
    'YieldStrength_MPa',    # IS 2062 grade (250/345/410)
    'DeadLoad_kN',          # IRC 6:2017 permanent load
    'LiveLoad_kN',          # IRC 6:2017 Class A live load
]

X = df[features]
y = df['Condition']

print(f"\n✅ Features ({len(features)} — IS code aligned, no leakage):")
for i, f in enumerate(features, 1):
    print(f"   {i}. {f}")
print(f"\n✅ Target: Condition (IS 800:2007 FOS basis)")

# ── TRAIN TEST SPLIT ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"\n✅ Train: {len(X_train)} | Test: {len(X_test)}")
print(f"   Class distribution preserved (stratify=True)")

# ── TRAIN RANDOM FOREST ──
print(f"\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    class_weight='balanced',  # handles At-Risk minority class
    n_jobs=-1
)
model.fit(X_train, y_train)
print(f"✅ Model trained!")

# ── EVALUATE ──
y_pred   = model.predict(X_test)
accuracy = (y_pred == y_test).mean() * 100

print(f"\n{'='*55}")
print(f"  MODEL PERFORMANCE")
print(f"  IS 800 FOS | No Leakage | IS Code Features")
print(f"{'='*55}")
print(f"  Accuracy : {accuracy:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── CONFUSION MATRIX ──
os.makedirs('outputs', exist_ok=True)
cm = confusion_matrix(
    y_test, y_pred,
    labels=['Safe','At-Risk','Critical']
)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Safe','At-Risk','Critical'],
            yticklabels=['Safe','At-Risk','Critical'])
plt.title(
    'Confusion Matrix — ML-Assisted SHM\n'
    'IS 800:2007 FOS | IS 9077 Corrosion | IRC 6 Loads'
)
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
sns.barplot(
    data=feat_df, x='Importance', y='Feature',
    hue='Feature', palette='viridis', legend=False
)
plt.title(
    'Feature Importance — ML-Assisted SHM\n'
    'IS 9077 Corrosion | IS 2062 Material | IRC 6 Loads\n'
    'Physical Parameters Driving Bridge Member Failure'
)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150)
plt.close()
print(f"✅ Feature importance → outputs/feature_importance.png")

# ── SAMPLE PREDICTIONS (IS code scenarios) ──
print(f"\n{'='*55}")
print(f"  SAMPLE PREDICTIONS — IS Code Scenarios")
print(f"{'='*55}")

scenarios = [
    {
        'name': 'IS 2062 E250 | New | Dead load only (IRC 6)',
        'data': {
            'MemberID': 1,
            'MemberType_enc': le.transform(['Bottom Chord'])[0],
            'Length_m': 3.0,
            'Area_original_mm2': 1903.0,
            'Area_effective_mm2': 1903.0,
            'CorrosionFactor': 1.0,       # no corrosion
            'YieldStrength_MPa': 250,
            'DeadLoad_kN': 80,
            'LiveLoad_kN': 0,
        }
    },
    {
        'name': 'IS 2062 E250 | 25yr IS 9077 C3 | IRC Class A',
        'data': {
            'MemberID': 5,
            'MemberType_enc': le.transform(['Bottom Chord'])[0],
            'Length_m': 3.0,
            'Area_original_mm2': 1903.0,
            'Area_effective_mm2': round(1903*0.875, 1),
            'CorrosionFactor': 0.875,     # IS 9077 C3, 25yr
            'YieldStrength_MPa': 250,
            'DeadLoad_kN': 100,
            'LiveLoad_kN': 75,
        }
    },
    {
        'name': 'IS 2062 E345 | 50yr IS 9077 C3 | IRC Overload',
        'data': {
            'MemberID': 24,
            'MemberType_enc': le.transform(['Diagonal'])[0],
            'Length_m': 5.0,
            'Area_original_mm2': 1903.0,
            'Area_effective_mm2': round(1903*0.80, 1),
            'CorrosionFactor': 0.80,      # IS 9077 C3, 50yr
            'YieldStrength_MPa': 345,
            'DeadLoad_kN': 120,
            'LiveLoad_kN': 112,
        }
    },
    {
        'name': 'IS 2062 E250 | Severe IS 9077 | Critical Load',
        'data': {
            'MemberID': 15,
            'MemberType_enc': le.transform(['End Post'])[0],
            'Length_m': 5.0,
            'Area_original_mm2': 1200.0,
            'Area_effective_mm2': round(1200*0.75, 1),
            'CorrosionFactor': 0.75,
            'YieldStrength_MPa': 250,
            'DeadLoad_kN': 150,
            'LiveLoad_kN': 150,
        }
    },
]

for s in scenarios:
    inp   = pd.DataFrame([s['data']])
    pred  = model.predict(inp)[0]
    proba = model.predict_proba(inp)[0]
    print(f"\n  {s['name']}")
    print(f"  Result : ⚡ {pred}")
    for cls, p in zip(model.classes_, proba):
        bar = '█' * int(p*30)
        print(f"    {cls:<10}: {bar} {p*100:.1f}%")

# ── SAVE ──
os.makedirs('models', exist_ok=True)
with open('models/bridge_rf_model.pkl','wb') as f:
    pickle.dump(model, f)
with open('models/label_encoder.pkl','wb') as f:
    pickle.dump(le, f)

print(f"\n✅ Model   → models/bridge_rf_model.pkl")
print(f"✅ Encoder → models/label_encoder.pkl")
print(f"\n🎉 IS Code compliant ML pipeline complete!")
print(f"""
   IS Compliance:
   → IS 2062:2011  E=200GPa removed (constant, zero variance)
   → IS 9077:1979  CorrosionFactor as key feature
   → IRC 6:2017    Dead/Live loads as features
   → IS 800:2007   FOS 1.67/1.25 condition thresholds
   → No data leakage (stress/force/ratio removed)
""")
