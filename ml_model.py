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
# IS 2062:2011  — E=200GPa constant | Fy=250/345/410 MPa
# IS 9077:1979  — C2/C3/C4/C5 corrosion environments
# IRC 6:2017    — Rural/Urban/Highway/Industrial loads
# IS 800:2007   — FOS condition labels
#
# FEATURES (13 — no leakage):
# Physics  : MemberID, MemberType, Length, Areas, CF, Fy
# Loading  : DeadLoad, LiveLoad
# Context  : AgeYears, UseCase, Environment, Maintenance
#
# REMOVED (leakage):
# Force_kN, Stress_MPa, StressRatio, FOS, E_effective_GPa
# ══════════════════════════════════════════════════════

# ── LOAD DATA ──
df = pd.read_csv('data/bridge_database.csv')
print(f"✅ Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"\nCondition split (IS 800:2007 FOS basis):")
print(df['Condition'].value_counts())

# ── ENCODE ALL CATEGORICAL FEATURES ──
le_member = LabelEncoder()
le_use    = LabelEncoder()
le_env    = LabelEncoder()
le_maint  = LabelEncoder()

df['MemberType_enc']  = le_member.fit_transform(df['MemberType'])
df['UseCase_enc']     = le_use.fit_transform(df['UseCase'])
df['Environment_enc'] = le_env.fit_transform(df['Environment'])
df['Maintenance_enc'] = le_maint.fit_transform(df['Maintenance'])

print(f"\nEncoded categories:")
print(f"  MemberType  : {list(le_member.classes_)}")
print(f"  UseCase     : {list(le_use.classes_)}")
print(f"  Environment : {list(le_env.classes_)}")
print(f"  Maintenance : {list(le_maint.classes_)}")

# ── FEATURES — 13 TOTAL, NO LEAKAGE ──
features = [
    # Physics features
    'MemberID',             # position in truss
    'MemberType_enc',       # structural role
    'Length_m',             # geometry
    'Area_original_mm2',    # IS 808 design section
    'Area_effective_mm2',   # IS 9077 corroded area
    'CorrosionFactor',      # IS 9077 area retention
    'YieldStrength_MPa',    # IS 2062 grade
    # Loading features
    'DeadLoad_kN',          # IRC 6:2017 permanent
    'LiveLoad_kN',          # IRC 6:2017 Class A
    # Context features
    'AgeYears',             # years in service
    'UseCase_enc',          # Rural/Urban/Highway/Industrial
    'Environment_enc',      # IS 9077 C2/C3/C4/C5
    'Maintenance_enc',      # None/Partial/Good
]

X = df[features]
y = df['Condition']

print(f"\n✅ Features ({len(features)} — IS code aligned, no leakage):")
for i,f in enumerate(features, 1):
    print(f"   {i:>2}. {f}")
print(f"\n✅ Target: Condition (IS 800:2007 FOS basis)")

# ── TRAIN TEST SPLIT ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"\n✅ Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"   Class distribution preserved (stratify=True)")

# ── TRAIN RANDOM FOREST ──
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

# ── EVALUATE ON INTERNAL TEST ──
y_pred   = model.predict(X_test)
accuracy = (y_pred == y_test).mean() * 100

print(f"\n{'='*55}")
print(f"  MODEL PERFORMANCE (Internal 80/20 split)")
print(f"  IS 800 FOS | 13 Features | No Leakage")
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
    'Confusion Matrix — Bridge Member Condition\n'
    'IS 800:2007 FOS | IS 9077 Corrosion | IRC 6 Loads\n'
    '13 Features | No Leakage'
)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150)
plt.close()
print(f"✅ outputs/confusion_matrix.png")

# ── FEATURE IMPORTANCE ──
feat_df = pd.DataFrame({
    'Feature':    features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nFeature Importance:")
print(feat_df.to_string(index=False))

plt.figure(figsize=(11, 7))
colors = ['#ff6b6b' if f in ['AgeYears','UseCase_enc',
          'Environment_enc','Maintenance_enc']
          else '#4ecdc4' for f in feat_df['Feature']]
sns.barplot(data=feat_df, x='Importance', y='Feature',
            hue='Feature', palette=colors, legend=False)
plt.title(
    'Feature Importance — ML-Assisted SHM\n'
    'Red = Context features | Teal = Physics features\n'
    'IS 9077 Corrosion | IS 2062 Material | IRC 6 Loads'
)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150)
plt.close()
print(f"✅ outputs/feature_importance.png")

# ── INDEPENDENT TEST SET (if available) ──
if os.path.exists('data/bridge_test_data.csv'):
    print(f"\n{'='*55}")
    print(f"  INDEPENDENT TEST SET EVALUATION")
    print(f"  Unseen load cases + environments")
    print(f"{'='*55}")

    df_test = pd.read_csv('data/bridge_test_data.csv')
    df_test['MemberType_enc']  = le_member.transform(
        df_test['MemberType'])
    df_test['UseCase_enc']     = le_use.transform(
        df_test['UseCase'])
    df_test['Environment_enc'] = le_env.transform(
        df_test['Environment'])
    df_test['Maintenance_enc'] = le_maint.transform(
        df_test['Maintenance'])

    X_ind    = df_test[features]
    y_ind    = df_test['Condition']
    y_ind_p  = model.predict(X_ind)
    ind_acc  = (y_ind_p == y_ind).mean() * 100

    print(f"\n  Independent rows : {len(X_ind):,}")
    print(f"  Accuracy         : {ind_acc:.2f}%")
    print(f"\nClassification Report (Independent):")
    print(classification_report(y_ind, y_ind_p))

    cm_ind = confusion_matrix(
        y_ind, y_ind_p,
        labels=['Safe','At-Risk','Critical']
    )
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_ind, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Safe','At-Risk','Critical'],
                yticklabels=['Safe','At-Risk','Critical'])
    plt.title(
        'Confusion Matrix — Independent Test Set\n'
        'Unseen Ages + Environments | True Generalization'
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix_independent.png',
                dpi=150)
    plt.close()
    print(f"✅ outputs/confusion_matrix_independent.png")

# ── SAMPLE PREDICTIONS ──
print(f"\n{'='*55}")
print(f"  SAMPLE PREDICTIONS — IS Code Scenarios")
print(f"{'='*55}")

scenarios = [
    {
        'name': 'Year 0 | New | C3 Urban | Dead only',
        'data': {
            'MemberID':           1,
            'MemberType_enc':     le_member.transform(
                                  ['Bottom Chord'])[0],
            'Length_m':           3.0,
            'Area_original_mm2':  1903.0,
            'Area_effective_mm2': 1903.0,
            'CorrosionFactor':    1.0,
            'YieldStrength_MPa':  250,
            'DeadLoad_kN':        220.0,
            'LiveLoad_kN':        0.0,
            'AgeYears':           0,
            'UseCase_enc':        le_use.transform(['Urban'])[0],
            'Environment_enc':    le_env.transform(
                                  ['C3_Urban'])[0],
            'Maintenance_enc':    le_maint.transform(['Good'])[0],
        }
    },
    {
        'name': 'Year 25 | C3 Urban | Partial Maint | IRC Class A',
        'data': {
            'MemberID':           5,
            'MemberType_enc':     le_member.transform(
                                  ['Bottom Chord'])[0],
            'Length_m':           3.0,
            'Area_original_mm2':  1903.0,
            'Area_effective_mm2': round(1903*0.875, 1),
            'CorrosionFactor':    0.875,
            'YieldStrength_MPa':  250,
            'DeadLoad_kN':        280.0,
            'LiveLoad_kN':        130.0,
            'AgeYears':           25,
            'UseCase_enc':        le_use.transform(['Urban'])[0],
            'Environment_enc':    le_env.transform(
                                  ['C3_Urban'])[0],
            'Maintenance_enc':    le_maint.transform(
                                  ['Partial'])[0],
        }
    },
    {
        'name': 'Year 40 | C4 Industrial | No Maint | Overload',
        'data': {
            'MemberID':           5,
            'MemberType_enc':     le_member.transform(
                                  ['Bottom Chord'])[0],
            'Length_m':           3.0,
            'Area_original_mm2':  1903.0,
            'Area_effective_mm2': round(1903*0.80, 1),
            'CorrosionFactor':    0.80,
            'YieldStrength_MPa':  250,
            'DeadLoad_kN':        320.0,
            'LiveLoad_kN':        200.0,
            'AgeYears':           40,
            'UseCase_enc':        le_use.transform(
                                  ['Industrial'])[0],
            'Environment_enc':    le_env.transform(
                                  ['C4_Industrial'])[0],
            'Maintenance_enc':    le_maint.transform(['None'])[0],
        }
    },
    {
        'name': 'Year 50 | C5 Coastal | No Maint | Critical',
        'data': {
            'MemberID':           5,
            'MemberType_enc':     le_member.transform(
                                  ['Bottom Chord'])[0],
            'Length_m':           3.0,
            'Area_original_mm2':  1903.0,
            'Area_effective_mm2': round(1903*0.70, 1),
            'CorrosionFactor':    0.70,
            'YieldStrength_MPa':  250,
            'DeadLoad_kN':        350.0,
            'LiveLoad_kN':        245.0,
            'AgeYears':           50,
            'UseCase_enc':        le_use.transform(
                                  ['Highway'])[0],
            'Environment_enc':    le_env.transform(
                                  ['C5_Coastal'])[0],
            'Maintenance_enc':    le_maint.transform(['None'])[0],
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

# ── SAVE ALL ──
os.makedirs('models', exist_ok=True)
with open('models/bridge_rf_model.pkl','wb') as f:
    pickle.dump(model, f)
with open('models/label_encoder.pkl','wb') as f:
    pickle.dump(le_member, f)
with open('models/le_use.pkl','wb') as f:
    pickle.dump(le_use, f)
with open('models/le_env.pkl','wb') as f:
    pickle.dump(le_env, f)
with open('models/le_maint.pkl','wb') as f:
    pickle.dump(le_maint, f)

print(f"\n✅ Models saved:")
print(f"   models/bridge_rf_model.pkl")
print(f"   models/label_encoder.pkl")
print(f"   models/le_use.pkl")
print(f"   models/le_env.pkl")
print(f"   models/le_maint.pkl")
print(f"\n🎉 ML pipeline complete! (13 features, IS code aligned)")
