import pandas as pd
import numpy as np
import os

# ══════════════════════════════════════════════════════
# PROCESS REAL TRUSS ANALYSIS DATA
# Cleans and standardizes the 3 uploaded CSV snapshots
# for use in ML validation
#
# Input files (raw_data/):
#   truss_snapshot_year5.csv   — light load, new
#   truss_snapshot_year25.csv  — moderate, aging
#   truss_snapshot_year40.csv  — heavy, corroded
#
# Removes leakage columns: Force_kN, Stress_MPa, Ratio
# Adds missing features:   Area_original, Fy, Dead, Live
# Output: data/bridge_real_snapshots.csv
# ══════════════════════════════════════════════════════

# Snapshot metadata — what each file represents
SNAPSHOT_META = {
    'truss_snapshot_year5.csv': {
        'AgeYears':   5,
        'UseCase':    'Urban',
        'Environment':'C3_Urban',
        'Maintenance':'Good',
        'DeadLoad_kN': 220.0,   # inferred from force levels
        'LiveLoad_kN':  30.0,
    },
    'truss_snapshot_year25.csv': {
        'AgeYears':   25,
        'UseCase':    'Urban',
        'Environment':'C3_Urban',
        'Maintenance':'Partial',
        'DeadLoad_kN': 280.0,
        'LiveLoad_kN': 130.0,
    },
    'truss_snapshot_year40.csv': {
        'AgeYears':   40,
        'UseCase':    'Highway',
        'Environment':'C3_Urban',
        'Maintenance':'None',
        'DeadLoad_kN': 320.0,
        'LiveLoad_kN': 200.0,
    },
}

# Member type mapping — clean Type column
TYPE_MAP = {
    '1 (Bottom)':   'Bottom Chord',
    '2 (Top)':      'Top Chord',
    '3 (End Post)': 'End Post',
    '4 (Vertical)': 'Vertical',
    '5 (Diagonal)': 'Diagonal',
}

# Standard design values
AREA_ORIGINAL = 1903.0  # mm² — ISA 100×100×10 (IS 808)
FY_DEFAULT    = 250     # MPa — IS 2062 Grade E250

os.makedirs('data', exist_ok=True)
os.makedirs('raw_data', exist_ok=True)

all_cleaned = []
print("PROCESSING REAL TRUSS ANALYSIS DATA")
print("━"*45)

for fname, meta in SNAPSHOT_META.items():
    fpath = f'raw_data/{fname}'
    if not os.path.exists(fpath):
        print(f"  ⚠️  {fname} not found in raw_data/")
        print(f"      Copy your CSV there and rename it")
        continue

    df = pd.read_csv(fpath)
    print(f"\n  {fname}")
    print(f"  Raw shape : {df.shape}")

    # Clean member type
    df['MemberType'] = df['Type'].map(TYPE_MAP)

    # Add standard design features
    df['Area_original_mm2'] = AREA_ORIGINAL
    df['YieldStrength_MPa'] = FY_DEFAULT

    # Add context from metadata
    for col, val in meta.items():
        df[col] = val

    # Remove leakage columns
    drop_cols = ['Type', 'Force_kN', 'Stress_MPa', 'Ratio']
    df = df.drop(columns=drop_cols)

    # Rename to match ML features
    df = df.rename(columns={
        'ID':         'MemberID',
        'L_m':        'Length_m',
        'A_eff_mm2':  'Area_effective_mm2',
        'CF':         'CorrosionFactor',
        'Cond':       'Condition',
    })

    # Final column order
    final_cols = [
        'MemberID', 'MemberType', 'Length_m',
        'Area_original_mm2', 'Area_effective_mm2',
        'CorrosionFactor', 'YieldStrength_MPa',
        'DeadLoad_kN', 'LiveLoad_kN',
        'AgeYears', 'UseCase', 'Environment', 'Maintenance',
        'Condition'
    ]
    df = df[final_cols]
    all_cleaned.append(df)

    print(f"  Cleaned   : {df.shape}")
    print(f"  Age       : Year {meta['AgeYears']}")
    print(f"  Conditions: {df['Condition'].value_counts().to_dict()}")

if all_cleaned:
    combined = pd.concat(all_cleaned, ignore_index=True)
    combined.to_csv('data/bridge_real_snapshots.csv', index=False)
    print(f"\n✅ Combined real snapshots: {combined.shape}")
    print(f"   Conditions: {combined['Condition'].value_counts().to_dict()}")
    print(f"   Saved → data/bridge_real_snapshots.csv")
else:
    print("\n⚠️  No files processed.")
    print("   Rename your 3 CSVs and place in raw_data/ folder:")
    print("   File1 → raw_data/truss_snapshot_year5.csv")
    print("   File2 → raw_data/truss_snapshot_year25.csv")
    print("   File3 → raw_data/truss_snapshot_year40.csv")
