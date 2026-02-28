import openseespy.opensees as ops
import numpy as np
import pandas as pd

# Member type labels
member_types = {
    **{i: 'Bottom Chord' for i in range(1, 9)},
    **{i: 'Top Chord' for i in range(9, 17)},
    **{i: 'Vertical' for i in range(17, 24)},
    **{i: 'Diagonal' for i in range(24, 32)}
}

member_list = [
    (1,1,2),(2,2,3),(3,3,4),(4,4,5),
    (5,5,6),(6,6,7),(7,7,8),(8,8,9),
    (9,1,10),(10,10,11),(11,11,12),(12,12,13),
    (13,13,14),(14,14,15),(15,15,16),(16,16,9),
    (17,2,10),(18,3,11),(19,4,12),(20,5,13),
    (21,6,14),(22,7,15),(23,8,16),
    (24,10,3),(25,11,4),(26,12,5),(27,13,6),
    (28,14,7),(29,15,8),(30,10,2),(31,16,9)
]

def build_model(A):
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 2)

    # Nodes
    ops.node(1,0,0); ops.node(2,3,0); ops.node(3,6,0)
    ops.node(4,9,0); ops.node(5,12,0); ops.node(6,15,0)
    ops.node(7,18,0); ops.node(8,21,0); ops.node(9,24,0)
    ops.node(10,3,4); ops.node(11,6,4); ops.node(12,9,4)
    ops.node(13,12,4); ops.node(14,15,4)
    ops.node(15,18,4); ops.node(16,21,4)

    # Supports
    ops.fix(1, 1, 1)
    ops.fix(9, 0, 1)

    # Material
    E = 200e6
    ops.uniaxialMaterial('Elastic', 1, E)

    # Members
    for mem_id, node_i, node_j in member_list:
        ops.element('Truss', mem_id, node_i, node_j, A, 1)

def run_analysis(dead, live, A, Fy, lc_name):
    build_model(A)

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    total = -(dead + live)
    for node in range(2, 9):
        ops.load(node, 0, total)

    ops.system('BandSPD')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    ops.analyze(1)

    results = []
    for mem_id, node_i, node_j in member_list:
        force = ops.basicForce(mem_id)[0]
        stress_mpa = (force / A) / 1000
        stress_ratio = abs(stress_mpa) / (Fy / 1000)

        if stress_ratio < 0.6:
            condition = 'Safe'
        elif stress_ratio < 0.8:
            condition = 'At-Risk'
        else:
            condition = 'Critical'

        results.append({
            'LoadCase': lc_name,
            'MemberID': mem_id,
            'MemberType': member_types[mem_id],
            'NodeI': node_i,
            'NodeJ': node_j,
            'Area_mm2': A * 1e6,
            'YieldStrength_MPa': Fy / 1000,
            'DeadLoad_kN': dead,
            'LiveLoad_kN': live,
            'AxialForce_kN': round(force, 3),
            'Stress_MPa': round(stress_mpa, 3),
            'StressRatio': round(stress_ratio, 4),
            'Condition': condition
        })

    return results

# ── GENERATE LARGE DATASET ──
# Vary loads and material properties to get 500+ rows

all_results = []

# Different section areas (mm² converted to m²)
areas = [1903e-6, 1500e-6, 2500e-6, 3000e-6, 1200e-6]

# Different yield strengths (kN/m²)
yield_strengths = [250000, 345000, 410000]

# Different load combinations
load_cases = [
    (80,  0,   'LC_Light_Dead'),
    (100, 0,   'LC1_DeadOnly'),
    (100, 50,  'LC_Light_Live'),
    (100, 75,  'LC2_FullLoad'),
    (100, 112, 'LC3_Overload'),
    (120, 75,  'LC_HeavyDead'),
    (120, 112, 'LC_Extreme'),
    (150, 150, 'LC_Critical'),
]

total_runs = len(areas) * len(yield_strengths) * len(load_cases)
print(f"Running {total_runs} combinations...")

count = 0
for A in areas:
    for Fy in yield_strengths:
        for dead, live, lc_name in load_cases:
            tag = f"{lc_name}_A{int(A*1e6)}_Fy{int(Fy/1000)}"
            results = run_analysis(dead, live, A, Fy, tag)
            all_results.extend(results)
            count += 1
            if count % 10 == 0:
                print(f"  Progress: {count}/{total_runs} done...")

# Save to CSV
df = pd.DataFrame(all_results)
df.to_csv('data/bridge_database.csv', index=False)

print(f"\n✅ Dataset generated successfully!")
print(f"   Total rows: {len(df)}")
print(f"   Columns: {list(df.columns)}")
print(f"\nCondition distribution:")
print(df['Condition'].value_counts())
print(f"\nSaved to: data/bridge_database.csv")
