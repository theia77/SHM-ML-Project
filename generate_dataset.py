import numpy as np
import pandas as pd
import os

nodes = {
    1:(0,0),2:(3,0),3:(6,0),4:(9,0),5:(12,0),
    6:(15,0),7:(18,0),8:(21,0),9:(24,0),
    10:(3,4),11:(6,4),12:(9,4),13:(12,4),
    14:(15,4),15:(18,4),16:(21,4)
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

member_types = {
    **{i:'Bottom Chord' for i in range(1,9)},
    **{i:'Top Chord'    for i in range(9,17)},
    **{i:'Vertical'     for i in range(17,24)},
    **{i:'Diagonal'     for i in range(24,32)}
}

def solve_truss(dead_kN, live_kN, A_m2, E=200e9):
    n_dof = len(nodes) * 2
    K = np.zeros((n_dof, n_dof))

    for mem_id, ni, nj in member_list:
        xi,yi = nodes[ni]
        xj,yj = nodes[nj]
        L = np.sqrt((xj-xi)**2 + (yj-yi)**2)
        c,s = (xj-xi)/L, (yj-yi)/L
        k = E * A_m2 / L
        ke = k * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])
        dofs = [2*(ni-1), 2*(ni-1)+1, 2*(nj-1), 2*(nj-1)+1]
        for r in range(4):
            for col in range(4):
                K[dofs[r], dofs[col]] += ke[r, col]

    F = np.zeros(n_dof)
    total = (dead_kN + live_kN)*1000
    for node in [10,11,12,13,14,15,16]:
        F[2*(node-1)+1] -= total

    fixed = [0, 1, 17]
    free  = [i for i in range(n_dof) if i not in fixed]
    U = np.zeros(n_dof)
    U_f = np.linalg.solve(K[np.ix_(free,free)], F[free])
    for i,d in enumerate(free):
        U[d] = U_f[i]

    forces = {}
    for mem_id, ni, nj in member_list:
        xi,yi = nodes[ni]
        xj,yj = nodes[nj]
        L = np.sqrt((xj-xi)**2 + (yj-yi)**2)
        c,s = (xj-xi)/L, (yj-yi)/L
        u = [U[2*(ni-1)], U[2*(ni-1)+1], U[2*(nj-1)], U[2*(nj-1)+1]]
        forces[mem_id] = (E*A_m2/L) * np.dot([-c,-s,c,s], u)
    return forces

areas           = [1200e-6, 1500e-6, 1903e-6, 2500e-6, 3000e-6]
yield_strengths = [250, 345, 410]
load_cases      = [
    (80,  0,   'LC_LightDead'),
    (100, 0,   'LC1_DeadOnly'),
    (100, 50,  'LC_LightLive'),
    (100, 75,  'LC2_FullLoad'),
    (100, 112, 'LC3_Overload'),
    (120, 75,  'LC_HeavyDead'),
    (120, 112, 'LC_Extreme'),
    (150, 150, 'LC_Critical'),
]

all_results = []
total = len(areas)*len(yield_strengths)*len(load_cases)
count = 0
print(f"Running {total} combinations...")

for A in areas:
    for Fy in yield_strengths:
        for dead, live, lc_name in load_cases:
            forces = solve_truss(dead, live, A)
            for mem_id, ni, nj in member_list:
                force        = forces[mem_id]
                stress_mpa   = (force/A)/1e6
                stress_ratio = abs(stress_mpa)/Fy
                xi,yi = nodes[ni]
                xj,yj = nodes[nj]
                length = round(np.sqrt((xj-xi)**2+(yj-yi)**2),3)

                if stress_ratio < 0.6:
                    condition = 'Safe'
                elif stress_ratio < 0.8:
                    condition = 'At-Risk'
                else:
                    condition = 'Critical'

                all_results.append({
                    'LoadCase':          lc_name,
                    'MemberID':          mem_id,
                    'MemberType':        member_types[mem_id],
                    'Length_m':          length,
                    'Area_mm2':          round(A*1e6,1),
                    'YieldStrength_MPa': Fy,
                    'DeadLoad_kN':       dead,
                    'LiveLoad_kN':       live,
                    'AxialForce_kN':     round(force,3),
                    'Stress_MPa':        round(stress_mpa,3),
                    'StressRatio':       round(stress_ratio,4),
                    'Condition':         condition
                })
            count += 1
            if count % 20 == 0:
                print(f"  Progress: {count}/{total} done...")

os.makedirs('data', exist_ok=True)
df = pd.DataFrame(all_results)
df.to_csv('data/bridge_database.csv', index=False)

print(f"\n✅ Done!")
print(f"   Rows    : {len(df)}")
print(f"   Columns : {len(df.columns)}")
print(f"\nCondition split:")
print(df['Condition'].value_counts())
print(f"\nSample:")
print(df.head())
print(f"\n✅ Saved to data/bridge_database.csv")
