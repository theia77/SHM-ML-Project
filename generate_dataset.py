import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── GEOMETRY ──
nodes = {
    1:(0,0),  2:(3,0),  3:(6,0),  4:(9,0),  5:(12,0),
    6:(15,0), 7:(18,0), 8:(21,0), 9:(24,0),
    10:(3,4), 11:(6,4), 12:(9,4), 13:(12,4),
    14:(15,4),15:(18,4),16:(21,4)
}

member_list = [
    (1,1,2),(2,2,3),(3,3,4),(4,4,5),
    (5,5,6),(6,6,7),(7,7,8),(8,8,9),
    (9,10,11),(10,11,12),(11,12,13),
    (12,13,14),(13,14,15),(14,15,16),
    (15,1,10),(16,9,16),
    (17,2,10),(18,3,11),(19,4,12),(20,5,13),
    (21,6,14),(22,7,15),(23,8,16),
    (24,10,3),(25,11,4),(26,12,5),
    (27,13,6),(28,14,7),(29,15,8)
]

member_types = {
    **{i:'Bottom Chord' for i in range(1,9)},
    **{i:'Top Chord'    for i in range(9,15)},
    **{i:'End Post'     for i in range(15,17)},
    **{i:'Vertical'     for i in range(17,24)},
    **{i:'Diagonal'     for i in range(24,30)}
}

# ── SOLVER ──
def solve_truss(dead_N, live_N, areas_dict, E_dict):
    """
    Solve truss with per-member area and E
    (supports degradation and corrosion per member)
    """
    n_dof = len(nodes) * 2
    K = np.zeros((n_dof, n_dof))

    for mem_id, ni, nj in member_list:
        xi,yi = nodes[ni]; xj,yj = nodes[nj]
        L = np.sqrt((xj-xi)**2 + (yj-yi)**2)
        c,s = (xj-xi)/L, (yj-yi)/L
        k = E_dict[mem_id] * areas_dict[mem_id] / L
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
    for node in [10,11,12,13,14,15,16]:
        F[2*(node-1)+1] -= (dead_N + live_N)

    fixed = [0, 1, 17]
    free  = [i for i in range(n_dof) if i not in fixed]
    U = np.zeros(n_dof)
    U_f = np.linalg.solve(K[np.ix_(free,free)], F[free])
    for i,d in enumerate(free):
        U[d] = U_f[i]

    forces = {}
    for mem_id, ni, nj in member_list:
        xi,yi = nodes[ni]; xj,yj = nodes[nj]
        L = np.sqrt((xj-xi)**2 + (yj-yi)**2)
        c,s = (xj-xi)/L, (yj-yi)/L
        u = [U[2*(ni-1)], U[2*(ni-1)+1], U[2*(nj-1)], U[2*(nj-1)+1]]
        forces[mem_id] = (E_dict[mem_id]*areas_dict[mem_id]/L) * np.dot([-c,-s,c,s], u)
    return forces

# ── DATASET PARAMETERS ──
base_areas      = [1200e-6, 1500e-6, 1903e-6, 2500e-6, 3000e-6]
yield_strengths = [250, 345, 410]
load_cases = [
    (80,   0,   'LC_LightDead'),
    (100,  0,   'LC1_DeadOnly'),
    (100,  50,  'LC_LightLive'),
    (100,  75,  'LC2_FullLoad'),
    (100,  112, 'LC3_Overload'),
    (120,  75,  'LC_HeavyDead'),
    (120,  112, 'LC_Extreme'),
    (150,  150, 'LC_Critical'),
]

# Degradation scenarios per combination
N_DEGRADATION_SAMPLES = 3  # 3 random degradation variants per combination

all_results = []
total = len(base_areas) * len(yield_strengths) * len(load_cases) * N_DEGRADATION_SAMPLES
count = 0
print(f"Running {total} combinations x 29 members = {total*29} rows...")
print(f"Includes: corrosion + stiffness degradation + load noise\n")

for A_base in base_areas:
    for Fy in yield_strengths:
        for dead_kN, live_kN, lc_name in load_cases:
            for deg_sample in range(N_DEGRADATION_SAMPLES):

                # ── DEGRADATION ──
                # 1. Corrosion: randomly reduce area per member (0-30%)
                corrosion_factors = {
                    mem_id: np.random.uniform(0.70, 1.0)
                    for mem_id, _, _ in member_list
                }
                areas_dict = {
                    mem_id: A_base * corrosion_factors[mem_id]
                    for mem_id, _, _ in member_list
                }

                # 2. Stiffness degradation: reduce E per member (0-20%)
                E_dict = {
                    mem_id: 200e9 * np.random.uniform(0.80, 1.0)
                    for mem_id, _, _ in member_list
                }

                # 3. Load noise: ±10% random variation
                dead_noise = dead_kN * np.random.uniform(0.90, 1.10)
                live_noise = live_kN * np.random.uniform(0.90, 1.10)

                # Convert to Newtons
                dead_N = dead_noise * 1000
                live_N = live_noise * 1000

                forces = solve_truss(dead_N, live_N, areas_dict, E_dict)

                for mem_id, ni, nj in member_list:
                    force        = forces[mem_id]           # N
                    A_eff        = areas_dict[mem_id]        # m² after corrosion
                    E_eff        = E_dict[mem_id]            # Pa after degradation
                    corr_factor  = corrosion_factors[mem_id]

                    stress_mpa   = (force / A_eff) / 1e6    # MPa
                    stress_ratio = abs(stress_mpa) / Fy
                    FOS          = Fy / abs(stress_mpa) if stress_mpa != 0 else 999

                    xi,yi = nodes[ni]; xj,yj = nodes[nj]
                    length = round(np.sqrt((xj-xi)**2+(yj-yi)**2), 3)

                    if stress_ratio < 0.6:
                        condition = 'Safe'
                    elif stress_ratio < 0.8:
                        condition = 'At-Risk'
                    else:
                        condition = 'Critical'

                    all_results.append({
                        # Geometry
                        'MemberID':           mem_id,
                        'MemberType':         member_types[mem_id],
                        'Length_m':           length,

                        # Material (degraded)
                        'Area_original_mm2':  round(A_base*1e6, 1),
                        'Area_effective_mm2': round(A_eff*1e6, 3),
                        'CorrosionFactor':    round(corr_factor, 4),
                        'E_effective_GPa':    round(E_eff/1e9, 3),
                        'YieldStrength_MPa':  Fy,

                        # Loads (with noise)
                        'LoadCase':           lc_name,
                        'DeadLoad_kN':        round(dead_noise, 2),
                        'LiveLoad_kN':        round(live_noise, 2),

                        # Results (kept for reference only — NOT used as ML features)
                        'AxialForce_kN':      round(force/1000, 4),
                        'Stress_MPa':         round(stress_mpa, 4),
                        'StressRatio':        round(stress_ratio, 4),
                        'FOS':                round(FOS, 4),

                        # Label
                        'Condition':          condition
                    })

                count += 1
                if count % 60 == 0:
                    print(f"  Progress: {count}/{total} done...")

# ── SAVE ──
os.makedirs('data', exist_ok=True)
df = pd.DataFrame(all_results)
df.to_csv('data/bridge_database.csv', index=False)

print(f"\n✅ Dataset complete!")
print(f"   Rows        : {len(df)}")
print(f"   Columns     : {len(df.columns)}")
print(f"\nCondition split:")
print(df['Condition'].value_counts())
print(f"\nCorrosion factor range: {df['CorrosionFactor'].min():.2f} – {df['CorrosionFactor'].max():.2f}")
print(f"E degradation range   : {df['E_effective_GPa'].min():.1f} – {df['E_effective_GPa'].max():.1f} GPa")
print(f"\nSample:")
print(df.head(3).to_string())
print(f"\n✅ Saved → data/bridge_database.csv")
