import numpy as np
import pandas as pd
import os

# ══════════════════════════════════════════════════════
# IS CODE REFERENCES:
# IS 2062:2011  — E=200 GPa constant | Fy=250/345/410 MPa
# IS 808:1989   — ISA angle sections (area values)
# IS 9077:1979  — Corrosion C3 Urban environment
# IRC 6:2017    — Dead and live load combinations
# IS 800:2007   — FOS based condition labeling
#
# Corrosion Chemistry (IS 9077):
# Fe → Fe²⁺ + 2e⁻             (anodic dissolution)
# O₂ + 2H₂O + 4e⁻ → 4OH⁻     (cathodic reaction)
# 4Fe + 3O₂ + 2H₂O → 2Fe₂O₃·H₂O (rust)
# Area loss/yr = 2×thickness_loss / section_thickness
# ══════════════════════════════════════════════════════

np.random.seed(42)

# ── GEOMETRY ──
nodes = {
    1:(0,0),  2:(3,0),  3:(6,0),  4:(9,0),  5:(12,0),
    6:(15,0), 7:(18,0), 8:(21,0), 9:(24,0),
    10:(3,4), 11:(6,4), 12:(9,4), 13:(12,4),
    14:(15,4),15:(18,4),16:(21,4)
}

member_list = [
    # Bottom chord (8)
    (1,1,2),(2,2,3),(3,3,4),(4,4,5),
    (5,5,6),(6,6,7),(7,7,8),(8,8,9),
    # Top chord (6)
    (9,10,11),(10,11,12),(11,12,13),
    (12,13,14),(13,14,15),(14,15,16),
    # End posts (2)
    (15,1,10),(16,9,16),
    # Verticals (7)
    (17,2,10),(18,3,11),(19,4,12),(20,5,13),
    (21,6,14),(22,7,15),(23,8,16),
    # Diagonals (6)
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

# ── IS 9077:1979 CORROSION RANGES (C3 Urban) ──
# area retention range per member type
# bottom chord worst — moisture accumulates at bottom
corrosion_range = {
    'Bottom Chord': (0.75, 1.00),  # max 25% loss over life
    'Top Chord':    (0.85, 1.00),  # max 15% loss — sheltered
    'End Post':     (0.75, 1.00),  # max 25% loss — end exposure
    'Vertical':     (0.80, 1.00),  # max 20% loss — moderate
    'Diagonal':     (0.80, 1.00),  # max 20% loss — moderate
}

# ── DIRECT STIFFNESS SOLVER ──
# IS 2062:2011 Cl.2.2.1 — E = 200 GPa, constant for all steel grades
E = 200e9  # N/m² — CONSTANT, never changes with age or corrosion

def solve_truss(dead_N, live_N, areas_dict):
    n_dof = len(nodes) * 2
    K = np.zeros((n_dof, n_dof))

    for mem_id, ni, nj in member_list:
        xi,yi = nodes[ni]; xj,yj = nodes[nj]
        L = np.sqrt((xj-xi)**2 + (yj-yi)**2)
        c,s = (xj-xi)/L, (yj-yi)/L
        k = E * areas_dict[mem_id] / L
        ke = k * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])
        dofs = [2*(ni-1), 2*(ni-1)+1,
                2*(nj-1), 2*(nj-1)+1]
        for r in range(4):
            for col in range(4):
                K[dofs[r], dofs[col]] += ke[r, col]

    # Loads on top chord nodes 10-16 (IRC 6:2017)
    F = np.zeros(n_dof)
    # IRC 6:2017 — total load divided equally
    # across 7 panel points (nodes 10-16)
    panel_load = (dead_N + live_N) / 7
    for node in [10,11,12,13,14,15,16]:
        F[2*(node-1)+1] -= panel_load
    # Node 1=Pin | Node 9=Roller (IRC 24:2010)
    fixed = [0, 1, 17]
    free  = [i for i in range(n_dof) if i not in fixed]
    U     = np.zeros(n_dof)
    U_f   = np.linalg.solve(K[np.ix_(free,free)], F[free])
    for i,d in enumerate(free):
        U[d] = U_f[i]

    forces = {}
    for mem_id, ni, nj in member_list:
        xi,yi = nodes[ni]; xj,yj = nodes[nj]
        L = np.sqrt((xj-xi)**2 + (yj-yi)**2)
        c,s = (xj-xi)/L, (yj-yi)/L
        u = [U[2*(ni-1)], U[2*(ni-1)+1],
             U[2*(nj-1)], U[2*(nj-1)+1]]
        forces[mem_id] = (E*areas_dict[mem_id]/L) * \
                          np.dot([-c,-s,c,s], u)
    return forces

# ── DATASET PARAMETERS ──
# IS 808:1989 — standard angle sections
base_areas = [
    1200e-6,   # ISA 75×75×8  — approx 1150 mm²
    1500e-6,   # ISA 90×90×8  — approx 1390 mm²
    1903e-6,   # ISA 100×100×10 — exact IS 808
    2500e-6,   # ISA 125×125×10 — approx 2430 mm²
    3000e-6,   # ISA 150×150×12 — approx 3480 mm²
]

# IS 2062:2011 — steel grades
yield_strengths = [250, 345, 410]  # MPa

# IRC 6:2017 — load combinations
load_cases = [
    (80,   0,   'LC_DeadOnly_Light'),
    (100,  0,   'LC_DeadOnly_Full'),
    (100,  50,  'LC_ServiceLight'),
    (100,  75,  'LC_ServiceFull'),      # IRC Class A
    (100,  112, 'LC_Overload'),         # 1.5×Live
    (120,  75,  'LC_HeavyDead'),
    (120,  112, 'LC_Extreme'),
    (150,  150, 'LC_Critical'),
]

N_DEGRADATION = 3  # variants per combination

all_results = []
total = len(base_areas)*len(yield_strengths)* \
        len(load_cases)*N_DEGRADATION
count = 0

print(f"IS Code Compliant Dataset Generation")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"IS 2062 : E=200 GPa constant | Fy=250/345/410 MPa")
print(f"IS 9077 : C3 Urban corrosion | area loss model")
print(f"IRC 6   : Class A loads | 8 combinations")
print(f"IS 800  : FOS 1.67/1.25 condition thresholds")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"Total   : {total} runs × 29 members = {total*29} rows\n")

for A_base in base_areas:
    for Fy in yield_strengths:
        for dead_kN, live_kN, lc_name in load_cases:
            for _ in range(N_DEGRADATION):

                # IS 9077 C3 — per member corrosion
                areas_dict = {}
                cf_dict    = {}
                for mem_id, _, _ in member_list:
                    mtype = member_types[mem_id]
                    lo, hi = corrosion_range[mtype]
                    cf = np.random.uniform(lo, hi)
                    cf_dict[mem_id]    = cf
                    areas_dict[mem_id] = A_base * cf

                # IRC 6:2017 — ±10% load variation
                dead_N = dead_kN * \
                         np.random.uniform(0.90,1.10) * 1000
                live_N = live_kN * \
                         np.random.uniform(0.90,1.10) * 1000

                forces = solve_truss(dead_N, live_N, areas_dict)

                for mem_id, ni, nj in member_list:
                    force        = forces[mem_id]
                    A_eff        = areas_dict[mem_id]
                    cf           = cf_dict[mem_id]
                    stress_mpa   = (force / A_eff) / 1e6
                    stress_ratio = abs(stress_mpa) / Fy

                    # IS 800:2007 FOS based labels
                    if stress_ratio < 0.60:       # FOS > 1.67
                        condition = 'Safe'
                    elif stress_ratio < 0.80:     # FOS 1.25-1.67
                        condition = 'At-Risk'
                    else:                         # FOS < 1.25
                        condition = 'Critical'

                    xi,yi = nodes[ni]; xj,yj = nodes[nj]
                    length = round(
                        np.sqrt((xj-xi)**2+(yj-yi)**2), 3)

                    all_results.append({
                        # ── ML FEATURES ──
                        'MemberID':           mem_id,
                        'MemberType':         member_types[mem_id],
                        'Length_m':           length,
                        'Area_original_mm2':  round(A_base*1e6,1),
                        'Area_effective_mm2': round(A_eff*1e6,3),
                        'CorrosionFactor':    round(cf,4),
                        # E_effective_GPa REMOVED
                        # IS 2062:2011 — E=200 GPa constant
                        # zero variance → not useful as ML feature
                        'YieldStrength_MPa':  Fy,
                        'DeadLoad_kN':        round(dead_N/1000,3),
                        'LiveLoad_kN':        round(live_N/1000,3),
                        # ── REFERENCE ONLY (not ML features) ──
                        'AxialForce_kN':      round(force/1000,4),
                        'Stress_MPa':         round(stress_mpa,4),
                        'StressRatio':        round(stress_ratio,4),
                        'FOS': round(Fy/abs(stress_mpa),4)
                               if stress_mpa != 0 else 999,
                        # ── LABEL ──
                        'Condition':          condition
                    })

                count += 1
                if count % 60 == 0:
                    print(f"  Progress: {count}/{total}...")

# ── SAVE ──
os.makedirs('data', exist_ok=True)
df = pd.DataFrame(all_results)
df.to_csv('data/bridge_database.csv', index=False)

print(f"\n✅ Dataset complete!")
print(f"   Rows     : {len(df)}")
print(f"   Columns  : {len(df.columns)}")
print(f"\nCondition split (IS 800:2007):")
print(df['Condition'].value_counts())
print(f"\nCorrosion range : {df['CorrosionFactor'].min():.2f}"
      f" – {df['CorrosionFactor'].max():.2f}")
print(f"E (IS 2062)     : 200 GPa constant — not in dataset")
print(f"\n✅ Saved → data/bridge_database.csv")
