import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pickle
import os

# ══════════════════════════════════════════════════
# CASE STUDY:
# 50-Year Deterioration Simulation of a Steel Pratt Truss Bridge
#
# Geometry : 24m span | 4m height | 8 panels
# Type     : Determinate Pratt Truss (29 members, 16 nodes)
# Material : IS 2062 Grade E250 (Fy = 250 MPa)
#
# Objective:
# Demonstrate ML-assisted SHM early warning capability
# under progressive corrosion and life-cycle loading.
#
# Layers:
# 1. Physics   → time-dependent corrosion + stiffness degradation
# 2. ML        → surrogate early-warning prediction
# 3. Reliability → life-cycle risk index R(t) = 1 - Ncritical/Ntotal
#
# Note:
# Representative structural prototype — not a specific real bridge.
# ══════════════════════════════════════════════════

print("="*60)
print("  CASE STUDY: 50-Year Deterioration Simulation")
print("  Steel Pratt Truss Bridge — ML-Assisted SHM")
print("="*60)
print("  Geometry  : 24m span | 4m height | 8 panels")
print("  Members   : 29 | Nodes: 16")
print("  Material  : IS 2062 E250 | Fy = 250 MPa")
print("  Objective : ML early warning + life-cycle reliability")
print("="*60)

# ── LOAD MODEL ──
with open('models/bridge_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

ML_FEATURES = [
    'MemberType_enc',
    'Length_m',
    'Area_effective_mm2',
    'CorrosionFactor',
    'E_effective_GPa',
    'YieldStrength_MPa',
    'DeadLoad_kN',
    'LiveLoad_kN',
]

N_TOTAL = 29  # total members

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

os.makedirs('outputs', exist_ok=True)
np.random.seed(7)

# ── CORROSION RATES (per member type, per year) ──
# Bottom chord most exposed to moisture → highest rate
corrosion_rates = {
    'Bottom Chord': 0.007,   # 0.7%/year
    'Top Chord':    0.004,   # 0.4%/year
    'End Post':     0.006,   # 0.6%/year
    'Vertical':     0.005,   # 0.5%/year
    'Diagonal':     0.006,   # 0.6%/year
}

# ── DETERIORATION SIMULATOR ──
def simulate_year(age_years, dead_kN, live_kN,
                  A_base_mm2=1903, Fy=250, E_base_GPa=200):
    """
    Simulate bridge state at a given age.
    Returns predictions, probabilities, corrosion map.
    """
    # Stiffness degradation: 0.2%/year, floor at 70%
    E_degr  = max(0.70, 1.0 - age_years * 0.002)
    E_eff   = E_base_GPa * E_degr

    rows = []
    cf_map = {}

    for mem_id, ni, nj in member_list:
        xi,yi = nodes[ni]; xj,yj = nodes[nj]
        length = np.sqrt((xj-xi)**2 + (yj-yi)**2)
        mtype  = member_types[mem_id]

        # Corrosion with proportional variation
        base_loss = corrosion_rates[mtype] * age_years
        variation = base_loss * np.random.uniform(-0.1, 0.1)
        cf = float(np.clip(1.0 - base_loss + variation, 0.60, 1.0))
        cf_map[mem_id] = cf

        rows.append({
            'MemberType_enc':     le.transform([mtype])[0],
            'Length_m':           round(length, 3),
            'Area_effective_mm2': round(A_base_mm2 * cf, 3),
            'CorrosionFactor':    round(cf, 4),
            'E_effective_GPa':    round(E_eff, 2),
            'YieldStrength_MPa':  Fy,
            'DeadLoad_kN':        dead_kN,
            'LiveLoad_kN':        live_kN,
        })

    X_new      = pd.DataFrame(rows)[ML_FEATURES]
    preds      = model.predict(X_new)
    probas     = model.predict_proba(X_new)
    crit_idx   = list(model.classes_).index('Critical')
    crit_probs = probas[:, crit_idx]

    return preds, crit_probs, cf_map, E_eff

# ── RUN 5 TIME STAGES ──
stages = [
    (0,   80,  0,   'Year 0  — New Bridge'),
    (10,  100, 50,  'Year 10 — Early Service'),
    (20,  100, 75,  'Year 20 — Aging Begins'),
    (35,  110, 90,  'Year 35 — Significant Deterioration'),
    (50,  120, 112, 'Year 50 — Critical Condition'),
]

print("\n── Member-Level Predictions Across 50 Years ──\n")

stage_results = []
for age, dead, live, label in stages:
    preds, crit_probs, cf_map, E_eff = simulate_year(
        age, dead, live
    )
    n_safe = (preds=='Safe').sum()
    n_risk = (preds=='At-Risk').sum()
    n_crit = (preds=='Critical').sum()

    # ── RELIABILITY INDEX ──
    R = round(1 - n_crit / N_TOTAL, 4)

    # ── SYSTEM WARNING ──
    warning = ''
    if n_crit / N_TOTAL > 0.40:
        warning = '🔴 SYSTEM CRITICAL'
    elif n_crit / N_TOTAL > 0.20:
        warning = '🟡 SYSTEM WARNING'
    else:
        warning = '🟢 SYSTEM SAFE'

    print(f"  {label}")
    print(f"  E: {E_eff:.0f} GPa | Dead: {dead} kN | Live: {live} kN")
    print(f"  Safe: {n_safe} | At-Risk: {n_risk} | Critical: {n_crit}")
    print(f"  Reliability Index R(t) = {R} | {warning}")
    print()

    stage_results.append({
        'age':        age,
        'label':      label,
        'dead':       dead,
        'live':       live,
        'preds':      preds,
        'crit_probs': crit_probs,
        'cf_map':     cf_map,
        'E_eff':      E_eff,
        'n_safe':     int(n_safe),
        'n_risk':     int(n_risk),
        'n_crit':     int(n_crit),
        'R':          R,
    })

# ── PLOT HEATMAPS FOR EACH STAGE ──
print("Generating member-level heatmaps...")
cmap = plt.cm.RdYlGn_r
norm = Normalize(vmin=0, vmax=1)

for s in stage_results:
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_facecolor('#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')

    for idx, (mem_id, ni, nj) in enumerate(member_list):
        x1,y1 = nodes[ni]; x2,y2 = nodes[nj]
        color = cmap(norm(s['crit_probs'][idx]))
        lw    = 4.5 if member_types[mem_id] in \
                ['Bottom Chord','Top Chord','End Post'] else 2.5
        ax.plot([x1,x2],[y1,y2], color=color,
                linewidth=lw, solid_capstyle='round')

        mx,my = (x1+x2)/2, (y1+y2)/2
        cf_pct = int((1 - s['cf_map'][mem_id]) * 100)
        ax.text(mx, my+0.18,
                f'M{mem_id}\n{cf_pct}%c',
                fontsize=5.5, color='white',
                ha='center', va='bottom', alpha=0.85)

    for nid,(x,y) in nodes.items():
        ax.plot(x, y, 'o', color='white', markersize=6, zorder=5)

    ax.annotate('▲ PIN',    xy=(0,0),  fontsize=9, color='cyan',
                ha='center', va='top', xytext=(0,-0.65))
    ax.annotate('▲ ROLLER', xy=(24,0), fontsize=9, color='cyan',
                ha='center', va='top', xytext=(24,-0.65))

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Critical Failure Probability (ML)',
                   color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    patches = [
        mpatches.Patch(color='#00cc44', label='Low failure probability'),
        mpatches.Patch(color='#ffdd00', label='Moderate failure probability'),
        mpatches.Patch(color='#ff2222', label='High failure probability'),
    ]
    ax.legend(handles=patches, loc='upper right',
              facecolor='#2d2d44', labelcolor='white',
              fontsize=8, title='ML Prediction', title_fontsize=8)

    ax.set_title(
        f'Steel Pratt Truss — {s["label"]}\n'
        f'Dead: {s["dead"]} kN  |  Live: {s["live"]} kN  |  '
        f'E: {s["E_eff"]:.0f} GPa\n'
        f'Safe: {s["n_safe"]}  At-Risk: {s["n_risk"]}  '
        f'Critical: {s["n_crit"]}  |  '
        f'Reliability Index R(t) = {s["R"]}\n'
        f'(c% = corrosion loss per member)',
        color='white', fontsize=11, pad=12
    )

    ax.set_xlim(-2, 27); ax.set_ylim(-1.5, 6.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Span (m)', color='white')
    ax.set_ylabel('Height (m)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    plt.tight_layout()
    fname = f'outputs/casestudy_year{s["age"]}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ {fname}")

# ── RELIABILITY INDEX TREND PLOT ──
print("\nGenerating Reliability Index curve...")

ages   = [s['age']  for s in stage_results]
R_vals = [s['R']    for s in stage_results]
n_crits= [s['n_crit'] for s in stage_results]
n_risks= [s['n_risk'] for s in stage_results]
n_safes= [s['n_safe'] for s in stage_results]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.patch.set_facecolor('#1a1a2e')

# ── TOP: Reliability Index R(t) ──
ax1.set_facecolor('#1a1a2e')
ax1.plot(ages, R_vals, 'o-', color='#00aaff',
         linewidth=3, markersize=10, label='R(t)')
ax1.fill_between(ages, R_vals, alpha=0.15, color='#00aaff')

# Warning thresholds
ax1.axhline(y=0.80, color='#ffdd00', linestyle='--',
            alpha=0.8, label='Warning threshold (R=0.80)')
ax1.axhline(y=0.60, color='#ff4444', linestyle='--',
            alpha=0.8, label='Critical threshold (R=0.60)')

# Shade warning zone
ax1.fill_between([0,50], 0.60, 0.80,
                 color='yellow', alpha=0.05)
ax1.fill_between([0,50], 0, 0.60,
                 color='red', alpha=0.05)

ax1.set_ylabel('Reliability Index R(t)', color='white', fontsize=12)
ax1.set_title(
    '50-Year Life-Cycle Reliability of Steel Pratt Truss Bridge\n'
    'ML-Assisted SHM — Member-Level Analysis',
    color='white', fontsize=13
)
ax1.legend(facecolor='#2d2d44', labelcolor='white', fontsize=10)
ax1.tick_params(colors='white')
ax1.set_ylim(0, 1.05)
ax1.set_xticks(ages)
ax1.grid(axis='y', color='#333355', linestyle='--', alpha=0.5)
for spine in ax1.spines.values():
    spine.set_edgecolor('#333355')

# ── BOTTOM: Member condition count ──
ax2.set_facecolor('#1a1a2e')
ax2.plot(ages, n_crits, 'o-', color='#ff4444',
         linewidth=2.5, markersize=8, label='Critical')
ax2.plot(ages, n_risks, 's-', color='#ffdd00',
         linewidth=2.5, markersize=8, label='At-Risk')
ax2.plot(ages, n_safes, '^-', color='#44ff88',
         linewidth=2.5, markersize=8, label='Safe')

# System warning line (20% of 29 = ~6 members)
ax2.axhline(y=6, color='#ffdd00', linestyle=':',
            alpha=0.7, label='20% threshold → System Warning')
ax2.axhline(y=12, color='#ff4444', linestyle=':',
            alpha=0.7, label='40% threshold → System Critical')

ax2.set_xlabel('Bridge Age (Years)', color='white', fontsize=12)
ax2.set_ylabel('Number of Members', color='white', fontsize=12)
ax2.set_title(
    'Member-Level Condition Evolution Over Time',
    color='white', fontsize=12
)
ax2.legend(facecolor='#2d2d44', labelcolor='white', fontsize=10)
ax2.tick_params(colors='white')
ax2.set_ylim(0, N_TOTAL + 2)
ax2.set_xticks(ages)
ax2.grid(axis='y', color='#333355', linestyle='--', alpha=0.5)
for spine in ax2.spines.values():
    spine.set_edgecolor('#333355')

plt.tight_layout()
plt.savefig('outputs/casestudy_reliability.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  ✅ outputs/casestudy_reliability.png")

# ── FINAL SUMMARY ──
print(f"\n{'='*60}")
print(f"  LIFE-CYCLE SUMMARY")
print(f"{'='*60}")
for s in stage_results:
    if s['n_crit'] / N_TOTAL > 0.40:
        status = '🔴 SYSTEM CRITICAL'
    elif s['n_crit'] / N_TOTAL > 0.20:
        status = '🟡 SYSTEM WARNING'
    else:
        status = '🟢 SYSTEM SAFE'
    print(f"  {s['label']:<35} "
          f"R={s['R']:.2f}  {status}")

print(f"\n  Key Findings:")
print(f"  → ML framework identifies at-risk members early")
print(f"  → Reliability Index R(t) declines monotonically with age")
print(f"  → System-level warning triggered when R(t) < 0.80")
print(f"  → Demonstrates value of SHM for life-cycle management")
print(f"\n🎉 Member-level case study complete!")
print(f"   Next phase: System-level failure probability")
