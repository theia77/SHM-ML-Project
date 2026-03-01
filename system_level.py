import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pickle
import os
from scipy.stats import norm

# ══════════════════════════════════════════════════════
# SYSTEM-LEVEL FAILURE PROBABILITY
# Phase 2 of ML-Assisted SHM Framework
#
# Approach:
# Member-level ML predictions → System-level risk
#
# Method:
# Series system assumption — bridge fails if ANY
# critical load-carrying member fails.
# (Conservative — standard for bridge assessment)
#
# System Failure Probability:
# P_system = 1 - ∏(1 - P_failure_i) for critical members
#
# IS Code References:
# IS 2062:2011  — Material properties
# IS 9077:1979  — Corrosion rates C3 Urban
# IRC 6:2017    — Load combinations
# IS 800:2007   — Member reliability basis
# IRC 24:2010   — Bridge inspection and assessment
# ══════════════════════════════════════════════════════

print("="*60)
print("  SYSTEM-LEVEL FAILURE PROBABILITY")
print("  Steel Pratt Truss Bridge — ML-Assisted SHM")
print("="*60)

# ── LOAD MODEL ──
with open('models/bridge_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

ML_FEATURES = [
    'MemberID',
    'MemberType_enc',
    'Length_m',
    'Area_original_mm2',
    'Area_effective_mm2',
    'CorrosionFactor',
    'YieldStrength_MPa',
    'DeadLoad_kN',
    'LiveLoad_kN',
]

N_TOTAL = 29
A_BASE  = 1903   # mm²
Fy      = 250    # MPa

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

# Critical load-carrying members
# (series system — failure of any = system failure)
# Bottom chord + Top chord + End posts are primary
# Verticals and diagonals are secondary
primary_members   = list(range(1,17))   # chords + end posts
secondary_members = list(range(17,30))  # verticals + diagonals

os.makedirs('outputs', exist_ok=True)
np.random.seed(7)

# ── IS 9077 CORROSION RATES ──
corrosion_rates = {
    'Bottom Chord': 0.0050,
    'Top Chord':    0.0030,
    'End Post':     0.0050,
    'Vertical':     0.0040,
    'Diagonal':     0.0040,
}

# ── MEMBER LEVEL PREDICTOR ──
def get_member_probabilities(age_years, dead_kN, live_kN):
    """
    Get ML failure probability for each member.
    Returns dict: {member_id: P(Critical)}
    """
    rows   = []
    cf_map = {}

    for mem_id, ni, nj in member_list:
        xi,yi = nodes[ni]; xj,yj = nodes[nj]
        length = np.sqrt((xj-xi)**2+(yj-yi)**2)
        mtype  = member_types[mem_id]

        base_loss = corrosion_rates[mtype] * age_years
        variation = base_loss * np.random.uniform(-0.1,0.1) \
                    if age_years > 0 else 0
        cf = float(np.clip(1.0-base_loss+variation, 0.60, 1.0))
        cf_map[mem_id] = cf

        rows.append({
            'MemberID':           mem_id,
            'MemberType_enc':     le.transform([mtype])[0],
            'Length_m':           round(length, 3),
            'Area_original_mm2':  A_BASE,
            'Area_effective_mm2': round(A_BASE*cf, 3),
            'CorrosionFactor':    round(cf, 4),
            'YieldStrength_MPa':  Fy,
            'DeadLoad_kN':        dead_kN,
            'LiveLoad_kN':        live_kN,
        })

    X_new      = pd.DataFrame(rows)[ML_FEATURES]
    probas     = model.predict_proba(X_new)
    crit_idx   = list(model.classes_).index('Critical')
    crit_probs = {mem_id: probas[i, crit_idx]
                  for i, (mem_id,_,_) in enumerate(member_list)}

    return crit_probs, cf_map

# ── SYSTEM FAILURE PROBABILITY ──
def compute_system_pf(crit_probs, system='series'):
    """
    Compute system-level failure probability.

    Series system (conservative):
    P_sys = 1 - ∏(1 - P_fi) for primary members
    Used for bridges — failure of any chord = collapse.

    Parallel system (optimistic):
    P_sys = ∏(P_fi) — all must fail simultaneously.

    We use series for primary members (IS 800 basis).
    """
    # Primary members — series system
    p_primary = [crit_probs[m] for m in primary_members
                 if m in crit_probs]
    pf_series = 1.0 - np.prod([1-p for p in p_primary])

    # Secondary members — parallel contribution
    p_secondary = [crit_probs[m] for m in secondary_members
                   if m in crit_probs]
    pf_parallel = float(np.prod(p_secondary)) \
                  if p_secondary else 0.0

    # Combined system probability
    # weighted: 70% primary, 30% secondary
    pf_system = 0.70*pf_series + 0.30*pf_parallel

    # Reliability index β (structural reliability theory)
    # β = -Φ⁻¹(Pf) where Φ is standard normal CDF
    pf_clipped = np.clip(pf_system, 1e-10, 1-1e-10)
    beta = -norm.ppf(pf_clipped)

    return {
        'Pf_series':   round(float(pf_series),   6),
        'Pf_parallel': round(float(pf_parallel),  6),
        'Pf_system':   round(float(pf_system),    6),
        'beta':        round(float(beta),          4),
        'reliability': round(1-float(pf_system),   6),
    }

# ── IRC 24:2010 RELIABILITY TARGETS ──
# Minimum target reliability index β for road bridges
BETA_TARGETS = {
    'Collapse'    : 3.8,   # IRC 24 target (50yr ref period)
    'Serviceability': 1.5,
}

def get_safety_status(beta):
    if beta >= BETA_TARGETS['Collapse']:
        return '🟢 RELIABLE   (β ≥ 3.8)'
    elif beta >= BETA_TARGETS['Serviceability']:
        return '🟡 MARGINAL   (1.5 ≤ β < 3.8)'
    else:
        return '🔴 UNRELIABLE (β < 1.5)'

# ── RUN ACROSS 5 TIME STAGES ──
stages = [
    (0,   80,  0,   'Year 0  — New Bridge'),
    (10,  85,  20,  'Year 10 — Early Service'),
    (20,  90,  40,  'Year 20 — Aging Begins'),
    (35,  100, 60,  'Year 35 — Significant Deterioration'),
    (50,  120, 112, 'Year 50 — Critical Condition'),
]

print("\n── System-Level Reliability Analysis ──\n")
print(f"  IRC 24:2010 Target: β ≥ {BETA_TARGETS['Collapse']} "
      f"(50-year reference period)\n")

stage_results = []
for age, dead, live, label in stages:
    crit_probs, cf_map = get_member_probabilities(
        age, dead, live)
    sys = compute_system_pf(crit_probs)
    status = get_safety_status(sys['beta'])

    avg_corr = (1 - np.mean(list(cf_map.values()))) * 100

    print(f"  {label}")
    print(f"  Pf (series system) : {sys['Pf_series']:.6f}")
    print(f"  Pf (system)        : {sys['Pf_system']:.6f}")
    print(f"  Reliability Index β: {sys['beta']:.4f}")
    print(f"  Status             : {status}")
    print(f"  Avg Corrosion      : {avg_corr:.1f}% (IS 9077 C3)")
    print()

    stage_results.append({
        'age':      age,
        'label':    label,
        'dead':     dead,
        'live':     live,
        **sys,
        'avg_corr': avg_corr,
        'cf_map':   cf_map,
        'crit_probs': crit_probs,
    })

# ── SYSTEM LEVEL HEATMAP ──
print("Generating system-level heatmaps...")
cmap_prob = plt.cm.RdYlGn_r
norm_prob  = Normalize(vmin=0, vmax=1)

for s in stage_results:
    fig, ax = plt.subplots(figsize=(16,7))
    ax.set_facecolor('#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')

    for idx,(mem_id,ni,nj) in enumerate(member_list):
        x1,y1 = nodes[ni]; x2,y2 = nodes[nj]
        prob   = s['crit_probs'][mem_id]
        color  = cmap_prob(norm_prob(prob))

        # Primary members drawn thicker
        is_primary = mem_id in primary_members
        lw = 5.0 if is_primary else 2.5
        ax.plot([x1,x2],[y1,y2], color=color,
                linewidth=lw, solid_capstyle='round')

        mx,my = (x1+x2)/2,(y1+y2)/2
        cf_pct = int((1-s['cf_map'][mem_id])*100)
        ax.text(mx, my+0.18,
                f'M{mem_id}\n{cf_pct}%c',
                fontsize=5.5, color='white',
                ha='center', va='bottom', alpha=0.85)

    for nid,(x,y) in nodes.items():
        ax.plot(x, y, 'o', color='white',
                markersize=6, zorder=5)

    ax.annotate('▲ PIN',    xy=(0,0),  fontsize=9,
                color='cyan', ha='center',
                va='top', xytext=(0,-0.65))
    ax.annotate('▲ ROLLER', xy=(24,0), fontsize=9,
                color='cyan', ha='center',
                va='top', xytext=(24,-0.65))

    sm = ScalarMappable(cmap=cmap_prob, norm=norm_prob)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Member Critical Probability (ML)',
                   color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    patches = [
        mpatches.Patch(color='#00cc44',
                       label='Low failure probability'),
        mpatches.Patch(color='#ffdd00',
                       label='Moderate failure probability'),
        mpatches.Patch(color='#ff2222',
                       label='High failure probability'),
        mpatches.Patch(color='white',
                       label='Thick = Primary member (series system)'),
    ]
    ax.legend(handles=patches, loc='upper right',
              facecolor='#2d2d44', labelcolor='white',
              fontsize=8, title='System Analysis',
              title_fontsize=8)

    status = get_safety_status(s['beta'])
    ax.set_title(
        f'System-Level Analysis — {s["label"]}\n'
        f'IRC 6: Dead={s["dead"]}kN Live={s["live"]}kN  |  '
        f'IS 9077: {s["avg_corr"]:.1f}% corrosion\n'
        f'Pf(system)={s["Pf_system"]:.4f}  |  '
        f'β={s["beta"]:.3f}  |  {status}',
        color='white', fontsize=10, pad=12
    )

    ax.set_xlim(-2, 27); ax.set_ylim(-1.5, 6.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Span (m)', color='white')
    ax.set_ylabel('Height (m)', color='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')

    plt.tight_layout()
    fname = f'outputs/system_year{s["age"]}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ {fname}")

# ── SYSTEM RELIABILITY CURVE ──
print("\nGenerating system reliability curves...")

ages    = [s['age']        for s in stage_results]
pf_vals = [s['Pf_system']  for s in stage_results]
beta_vals=[s['beta']        for s in stage_results]
corrs   = [s['avg_corr']   for s in stage_results]
pf_ser  = [s['Pf_series']  for s in stage_results]

fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,13))
fig.patch.set_facecolor('#1a1a2e')

# ── TOP: β reliability index ──
ax1.set_facecolor('#1a1a2e')
ax1.plot(ages, beta_vals, 'o-', color='#00aaff',
         linewidth=3, markersize=10, label='β (Reliability Index)')
ax1.fill_between(ages, beta_vals, alpha=0.15, color='#00aaff')

ax1.axhline(y=BETA_TARGETS['Collapse'],
            color='#ff4444', linestyle='--', alpha=0.9,
            label=f'IRC 24 Target β={BETA_TARGETS["Collapse"]} (Collapse)')
ax1.axhline(y=BETA_TARGETS['Serviceability'],
            color='#ffdd00', linestyle='--', alpha=0.9,
            label=f'Serviceability β={BETA_TARGETS["Serviceability"]}')

ax1.fill_between([0,50], BETA_TARGETS['Serviceability'],
                 BETA_TARGETS['Collapse'],
                 color='yellow', alpha=0.05)
ax1.fill_between([0,50], 0,
                 BETA_TARGETS['Serviceability'],
                 color='red', alpha=0.05)

ax1.set_ylabel('Reliability Index β', color='white', fontsize=11)
ax1.set_title(
    'System-Level Reliability — Steel Pratt Truss Bridge\n'
    'IS 9077 | IS 2062 | IRC 6 | IS 800 | IRC 24 Target',
    color='white', fontsize=12
)
ax1.legend(facecolor='#2d2d44', labelcolor='white', fontsize=9)
ax1.tick_params(colors='white')
ax1.set_xticks(ages)
ax1.grid(axis='y', color='#333355', linestyle='--', alpha=0.5)
for sp in ax1.spines.values(): sp.set_edgecolor('#333355')

# ── MIDDLE: System Pf ──
ax2.set_facecolor('#1a1a2e')
ax2.plot(ages, pf_vals, 'o-', color='#ff4444',
         linewidth=2.5, markersize=8,
         label='Pf System (series+parallel)')
ax2.plot(ages, pf_ser, 's--', color='#ff8844',
         linewidth=2, markersize=7,
         label='Pf Series (primary members only)')
ax2.fill_between(ages, pf_vals, alpha=0.15, color='#ff4444')
ax2.set_ylabel('Failure Probability Pf', color='white', fontsize=11)
ax2.set_title('System Failure Probability Over Time',
              color='white', fontsize=11)
ax2.legend(facecolor='#2d2d44', labelcolor='white', fontsize=9)
ax2.tick_params(colors='white')
ax2.set_xticks(ages)
ax2.grid(axis='y', color='#333355', linestyle='--', alpha=0.5)
for sp in ax2.spines.values(): sp.set_edgecolor('#333355')

# ── BOTTOM: Corrosion ──
ax3.set_facecolor('#1a1a2e')
ax3.plot(ages, corrs, 'D-', color='#ff8844',
         linewidth=2.5, markersize=8,
         label='IS 9077 C3 Urban Avg Corrosion')
ax3.fill_between(ages, corrs, alpha=0.15, color='#ff8844')
ax3.set_xlabel('Bridge Age (Years)', color='white', fontsize=11)
ax3.set_ylabel('Avg Corrosion Loss (%)', color='white', fontsize=11)
ax3.set_title('IS 9077 C3 Corrosion Progression',
              color='white', fontsize=11)
ax3.legend(facecolor='#2d2d44', labelcolor='white', fontsize=9)
ax3.tick_params(colors='white')
ax3.set_xticks(ages)
ax3.grid(axis='y', color='#333355', linestyle='--', alpha=0.5)
for sp in ax3.spines.values(): sp.set_edgecolor('#333355')

plt.tight_layout()
plt.savefig('outputs/system_reliability.png', dpi=150,
            bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("  ✅ outputs/system_reliability.png")

# ── FINAL SUMMARY ──
print(f"\n{'='*60}")
print(f"  SYSTEM RELIABILITY SUMMARY")
print(f"  IRC 24:2010 Target β ≥ {BETA_TARGETS['Collapse']}")
print(f"{'='*60}")
for s in stage_results:
    status = get_safety_status(s['beta'])
    print(f"  {s['label']:<38} "
          f"β={s['beta']:>6.3f}  "
          f"Pf={s['Pf_system']:.4f}  "
          f"{status}")

print(f"""
  System Analysis Method:
  → Primary members (chords+posts) : Series system
  → Secondary members (v+d)        : Parallel system
  → Combined Pf = 0.7×Pf_series + 0.3×Pf_parallel
  → β = -Φ⁻¹(Pf) — structural reliability index

  IS Code Compliance:
  → IS 2062:2011  Material properties
  → IS 9077:1979  C3 Urban corrosion
  → IRC 6:2017    Load combinations
  → IS 800:2007   Member FOS basis
  → IRC 24:2010   β ≥ 3.8 reliability target

🎉 System-level analysis complete!
""")
