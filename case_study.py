import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pickle
import os

# ══════════════════════════════════════════════════════
# CASE STUDY:
# 50-Year Deterioration Simulation of a Steel Pratt Truss Bridge
#
# Geometry : 24m span | 4m height | 8 panels
# Type     : Determinate Pratt Truss (29 members, 16 nodes)
#
# IS Code References:
# IS 2062:2011  — E=200 GPa constant | Fy=250 MPa Grade E250
# IS 9077:1979  — Corrosion C3 Urban: 0.003-0.005/yr area loss
# IRC 6:2017    — Dead=100kN, Live=75kN per panel (Class A)
# IS 800:2007   — FOS: Safe>1.67 | At-Risk 1.25-1.67 | Critical<1.25
#
# Corrosion Chemistry (IS 9077 basis):
# Fe → Fe²⁺ + 2e⁻           (anodic dissolution)
# O₂ + 2H₂O + 4e⁻ → 4OH⁻   (cathodic reaction)
# 4Fe + 3O₂ + 2H₂O → 2Fe₂O₃·H₂O (rust formation)
#
# Note: Representative structural prototype
# ══════════════════════════════════════════════════════

print("="*60)
print("  CASE STUDY: 50-Year Deterioration Simulation")
print("  Steel Pratt Truss Bridge — ML-Assisted SHM")
print("="*60)
print("  Geometry  : 24m span | 4m height | 8 panels")
print("  Members   : 29 | Nodes: 16")
print("  IS 2062   : Grade E250 | Fy=250 MPa | E=200 GPa (const)")
print("  IS 9077   : C3 Urban corrosion environment")
print("  IRC 6     : Class A loading")
print("  IS 800    : FOS based condition assessment")
print("="*60)

# ── LOAD MODEL ──
with open('models/bridge_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Must match ml_model.py training features exactly
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
A_BASE  = 1903   # mm² — ISA 100×100×10 (IS 808:1989)
Fy      = 250    # MPa — IS 2062:2011 Grade E250

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

# ── IS 9077 CORROSION RATES (C3 Urban) ──
# area loss/year = 2 × thickness_loss_mm / section_thickness_mm
# ISA 100×100×10: section_thickness = 10mm
corrosion_rates = {
    'Bottom Chord': 0.0050,  # 0.025mm/yr — most exposed to moisture
    'Top Chord':    0.0030,  # 0.015mm/yr — sheltered
    'End Post':     0.0050,  # 0.025mm/yr — end zone exposure
    'Vertical':     0.0040,  # 0.020mm/yr — moderate exposure
    'Diagonal':     0.0040,  # 0.020mm/yr — moderate exposure
}

# ── DETERIORATION SIMULATOR ──
def simulate_year(age_years, dead_kN, live_kN):
    """
    Simulate bridge condition at given age.
    IS 9077: time-dependent corrosion per member type
    IS 2062: E = 200 GPa constant (not modeled as variable)
    IS 800:  FOS condition labeling
    """
    rows   = []
    cf_map = {}

    for mem_id, ni, nj in member_list:
        xi,yi = nodes[ni]; xj,yj = nodes[nj]
        length = np.sqrt((xj-xi)**2 + (yj-yi)**2)
        mtype  = member_types[mem_id]

        # IS 9077 C3: proportional variation ±10%
        base_loss = corrosion_rates[mtype] * age_years
        variation = base_loss * np.random.uniform(-0.1, 0.1) \
                    if age_years > 0 else 0
        cf = float(np.clip(1.0 - base_loss + variation,
                           0.60, 1.0))
        cf_map[mem_id] = cf

        rows.append({
            'MemberID':           mem_id,
            'MemberType_enc':     le.transform([mtype])[0],
            'Length_m':           round(length, 3),
            'Area_original_mm2':  A_BASE,
            'Area_effective_mm2': round(A_BASE * cf, 3),
            'CorrosionFactor':    round(cf, 4),
            # E removed — IS 2062 constant, not a feature
            'YieldStrength_MPa':  Fy,
            'DeadLoad_kN':        dead_kN,
            'LiveLoad_kN':        live_kN,
        })

    X_new      = pd.DataFrame(rows)[ML_FEATURES]
    preds      = model.predict(X_new)
    probas     = model.predict_proba(X_new)
    crit_idx   = list(model.classes_).index('Critical')
    crit_probs = probas[:, crit_idx]

    return preds, crit_probs, cf_map

# ── 5 TIME STAGES (IRC 6:2017 gradual load increase) ──
stages = [
    (0,   80,  0,   'Year 0  — New Bridge'),
    (10,  85,  20,  'Year 10 — Early Service'),
    (20,  90,  40,  'Year 20 — Aging Begins'),
    (35,  100, 60,  'Year 35 — Significant Deterioration'),
    (50,  120, 112, 'Year 50 — Critical Condition'),
]

print("\n── IS 800:2007 FOS Based Predictions ──\n")

stage_results = []
for age, dead, live, label in stages:
    preds, crit_probs, cf_map = simulate_year(age, dead, live)

    n_safe = int((preds=='Safe').sum())
    n_risk = int((preds=='At-Risk').sum())
    n_crit = int((preds=='Critical').sum())
    R      = round(1 - n_crit / N_TOTAL, 4)
    avg_corr = (1 - np.mean(list(cf_map.values()))) * 100

    crit_ratio = n_crit / N_TOTAL
    if crit_ratio > 0.40:
        warning = '🔴 SYSTEM CRITICAL'
    elif crit_ratio > 0.20:
        warning = '🟡 SYSTEM WARNING'
    else:
        warning = '🟢 SYSTEM SAFE'

    print(f"  {label}")
    print(f"  IS 9077 Avg Corrosion : {avg_corr:.1f}%")
    print(f"  IRC 6 Loads           : Dead={dead}kN Live={live}kN")
    print(f"  IS 800 FOS Conditions : Safe={n_safe} "
          f"At-Risk={n_risk} Critical={n_crit}")
    print(f"  Reliability R(t)      : {R} | {warning}")
    print()

    stage_results.append({
        'age': age, 'label': label,
        'dead': dead, 'live': live,
        'preds': preds, 'crit_probs': crit_probs,
        'cf_map': cf_map,
        'n_safe': n_safe, 'n_risk': n_risk, 'n_crit': n_crit,
        'R': R, 'avg_corr': avg_corr,
    })

# ── HEATMAPS ──
print("Generating member-level heatmaps...")
cmap = plt.cm.RdYlGn_r
norm = Normalize(vmin=0, vmax=1)

for s in stage_results:
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_facecolor('#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')

    for idx,(mem_id,ni,nj) in enumerate(member_list):
        x1,y1 = nodes[ni]; x2,y2 = nodes[nj]
        color = cmap(norm(s['crit_probs'][idx]))
        lw    = 4.5 if member_types[mem_id] in \
                ['Bottom Chord','Top Chord','End Post'] else 2.5
        ax.plot([x1,x2],[y1,y2], color=color,
                linewidth=lw, solid_capstyle='round')
        mx,my = (x1+x2)/2,(y1+y2)/2
        cf_pct = int((1-s['cf_map'][mem_id])*100)
        ax.text(mx, my+0.18, f'M{mem_id}\n{cf_pct}%c',
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

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Critical Failure Probability (ML)',
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
    ]
    ax.legend(handles=patches, loc='upper right',
              facecolor='#2d2d44', labelcolor='white',
              fontsize=8, title='ML Prediction',
              title_fontsize=8)

    ax.set_title(
        f'Pratt Truss — {s["label"]}\n'
        f'IRC 6: Dead={s["dead"]}kN Live={s["live"]}kN  |  '
        f'IS 9077 C3: {s["avg_corr"]:.1f}% avg loss  |  '
        f'IS 2062 E=200GPa\n'
        f'IS 800: Safe={s["n_safe"]} '
        f'At-Risk={s["n_risk"]} Critical={s["n_crit"]}  |  '
        f'R(t)={s["R"]}',
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
    fname = f'outputs/casestudy_year{s["age"]}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ {fname}")

# ── RELIABILITY CURVE ──
print("\nGenerating Reliability Index curve...")
ages    = [s['age']     for s in stage_results]
R_vals  = [s['R']       for s in stage_results]
n_crits = [s['n_crit']  for s in stage_results]
n_risks = [s['n_risk']  for s in stage_results]
n_safes = [s['n_safe']  for s in stage_results]
corrs   = [s['avg_corr'] for s in stage_results]

fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,13))
fig.patch.set_facecolor('#1a1a2e')

# R(t)
ax1.set_facecolor('#1a1a2e')
ax1.plot(ages, R_vals, 'o-', color='#00aaff',
         linewidth=3, markersize=10, label='R(t)')
ax1.fill_between(ages, R_vals, alpha=0.15, color='#00aaff')
ax1.axhline(y=0.80, color='#ffdd00', linestyle='--',
            alpha=0.8, label='Warning R=0.80')
ax1.axhline(y=0.60, color='#ff4444', linestyle='--',
            alpha=0.8, label='Critical R=0.60')
ax1.fill_between([0,50],0.60,0.80,color='yellow',alpha=0.05)
ax1.fill_between([0,50],0,0.60,color='red',alpha=0.05)
ax1.set_ylabel('Reliability Index R(t)', color='white',fontsize=11)
ax1.set_title(
    '50-Year Life-Cycle Reliability — Steel Pratt Truss\n'
    'IS 9077 | IS 2062 | IRC 6 | IS 800 | ML-Assisted SHM',
    color='white', fontsize=12)
ax1.legend(facecolor='#2d2d44',labelcolor='white',fontsize=9)
ax1.tick_params(colors='white')
ax1.set_ylim(0,1.05); ax1.set_xticks(ages)
ax1.grid(axis='y',color='#333355',linestyle='--',alpha=0.5)
for sp in ax1.spines.values(): sp.set_edgecolor('#333355')

# Member counts
ax2.set_facecolor('#1a1a2e')
ax2.plot(ages,n_crits,'o-',color='#ff4444',linewidth=2.5,
         markersize=8,label='Critical (IS 800: FOS<1.25)')
ax2.plot(ages,n_risks,'s-',color='#ffdd00',linewidth=2.5,
         markersize=8,label='At-Risk (FOS 1.25-1.67)')
ax2.plot(ages,n_safes,'^-',color='#44ff88',linewidth=2.5,
         markersize=8,label='Safe (FOS>1.67)')
ax2.axhline(y=6, color='#ffdd00',linestyle=':',alpha=0.7,
            label='20% → System Warning')
ax2.axhline(y=12,color='#ff4444',linestyle=':',alpha=0.7,
            label='40% → System Critical')
ax2.set_ylabel('Number of Members',color='white',fontsize=11)
ax2.set_title('IS 800:2007 FOS Based Member Conditions',
              color='white',fontsize=11)
ax2.legend(facecolor='#2d2d44',labelcolor='white',fontsize=9)
ax2.tick_params(colors='white')
ax2.set_ylim(0,N_TOTAL+2); ax2.set_xticks(ages)
ax2.grid(axis='y',color='#333355',linestyle='--',alpha=0.5)
for sp in ax2.spines.values(): sp.set_edgecolor('#333355')

# Corrosion
ax3.set_facecolor('#1a1a2e')
ax3.plot(ages,corrs,'D-',color='#ff8844',linewidth=2.5,
         markersize=8,label='IS 9077 C3 Urban Avg Corrosion')
ax3.fill_between(ages,corrs,alpha=0.15,color='#ff8844')
ax3.set_xlabel('Bridge Age (Years)',color='white',fontsize=11)
ax3.set_ylabel('Average Corrosion Loss (%)',
               color='white',fontsize=11)
ax3.set_title('IS 9077 C3 Urban Corrosion Progression',
              color='white',fontsize=11)
ax3.legend(facecolor='#2d2d44',labelcolor='white',fontsize=9)
ax3.tick_params(colors='white')
ax3.set_xticks(ages)
ax3.grid(axis='y',color='#333355',linestyle='--',alpha=0.5)
for sp in ax3.spines.values(): sp.set_edgecolor('#333355')

plt.tight_layout()
plt.savefig('outputs/casestudy_reliability.png', dpi=150,
            bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("  ✅ outputs/casestudy_reliability.png")

# ── SUMMARY ──
print(f"\n{'='*60}")
print(f"  LIFE-CYCLE SUMMARY — IS Code Compliant")
print(f"{'='*60}")
for s in stage_results:
    cr = s['n_crit']/N_TOTAL
    status = ('🔴 SYSTEM CRITICAL' if cr>0.40 else
              '🟡 SYSTEM WARNING'  if cr>0.20 else
              '🟢 SYSTEM SAFE')
    print(f"  {s['label']:<38} "
          f"R={s['R']:.2f}  "
          f"Corr:{s['avg_corr']:.1f}%  {status}")

print(f"""
  IS Code Compliance Summary:
  → IS 2062:2011  E=200 GPa constant (Cl.2.2.1)
  → IS 9077:1979  C3 Urban corrosion rates
  → IRC 6:2017    Class A loading
  → IS 800:2007   FOS thresholds (1.25/1.67)
  → IS 808:1989   ISA 100×100×10 section

  Corrosion Chemistry (IS 9077 basis):
  → Fe + O₂ + H₂O → Fe₂O₃·H₂O (electrochemical)
  → Area loss modeled as linear with IS 9077 C3 rates
  → Bottom chord most vulnerable (moisture accumulation)

🎉 IS Code compliant case study complete!
   Next phase: System-level failure probability
""")
```

---

## Commit Messages:
```
truss_model: added IS code references and documentation
visualize: removed E parameter, IS code aligned, clean Pratt scenarios
case_study: removed E degradation, IS 9077 rates, IS 800 FOS, chemistry comments
