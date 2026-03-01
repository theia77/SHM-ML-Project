import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pickle
import os

# ══════════════════════════════════════════════════════
# IS Code References:
# IS 2062:2011  — E = 200 GPa constant (not a feature)
# IS 9077:1979  — Corrosion factor (C3 Urban)
# IRC 6:2017    — Load scenarios
# IS 800:2007   — FOS based condition labels
# ══════════════════════════════════════════════════════

# ── LOAD MODEL ──
with open('models/bridge_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Must match ml_model.py training features exactly
# E_effective_GPa removed — IS 2062 constant, zero variance
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

# ── VISUALIZATION FUNCTION ──
def predict_and_plot(dead_kN, live_kN, A_base_mm2, Fy,
                     corrosion_factor,
                     title, filename,
                     per_member_corrosion=False):
    """
    Predict member conditions and plot heatmap.
    IS 9077: corrosion_factor = area retention ratio
    IS 2062: E = 200 GPa constant (not passed as parameter)
    """
    rows = []
    corrosion_map = {}

    for mem_id, ni, nj in member_list:
        xi,yi = nodes[ni]; xj,yj = nodes[nj]
        length = np.sqrt((xj-xi)**2 + (yj-yi)**2)

        # IS 9077: per-member or uniform corrosion
        if per_member_corrosion:
            cf = np.random.uniform(corrosion_factor, 1.0)
        else:
            cf = corrosion_factor
        corrosion_map[mem_id] = cf

        rows.append({
            'MemberID':           mem_id,
            'MemberType_enc':     le.transform([member_types[mem_id]])[0],
            'Length_m':           round(length, 3),
            'Area_original_mm2':  A_base_mm2,
            'Area_effective_mm2': round(A_base_mm2 * cf, 3),
            'CorrosionFactor':    round(cf, 4),
            # E_effective_GPa removed — IS 2062 constant
            'YieldStrength_MPa':  Fy,
            'DeadLoad_kN':        dead_kN,
            'LiveLoad_kN':        live_kN,
        })

    # Enforce exact feature order from training
    X_new      = pd.DataFrame(rows)[ML_FEATURES]
    preds      = model.predict(X_new)
    probas     = model.predict_proba(X_new)
    crit_idx   = list(model.classes_).index('Critical')
    crit_probs = probas[:, crit_idx]

    # ── PLOT ──
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    cmap = plt.cm.RdYlGn_r
    norm = Normalize(vmin=0, vmax=1)

    for idx, (mem_id, ni, nj) in enumerate(member_list):
        x1,y1 = nodes[ni]; x2,y2 = nodes[nj]
        prob  = crit_probs[idx]
        color = cmap(norm(prob))
        lw    = 4.5 if member_types[mem_id] in \
                ['Bottom Chord','Top Chord','End Post'] else 2.5
        ax.plot([x1,x2],[y1,y2], color=color,
                linewidth=lw, solid_capstyle='round')

        mx,my = (x1+x2)/2,(y1+y2)/2
        ax.text(mx, my+0.18, f'M{mem_id}',
                fontsize=6.5, color='white',
                ha='center', va='bottom', alpha=0.85)

    for nid,(x,y) in nodes.items():
        ax.plot(x, y, 'o', color='white',
                markersize=7, zorder=5)
        ax.text(x, y-0.38, str(nid),
                fontsize=7, color='#bbbbbb', ha='center')

    ax.annotate('▲ PIN',    xy=(0,0),  fontsize=9,
                color='cyan', ha='center',
                va='top', xytext=(0,-0.65))
    ax.annotate('▲ ROLLER', xy=(24,0), fontsize=9,
                color='cyan', ha='center',
                va='top', xytext=(24,-0.65))

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax,
                        fraction=0.02, pad=0.02)
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
              fontsize=9, title='ML Prediction',
              title_fontsize=9)

    n_safe = (preds=='Safe').sum()
    n_risk = (preds=='At-Risk').sum()
    n_crit = (preds=='Critical').sum()
    corr_pct = int((1-corrosion_factor)*100)

    ax.set_title(
        f'{title}\n'
        f'IRC 6: Dead={dead_kN}kN Live={live_kN}kN  |  '
        f'IS 9077 C3 Corrosion: {corr_pct}%  |  '
        f'IS 2062 E=200GPa (const)\n'
        f'IS 800 FOS: Safe={n_safe}  At-Risk={n_risk}  '
        f'Critical={n_crit}',
        color='white', fontsize=11, pad=15
    )

    ax.set_xlim(-2, 27); ax.set_ylim(-1.5, 6.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Span (m)', color='white', fontsize=10)
    ax.set_ylabel('Height (m)', color='white', fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    plt.tight_layout()
    path = f'outputs/{filename}'
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"✅ {path}")
    print(f"   Safe:{n_safe} At-Risk:{n_risk} Critical:{n_crit}\n")

# ── RUN 4 IS CODE SCENARIOS ──
np.random.seed(42)
print("Generating IS code compliant heatmaps...\n")

# Scenario 1 — New bridge, dead load only
predict_and_plot(
    dead_kN=80, live_kN=0,
    A_base_mm2=1903, Fy=250,
    corrosion_factor=1.0,
    title='Scenario 1 — New Bridge | Dead Load Only (IRC 6:2017)',
    filename='heatmap_S1_new_light.png'
)

# Scenario 2 — New bridge, full IRC Class A service
predict_and_plot(
    dead_kN=100, live_kN=75,
    A_base_mm2=1903, Fy=250,
    corrosion_factor=1.0,
    title='Scenario 2 — New Bridge | Full IRC Class A Service Load',
    filename='heatmap_S2_new_full.png'
)

# Scenario 3 — 20% corroded, IS 9077 C3 (50yr top chord)
predict_and_plot(
    dead_kN=100, live_kN=75,
    A_base_mm2=1903, Fy=250,
    corrosion_factor=0.80,
    title='Scenario 3 — IS 9077 C3: 20% Corroded | Full Service Load',
    filename='heatmap_S3_corroded_full.png'
)

# Scenario 4 — Severe degradation, per-member corrosion, overload
predict_and_plot(
    dead_kN=150, live_kN=150,
    A_base_mm2=1200, Fy=250,
    corrosion_factor=0.70,
    title='Scenario 4 — IS 9077 C3: 30% Corroded | IRC Overload Condition',
    filename='heatmap_S4_degraded_critical.png',
    per_member_corrosion=True
)

print("🎉 All heatmaps complete — IS code compliant!")
