import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pickle
import os

# ══════════════════════════════════════════════════════
# SYSTEM-LEVEL RISK ANALYSIS
# Surrogate Risk Framework — ML-Assisted SHM
#
# IS 2062:2011 | IS 9077:1979 | IRC 6:2017 | IS 800:2007
# ══════════════════════════════════════════════════════

print("="*60)
print("  SYSTEM-LEVEL RISK ANALYSIS")
print("  Steel Pratt Truss — Surrogate Risk Framework")
print("="*60)

with open('models/bridge_rf_model.pkl','rb') as f:
    model = pickle.load(f)
with open('models/label_encoder.pkl','rb') as f:
    le_member = pickle.load(f)
with open('models/le_use.pkl','rb') as f:
    le_use = pickle.load(f)
with open('models/le_env.pkl','rb') as f:
    le_env = pickle.load(f)
with open('models/le_maint.pkl','rb') as f:
    le_maint = pickle.load(f)

ML_FEATURES = [
    'MemberID', 'MemberType_enc', 'Length_m',
    'Area_original_mm2', 'Area_effective_mm2',
    'CorrosionFactor', 'YieldStrength_MPa',
    'DeadLoad_kN', 'LiveLoad_kN',
    'AgeYears', 'UseCase_enc',
    'Environment_enc', 'Maintenance_enc',
]

N_TOTAL = 29
A_BASE  = 1903
Fy      = 250

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

primary_members   = list(range(1,17))
secondary_members = list(range(17,30))

os.makedirs('outputs', exist_ok=True)
np.random.seed(7)

corrosion_rates = {
    'Bottom Chord': 0.0050,
    'Top Chord':    0.0030,
    'End Post':     0.0050,
    'Vertical':     0.0040,
    'Diagonal':     0.0040,
}

def get_member_probabilities(age_years, dead_kN, live_kN,
                              use_case, environment,
                              maintenance):
    rows   = []
    cf_map = {}

    maint_factors = {
        'Good':0.30, 'Partial':0.60, 'None':1.0}
    mf = maint_factors[maintenance]

    for mem_id, ni, nj in member_list:
        xi,yi=nodes[ni]; xj,yj=nodes[nj]
        length=np.sqrt((xj-xi)**2+(yj-yi)**2)
        mtype=member_types[mem_id]

        base_loss = corrosion_rates[mtype]*mf*age_years
        variation = base_loss*np.random.uniform(-0.1,0.1)\
                    if age_years>0 else 0
        cf=float(np.clip(1.0-base_loss+variation,0.60,1.0))
        cf_map[mem_id]=cf

        rows.append({
            'MemberID':           mem_id,
            'MemberType_enc':     le_member.transform(
                                  [mtype])[0],
            'Length_m':           round(length,3),
            'Area_original_mm2':  A_BASE,
            'Area_effective_mm2': round(A_BASE*cf,3),
            'CorrosionFactor':    round(cf,4),
            'YieldStrength_MPa':  Fy,
            'DeadLoad_kN':        dead_kN,
            'LiveLoad_kN':        live_kN,
            'AgeYears':           age_years,
            'UseCase_enc':        le_use.transform(
                                  [use_case])[0],
            'Environment_enc':    le_env.transform(
                                  [environment])[0],
            'Maintenance_enc':    le_maint.transform(
                                  [maintenance])[0],
        })

    X_new      = pd.DataFrame(rows)[ML_FEATURES]
    preds      = model.predict(X_new)
    probas     = model.predict_proba(X_new)
    crit_idx   = list(model.classes_).index('Critical')
    crit_probs = {
        mem_id: probas[i,crit_idx]
        for i,(mem_id,_,_) in enumerate(member_list)
    }
    return preds, crit_probs, cf_map

def compute_system_risk(crit_probs, preds):
    p_primary = {
        m: crit_probs[m]
        for m in primary_members if m in crit_probs
    }
    sri_mean = float(np.mean(list(p_primary.values())))
    sri_max  = float(np.max(list(p_primary.values())))
    weights  = {
        m: 1.5 if member_types[m]=='Bottom Chord' else 1.0
        for m in p_primary
    }
    total_w      = sum(weights[m] for m in p_primary)
    sri_weighted = float(
        sum(p_primary[m]*weights[m]
            for m in p_primary)/total_w
    )
    health_score = round(1.0-sri_weighted,4)
    n_primary_crit = sum(
        1 for m in primary_members
        if m in crit_probs and crit_probs[m]>0.5
    )
    return {
        'SRI_mean':       round(sri_mean,4),
        'SRI_max':        round(sri_max,4),
        'SRI_weighted':   round(sri_weighted,4),
        'health_score':   health_score,
        'n_primary_crit': n_primary_crit,
        'primary_crit_ratio': round(
            n_primary_crit/len(primary_members),4),
    }

def get_risk_status(sri_weighted):
    if sri_weighted < 0.40:  return 'LOW RISK'
    elif sri_weighted < 0.70: return 'MODERATE RISK'
    else:                     return 'HIGH RISK'

# ── 5 TIME STAGES ── same as case_study.py
stages = [
    (0,   200, 0,   'Urban',   'C3_Urban', 'Good',
     'Year 0  — New Bridge'),
    (10,  250, 75,  'Urban',   'C3_Urban', 'Good',
     'Year 10 — Early Service'),
    (20,  280, 150, 'Urban',   'C3_Urban', 'Partial',
     'Year 20 — Aging Begins'),
    (35,  320, 200, 'Highway', 'C3_Urban', 'Partial',
     'Year 35 — Significant Deterioration'),
    (50,  350, 245, 'Highway', 'C3_Urban', 'None',
     'Year 50 — Critical Condition'),
]

print("\n── System Risk Analysis ──\n")
stage_results = []

for (age,dead,live,use,env,maint,label) in stages:
    preds,crit_probs,cf_map = get_member_probabilities(
        age,dead,live,use,env,maint)

    sys      = compute_system_risk(crit_probs,preds)
    status   = get_risk_status(sys['SRI_weighted'])
    avg_corr = (1-np.mean(list(cf_map.values())))*100

    n_safe=int((preds=='Safe').sum())
    n_risk=int((preds=='At-Risk').sum())
    n_crit=int((preds=='Critical').sum())
    R=round(1-n_crit/N_TOTAL,4)

    print(f"  {label}")
    print(f"  IRC6    : Dead={dead}kN Live={live}kN")
    print(f"  Context : {use} | {env} | {maint}")
    print(f"  IS9077  : Avg Corrosion={avg_corr:.1f}%")
    print(f"  IS800   : Safe={n_safe} "
          f"At-Risk={n_risk} Critical={n_crit}")
    print(f"  R(t)        : {R:.4f}")
    print(f"  SRI_weighted: {sys['SRI_weighted']:.4f}")
    print(f"  Health Score: {sys['health_score']:.4f}")
    print(f"  Status      : {status}")
    print()

    stage_results.append({
        'age':age,'label':label,
        'dead':dead,'live':live,
        'use':use,'env':env,'maint':maint,
        'preds':preds,'crit_probs':crit_probs,
        'cf_map':cf_map,
        'n_safe':n_safe,'n_risk':n_risk,'n_crit':n_crit,
        'R':R,'avg_corr':avg_corr,
        'status':status, **sys,
    })

# ── HEATMAPS ──
print("Generating system risk heatmaps...")
cmap=plt.cm.RdYlGn_r
norm=Normalize(vmin=0,vmax=1)

for s in stage_results:
    fig,ax=plt.subplots(figsize=(16,7))
    ax.set_facecolor('#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')

    for idx,(mem_id,ni,nj) in enumerate(member_list):
        x1,y1=nodes[ni]; x2,y2=nodes[nj]
        prob=s['crit_probs'][mem_id]
        color=cmap(norm(prob))
        is_primary=mem_id in primary_members
        lw=5.0 if is_primary else 2.5
        ax.plot([x1,x2],[y1,y2],color=color,
                linewidth=lw,solid_capstyle='round')
        mx,my=(x1+x2)/2,(y1+y2)/2
        cf_pct=int((1-s['cf_map'][mem_id])*100)
        ax.text(mx,my+0.18,f'M{mem_id}\n{cf_pct}%c',
                fontsize=5.5,color='white',
                ha='center',va='bottom',alpha=0.85)

    for nid,(x,y) in nodes.items():
        ax.plot(x,y,'o',color='white',markersize=6,zorder=5)

    ax.annotate('▲ PIN',    xy=(0,0),  fontsize=9,
                color='cyan',ha='center',
                va='top',xytext=(0,-0.65))
    ax.annotate('▲ ROLLER', xy=(24,0), fontsize=9,
                color='cyan',ha='center',
                va='top',xytext=(24,-0.65))

    sm=ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    cbar=plt.colorbar(sm,ax=ax,fraction=0.02,pad=0.02)
    cbar.set_label('ML Critical Probability (Surrogate)',
                   color='white',fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(),color='white')

    patches=[
        mpatches.Patch(color='#00cc44',label='Low risk'),
        mpatches.Patch(color='#ffdd00',label='Moderate risk'),
        mpatches.Patch(color='#ff2222',label='High risk'),
        mpatches.Patch(color='white',
                       label='Thick = Primary member'),
    ]
    ax.legend(handles=patches,loc='upper right',
              facecolor='#2d2d44',labelcolor='white',
              fontsize=8,title='System Risk',
              title_fontsize=8)

    ax.set_title(
        f'System Risk — {s["label"]}\n'
        f'IRC6: Dead={s["dead"]}kN Live={s["live"]}kN | '
        f'{s["use"]} | {s["env"]} | {s["maint"]}\n'
        f'R(t)={s["R"]:.3f} | '
        f'SRI={s["SRI_weighted"]:.3f} | '
        f'Health={s["health_score"]:.3f} | '
        f'{s["status"]}',
        color='white',fontsize=10,pad=12
    )

    ax.set_xlim(-2,27); ax.set_ylim(-1.5,6.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Span (m)',color='white')
    ax.set_ylabel('Height (m)',color='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')

    plt.tight_layout()
    fname=f'outputs/system_year{s["age"]}.png'
    plt.savefig(fname,dpi=150,bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ {fname}")

# ── TREND PLOT ──
print("\nGenerating system risk trend...")
ages     =[s['age']          for s in stage_results]
sri_vals =[s['SRI_weighted'] for s in stage_results]
health   =[s['health_score'] for s in stage_results]
R_vals   =[s['R']            for s in stage_results]
corrs    =[s['avg_corr']     for s in stage_results]
n_crits  =[s['n_crit']       for s in stage_results]
n_risks  =[s['n_risk']       for s in stage_results]
n_safes  =[s['n_safe']       for s in stage_results]

fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(12,13))
fig.patch.set_facecolor('#1a1a2e')

ax1.set_facecolor('#1a1a2e')
ax1.plot(ages,R_vals,'o-',color='#00aaff',
         linewidth=3,markersize=10,
         label='R(t) Member Reliability')
ax1.plot(ages,health,'s--',color='#44ff88',
         linewidth=2.5,markersize=8,
         label='Health Score (1-SRI)')
ax1.fill_between(ages,R_vals,alpha=0.12,color='#00aaff')
ax1.axhline(y=0.80,color='#ffdd00',linestyle='--',
            alpha=0.7,label='Warning (0.80)')
ax1.axhline(y=0.60,color='#ff4444',linestyle='--',
            alpha=0.7,label='Critical (0.60)')
ax1.fill_between([0,50],0.60,0.80,
                 color='yellow',alpha=0.05)
ax1.fill_between([0,50],0,0.60,color='red',alpha=0.05)
ax1.set_ylabel('Index Value',color='white',fontsize=11)
ax1.set_title(
    'System Risk — Steel Pratt Truss | Surrogate Framework\n'
    'IS9077 C3 Urban | IS2062 | IRC6 Urban→Highway | IS800',
    color='white',fontsize=12)
ax1.legend(facecolor='#2d2d44',labelcolor='white',fontsize=9)
ax1.tick_params(colors='white')
ax1.set_ylim(0,1.05); ax1.set_xticks(ages)
ax1.grid(axis='y',color='#333355',linestyle='--',alpha=0.5)
for sp in ax1.spines.values(): sp.set_edgecolor('#333355')

ax2.set_facecolor('#1a1a2e')
ax2_r=ax2.twinx()
ax2.plot(ages,n_crits,'o-',color='#ff4444',linewidth=2.5,
         markersize=8,label='Critical (IS 800)')
ax2.plot(ages,n_risks,'s-',color='#ffdd00',linewidth=2.5,
         markersize=8,label='At-Risk')
ax2.plot(ages,n_safes,'^-',color='#44ff88',linewidth=2.5,
         markersize=8,label='Safe')
ax2_r.plot(ages,sri_vals,'D--',color='#ff88ff',
           linewidth=2,markersize=7,
           label='SRI_weighted')
ax2_r.set_ylabel('SRI',color='#ff88ff',fontsize=10)
ax2_r.tick_params(colors='#ff88ff')
ax2_r.set_ylim(0,1.05)
ax2.set_ylabel('Members',color='white',fontsize=11)
ax2.set_title('IS800 Member Conditions + SRI Trend',
              color='white',fontsize=11)
lines1,labs1=ax2.get_legend_handles_labels()
lines2,labs2=ax2_r.get_legend_handles_labels()
ax2.legend(lines1+lines2,labs1+labs2,
           facecolor='#2d2d44',labelcolor='white',fontsize=8)
ax2.tick_params(colors='white')
ax2.set_ylim(0,N_TOTAL+2); ax2.set_xticks(ages)
ax2.grid(axis='y',color='#333355',linestyle='--',alpha=0.5)
for sp in ax2.spines.values(): sp.set_edgecolor('#333355')

ax3.set_facecolor('#1a1a2e')
ax3.plot(ages,corrs,'D-',color='#ff8844',linewidth=2.5,
         markersize=8,label='IS9077 C3 Avg Corrosion')
ax3.fill_between(ages,corrs,alpha=0.15,color='#ff8844')
ax3.set_xlabel('Bridge Age (Years)',color='white',fontsize=11)
ax3.set_ylabel('Avg Corrosion Loss (%)',
               color='white',fontsize=11)
ax3.set_title('IS 9077 C3 Corrosion Progression',
              color='white',fontsize=11)
ax3.legend(facecolor='#2d2d44',labelcolor='white',fontsize=9)
ax3.tick_params(colors='white')
ax3.set_xticks(ages)
ax3.grid(axis='y',color='#333355',linestyle='--',alpha=0.5)
for sp in ax3.spines.values(): sp.set_edgecolor('#333355')

plt.tight_layout()
plt.savefig('outputs/system_reliability.png',dpi=150,
            bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("  ✅ outputs/system_reliability.png")

print(f"\n{'='*60}")
print(f"  SYSTEM RISK SUMMARY")
print(f"{'='*60}")
for s in stage_results:
    print(f"  {s['label']:<38} "
          f"R={s['R']:.2f} "
          f"SRI={s['SRI_weighted']:.3f} "
          f"Health={s['health_score']:.3f} "
          f"{s['status']}")
print(f"\n🎉 System-level analysis complete!")
