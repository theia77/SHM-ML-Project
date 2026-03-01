# ══════════════════════════════════════════════════════
# TRUSS MODEL — Reference Definition File
#
# IS Code References:
# IS 2062:2011  — Steel material properties
# IS 808:1989   — Angle section dimensions
# IS 800:2007   — Steel structure design
# IRC 6:2017    — Bridge loading standards
# IRC 24:2010   — Steel road bridge code
#
# Bridge Type : Simply Supported Pratt Through Truss
# Span        : 24m (8 panels × 3m)
# Height      : 4m (depth/span = 1/6 — IS 800 standard)
# Members     : 29 (statically determinate: m = 2j-3)
# Nodes       : 16
# ══════════════════════════════════════════════════════

import numpy as np
import pandas as pd

# ── NODES ──
# Bottom chord: nodes 1-9  (Y = 0)
# Top chord:    nodes 10-16 (Y = 4m)

nodes = {
    # Bottom chord
    1:(0,0),  2:(3,0),  3:(6,0),  4:(9,0),  5:(12,0),
    6:(15,0), 7:(18,0), 8:(21,0), 9:(24,0),
    # Top chord
    10:(3,4), 11:(6,4), 12:(9,4), 13:(12,4),
    14:(15,4),15:(18,4),16:(21,4)
}

# ── MEMBER LIST ──
# Format: (MemberID, NodeI, NodeJ)
# Pratt truss: diagonals slope toward center — tension under gravity

member_list = [
    # Bottom chord (8) — tension members under gravity load
    (1,1,2),(2,2,3),(3,3,4),(4,4,5),
    (5,5,6),(6,6,7),(7,7,8),(8,8,9),

    # Top chord (6) — compression members under gravity load
    # Connects top nodes only (10→16)
    (9,10,11),(10,11,12),(11,12,13),
    (12,13,14),(13,14,15),(14,15,16),

    # End posts (2) — connect supports to top chord
    (15,1,10),   # Left end post
    (16,9,16),   # Right end post

    # Verticals (7) — carry shear
    (17,2,10),(18,3,11),(19,4,12),(20,5,13),
    (21,6,14),(22,7,15),(23,8,16),

    # Diagonals (6) — Pratt orientation, tension members
    (24,10,3),(25,11,4),(26,12,5),
    (27,13,6),(28,14,7),(29,15,8)
]

# ── MEMBER TYPE LABELS ──
member_types = {
    **{i:'Bottom Chord' for i in range(1,9)},
    **{i:'Top Chord'    for i in range(9,15)},
    **{i:'End Post'     for i in range(15,17)},
    **{i:'Vertical'     for i in range(17,24)},
    **{i:'Diagonal'     for i in range(24,30)}
}

# ── MATERIAL PROPERTIES (IS 2062:2011) ──
E  = 200e9    # N/m² = 200 GPa — Young's Modulus (Cl. 2.2.1)
              # CONSTANT — does not change with age or corrosion
A  = 1903e-6  # m²   — ISA 100×100×10 (IS 808:1989, Table 1)
Fy = 250e6    # N/m² — Yield strength, Grade E250 (IS 2062 Table 1)
Fu = 410e6    # N/m² — Ultimate strength, Grade E250
density = 7850 # kg/m³ — Steel density

# ── SUPPORT CONDITIONS ──
# Node 1 → Pin   (restrain X and Y translation)
# Node 9 → Roller (restrain Y translation only)
# Simply supported — standard for road bridges (IRC 24:2010)

# ── DETERMINACY CHECK ──
j = len(nodes)          # 16 joints
m = len(member_list)    # 29 members
r = 3                   # reactions (pin=2, roller=1)

print(f"✅ Truss Model — IS Code Summary")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  Nodes        : {j}")
print(f"  Members      : {m}")
print(f"  Reactions    : {r}")
print(f"  2j - 3       : {2*j-3}")
print(f"  Determinacy  : "
      f"{'✅ Statically Determinate' if m==2*j-3 else f'⚠️ Indeterminate by {m-(2*j-3)}'}")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  Span         : 24m (8 panels × 3m)")
print(f"  Height       : 4m (depth/span = 1/6)")
print(f"  Bridge type  : Simply Supported Pratt Truss")
print(f"  Support      : Pin (Node 1) + Roller (Node 9)")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  IS 2062:2011 : Grade E250")
print(f"  E            : 200 GPa (constant)")
print(f"  Fy           : 250 MPa")
print(f"  Fu           : 410 MPa")
print(f"  Section      : ISA 100×100×10 (IS 808)")
print(f"  Area         : 1903 mm²")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  IRC 6:2017   : Dead=100 kN, Live=75 kN per panel")
print(f"  IS 800:2007  : FOS basis for condition labeling")
print(f"  IS 9077:1979 : C3 Urban corrosion environment")
# IRC 6:2017 — TOTAL loads per truss (kN)
# Dead: 200-400 kN | Live: 0-300 kN
# panel_load = (dead + live) / 7 nodes

