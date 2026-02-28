import numpy as np
import pandas as pd

# ── MODEL INFO ──
# Pratt Truss Bridge — 24m span, 4m height, 8 panels
# Statically Determinate: m = 2j - 3 = 29
# Units: SI (N, m, Pa)

# ── NODES ──
# Bottom chord (9 nodes)
# Top chord    (7 nodes)

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

member_list = [
    # Bottom chord (8 members) — in tension under gravity load
    (1,1,2),(2,2,3),(3,3,4),(4,4,5),
    (5,5,6),(6,6,7),(7,7,8),(8,8,9),

    # Top chord (6 members) — in compression under gravity load
    # Connects top nodes only (10 to 16)
    (9,10,11),(10,11,12),(11,12,13),
    (12,13,14),(13,14,15),(14,15,16),

    # End posts (2 members) — connect supports to top chord
    (15,1,10),   # Left end post
    (16,9,16),   # Right end post

    # Verticals (7 members)
    (17,2,10),(18,3,11),(19,4,12),(20,5,13),
    (21,6,14),(22,7,15),(23,8,16),

    # Diagonals — Pratt orientation (6 members)
    # Slope towards center → tension members
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

# ── MATERIAL PROPERTIES ──
# Steel IS 2062 Grade E250
E  = 200e9    # N/m²  (200 GPa) — SI units
A  = 1903e-6  # m²    (ISA 100x100x10 angle section)
Fy = 250e6    # N/m²  (250 MPa yield strength)

# ── SUPPORTS ──
# Node 1 → Pin   (restrain X and Y)
# Node 9 → Roller (restrain Y only)

# ── DETERMINACY CHECK ──
j = len(nodes)
m = len(member_list)
print(f"✅ Truss Model Summary")
print(f"   Nodes        : {j}")
print(f"   Members      : {m}")
print(f"   2j - 3       : {2*j-3}")
print(f"   Determinacy  : {'✅ Statically Determinate' if m == 2*j-3 else f'⚠️ Indeterminate by {m-(2*j-3)}'}")
print(f"   Span         : 24m")
print(f"   Height       : 4m")
print(f"   Panels       : 8 x 3m")
print(f"   E            : 200 GPa (200e9 N/m²)")
print(f"   Section      : ISA 100x100x10 | A = 1903 mm²")
print(f"   Yield Str.   : 250 MPa")
print(f"   Supports     : Node 1 = Pin | Node 9 = Roller")
