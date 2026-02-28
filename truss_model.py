import openseespy.opensees as ops
import numpy as np
import pandas as pd

# Clear any previous model
ops.wipe()

# 2D model, 2 DOF per node
ops.model('basic', '-ndm', 2, '-ndf', 2)

# ── NODES ──
# Bottom chord
ops.node(1,  0, 0)
ops.node(2,  3, 0)
ops.node(3,  6, 0)
ops.node(4,  9, 0)
ops.node(5, 12, 0)
ops.node(6, 15, 0)
ops.node(7, 18, 0)
ops.node(8, 21, 0)
ops.node(9, 24, 0)

# Top chord
ops.node(10,  3, 4)
ops.node(11,  6, 4)
ops.node(12,  9, 4)
ops.node(13, 12, 4)
ops.node(14, 15, 4)
ops.node(15, 18, 4)
ops.node(16, 21, 4)

# ── SUPPORTS ──
ops.fix(1, 1, 1)   # Pin
ops.fix(9, 0, 1)   # Roller

# ── MATERIAL ──
E = 200e6    # kN/m² (Steel)
A = 1903e-6  # m² (ISA 100x100x10)
ops.uniaxialMaterial('Elastic', 1, E)

# ── MEMBERS ──
member_list = [
    # Bottom chord
    (1,  1,  2), (2,  2,  3), (3,  3,  4), (4,  4,  5),
    (5,  5,  6), (6,  6,  7), (7,  7,  8), (8,  8,  9),
    # Top chord
    (9,  1, 10), (10, 10, 11), (11, 11, 12), (12, 12, 13),
    (13, 13, 14), (14, 14, 15), (15, 15, 16), (16, 16,  9),
    # Verticals
    (17,  2, 10), (18,  3, 11), (19,  4, 12), (20,  5, 13),
    (21,  6, 14), (22,  7, 15), (23,  8, 16),
    # Diagonals
    (24, 10,  3), (25, 11,  4), (26, 12,  5), (27, 13,  6),
    (28, 14,  7), (29, 15,  8), (30, 10,  2), (31, 16,  9)
]

for mem_id, node_i, node_j in member_list:
    ops.element('Truss', mem_id, node_i, node_j, A, 1)

print(f"✅ Truss model built successfully!")
print(f"   Nodes: 16")
print(f"   Members: {len(member_list)}")
print(f"   Span: 24m | Height: 4m | Panels: 8")
```

**Step 5:** Scroll down → write commit message:
```
Added truss model - 16 nodes, 31 members, Pratt truss 24m span
