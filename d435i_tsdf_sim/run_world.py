#!/usr/bin/env python3
"""Headless PyBullet smoke test for the D435i/object simulation world."""
import pybullet as p, pybullet_data, time
cid=p.connect(p.DIRECT); p.setAdditionalSearchPath(pybullet_data.getDataPath()); p.setGravity(0,0,-9.81)
floor=p.loadURDF('plane.urdf'); col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.12,.09,.10]); obj=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,basePosition=[0,0,0])
for i in range(12):
    import math
    t=2*math.pi*i/12; cam=[.48*math.cos(t),.48*math.sin(t),.30]; p.computeViewMatrix(cam,[0,0,0],[0,0,1])
print(f'PyBullet world OK: bodies={p.getNumBodies()}, views=12, object=0.24x0.18x0.20 m')
p.disconnect()
