#!/usr/bin/env python3
"""Actual PyBullet D435i RGB-D rendering followed by voxel TSDF fusion."""
import argparse, json, math
from pathlib import Path
import numpy as np, pybullet as p, pybullet_data
from PIL import Image
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter

def view_matrix(view):
    # PyBullet returns OpenGL matrices in column-major order.  Keeping this
    # conversion explicit avoids mixing the renderer and NumPy conventions.
    return np.asarray(view, dtype=np.float64).reshape((4, 4), order='F')
def gl_matrix(mat):
    return np.asarray(mat, dtype=np.float64).reshape((4, 4), order='F')
def ply(path,v,f):
    with open(path,'w') as h:
        h.write(f'ply\nformat ascii 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n'); h.writelines(f'{x:.6f} {y:.6f} {z:.6f}\n' for x,y,z in v); h.writelines(f'3 {a} {b} {c}\n' for a,b,c in f)
def add_box(p, half, pos, color):
    col=p.createCollisionShape(p.GEOM_BOX,halfExtents=half)
    vis=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=color)
    p.createMultiBody(0,col,vis,basePosition=pos)
def add_furniture_chair(p):
    """A chair with seat, four legs, back and two support rails (meters)."""
    wood=[.38,.18,.07,1]
    add_box(p,[.21,.21,.025],[0,0,.43],wood)              # seat
    add_box(p,[.21,.025,.225],[0,.185,.655],wood)         # backrest
    for x in [-.18,.18]:
        for y in [-.18,.18]: add_box(p,[.025,.025,.215],[x,y,.215],wood)
    add_box(p,[.19,.018,.018],[0,-.18,.31],wood)          # lower rail
    add_box(p,[.018,.19,.018],[-.18,0,.31],wood)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='output_pybullet'); ap.add_argument('--views',type=int,default=12); ap.add_argument('--voxel',type=float,default=.004); ap.add_argument('--rings',type=int,default=3,help='camera elevation rings: side/top/bottom'); ap.add_argument('--smooth',type=float,default=.6,help='TSDF Gaussian sigma in voxels'); ap.add_argument('--target',choices=['mug','chair'],default='mug'); a=ap.parse_args(); out=Path(a.output); (out/'depth').mkdir(parents=True,exist_ok=True)
    # D435i depth stream model: 640x480 mode, ~87 deg horizontal FOV,
    # 50 mm stereo baseline, 0.105 m minimum reliable range.
    # PyBullet's computeProjectionMatrixFOV() takes a *vertical* FOV.  The
    # TSDF intrinsics must use the same convention; treating 87 degrees as a
    # horizontal FOV projected every voxel at the wrong image position and
    # created layered "ghost" surfaces on thin chair parts.
    W,H=640,480; fov_y=87.; near,far=.105,2.
    fy=.5*H/math.tan(math.radians(fov_y/2)); fx=fy*(W/H)
    K=(fx,fy,W/2-.5,H/2-.5)
    # PyBullet mug: curved body, open top, inner wall and handle.  Center and
    # scale its known source bounds to make a roughly 18 cm target.
    mesh_scale=1.5; source_size=np.array([.082,.121633,.1]); source_center=np.array([0.,.0198165,.05]); size=source_size*mesh_scale if a.target=='mug' else np.array([.42,.42,.88])
    lo=(-size/2-.04) if a.target=='mug' else np.array([-.25,-.25,-.04]); shape=np.ceil((size+.08)/a.voxel).astype(int)+1; xyz=lo+(np.indices(shape).transpose(1,2,3,0)+.5)*a.voxel; xyz=xyz.reshape(-1,3); field=np.zeros(len(xyz),np.float32); weight=np.zeros(len(xyz),np.float32); trunc=5*a.voxel
    cid=p.connect(p.DIRECT); p.setAdditionalSearchPath(pybullet_data.getDataPath()); p.setGravity(0,0,-9.81); # target-only sensor test: no floor/background depth
    if a.target=='mug':
        target_mesh=Path(__file__).parent/'assets/mug.obj'; target_collision=Path(__file__).parent/'assets/mug_col.obj'; base=(-source_center*mesh_scale).tolist()
        visual=p.createVisualShape(p.GEOM_MESH,fileName=str(target_mesh),meshScale=[mesh_scale]*3,rgbaColor=[.12,.42,.82,1]); col=p.createCollisionShape(p.GEOM_MESH,fileName=str(target_collision),meshScale=[mesh_scale]*3,flags=p.GEOM_FORCE_CONCAVE_TRIMESH); p.createMultiBody(0,col,visual,basePosition=base)
    else: add_furniture_chair(p)
    for i in range(a.views):
        nring=max(1,a.rings); per=max(1,a.views//nring); ring=i//per; j=i%per; t=2*math.pi*j/per
        center_z=.44 if a.target=='chair' else 0.; radius=.78 if a.target=='chair' else .48
        if a.target=='chair' and nring >= 5:
            # Five elevations cover leg undersides, seat, back and top edge.
            zring=[.02,.22,.44,.72,1.02][min(ring,4)]
        elif nring==3: zring=[center_z,center_z+.45,center_z-.30][min(ring,2)]
        else: zring=center_z+.24*math.sin(2*math.pi*ring/nring)
        cam=np.array([radius*math.cos(t),radius*math.sin(t),zring]); target=[0,0,center_z]; view=p.computeViewMatrix(cam,target,[0,0,1]); V=view_matrix(view); proj=p.computeProjectionMatrixFOV(fov_y,W/H,near,far); P=gl_matrix(proj); _,_,rgba,depthbuf,_=p.getCameraImage(W,H,view,proj,renderer=p.ER_TINY_RENDERER); depthbuf=np.asarray(depthbuf).reshape(H,W); depth=far*near/(far-(far-near)*depthbuf); depth[depthbuf>=.9999]=0; Image.fromarray((depth*1000).astype(np.uint16)).save(out/'depth'/f'{i:03d}.png')
        # Project with the exact OpenGL V/P pair used by getCameraImage.
        clip=(np.c_[xyz,np.ones(len(xyz))] @ (P@V).T); ndc=clip[:,:3]/np.maximum(clip[:,3,None],1e-9); u=np.rint((ndc[:,0]+1)*.5*W).astype(int); v=np.rint((1-ndc[:,1])*.5*H).astype(int); pc=(np.c_[xyz,np.ones(len(xyz))] @ V.T); z=-pc[:,2]; ok=(clip[:,3]>0)&(u>=0)&(u<W)&(v>=0)&(v<H); uu=np.clip(u,0,W-1); vv=np.clip(v,0,H-1); d=depth[vv,uu]; sdf=d-z; ok &= d>0; ok &= (sdf>=-trunc)&(sdf<=trunc); field[ok]+=np.clip(sdf[ok]/trunc,-1,1); weight[ok]+=1
    p.disconnect(); observed=weight>=2; ts=np.divide(field,np.maximum(weight,1),where=weight>0,out=np.ones_like(field)); ts[~observed]=1; ts=gaussian_filter(ts.reshape(shape),sigma=a.smooth,mode='nearest') if a.smooth>0 else ts.reshape(shape); v,f,_,_=marching_cubes(ts,0,spacing=(a.voxel,)*3); v+=lo; keep=np.all((v>=lo-a.voxel)&(v<=lo+size+2*a.voxel),1); rem=-np.ones(len(v),int); rem[keep]=np.arange(keep.sum()); fk=np.all(keep[f],1); ply(out/'tsdf_mesh.ply',v[keep],rem[f[fk]]); rec=v[keep].max(0)-v[keep].min(0); err=np.abs(rec-size); report={'renderer':'PyBullet TinyRenderer','camera':'Intel RealSense D435i (simulated)','target':('procedural furniture chair' if a.target=='chair' else 'PyBullet mug mesh'),'resolution':[W,H],'vertical_fov_deg':fov_y,'intrinsics_px':{'fx':fx,'fy':fy,'cx':K[2],'cy':K[3]},'near_clip_m':near,'stereo_baseline_m':0.050,'views':a.views,'voxel_m':a.voxel,'min_observations':2,'tsdf_smoothing_sigma_voxels':a.smooth,'ground_truth_size_m':size.tolist(),'reconstructed_bbox_m':rec.tolist(),'bbox_abs_error_m':err.tolist(),'pass':bool(err.max()<=3*a.voxel)}; (out/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
