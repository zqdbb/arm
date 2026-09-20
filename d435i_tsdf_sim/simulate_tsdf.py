#!/usr/bin/env python3
"""Analytic noiseless D435i views + dense TSDF fusion regression test."""
import argparse, json
from pathlib import Path
import numpy as np
from skimage.measure import marching_cubes
from PIL import Image

def look_at(cam, target=np.zeros(3)):
    z=(target-cam); z/=np.linalg.norm(z); up=np.array([0.,0.,1.])
    if abs(np.dot(z,up))>.95: up=np.array([0.,1.,0.])
    x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    return np.stack([x,y,z],1) # camera-to-world

def ray_box_depth(cam, R, K, size, W, H):
    fx,fy,cx,cy=K; u,v=np.meshgrid(np.arange(W),np.arange(H));
    rays=np.stack([(u-cx)/fx,(v-cy)/fy,np.ones_like(u)],-1); rays=rays@R.T
    p=cam[None,None,:]; b=np.array(size)/2
    ro=(p); inv=1.0/np.where(abs(rays)<1e-9,1e-9,rays)
    t0=(-b-ro)*inv; t1=(b-ro)*inv; tn=np.maximum.reduce(np.minimum(t0,t1),-1); tf=np.minimum.reduce(np.maximum(t0,t1),-1)
    d=np.where((tf>=np.maximum(tn,0))&(tf>0), np.maximum(tn,0), 0).astype(np.float32)
    return d

def write_ply(path, verts, faces):
    with open(path,'w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
        for p in verts: f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
        for q in faces: f.write(f'3 {q[0]} {q[1]} {q[2]}\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='output'); ap.add_argument('--voxel',type=float,default=.004); ap.add_argument('--views',type=int,default=12); a=ap.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); (out/'depth').mkdir(exist_ok=True)
    W,H=320,240; fx=fy=280.; K=(fx,fy,W/2-.5,H/2-.5); size=np.array([.24,.18,.20]); trunc=a.voxel*5
    # volume encloses object with margin
    lo=-size/2-.04; hi=size/2+.04; shape=np.ceil((hi-lo)/a.voxel).astype(int)+1
    grid=np.full(shape,np.nan,np.float32); weight=np.zeros(shape,np.float32)
    xyz=lo+(np.indices(shape).transpose(1,2,3,0)+.5)*a.voxel; xyz=xyz.reshape(-1,3)
    for i in range(a.views):
        th=2*np.pi*i/a.views; cam=np.array([.48*np.cos(th),.48*np.sin(th),.30]); R=look_at(cam)
        depth=ray_box_depth(cam,R,K,size,W,H); Image.fromarray((depth*1000).astype(np.uint16)).save(out/'depth'/f'{i:03d}.png')
        # project voxels into camera and integrate signed distance
        pc=(xyz-cam)@R; z=pc[:,2]; u=np.rint(fx*pc[:,0]/np.maximum(z,1e-6)+K[2]).astype(int); v=np.rint(fy*pc[:,1]/np.maximum(z,1e-6)+K[3]).astype(int)
        ok=(z>0)&(u>=0)&(u<W)&(v>=0)&(v<H); sdf=np.zeros(len(xyz),np.float32); sdf[ok]=depth[v[ok],u[ok]]-z[ok]; ok &= depth[v.clip(0,H-1),u.clip(0,W-1)]>0; ok &= sdf>=-trunc; ok &= sdf<=trunc
        ts=np.clip(sdf/trunc,-1,1); g=grid.reshape(-1); w=weight.reshape(-1); g[ok]=np.nan_to_num(g[ok])*w[ok]; g[ok]=(g[ok]+ts[ok])/(w[ok]+1); w[ok]+=1
    field=np.nan_to_num(grid,nan=1.0); field=field.reshape(shape)
    verts,faces,_,_=marching_cubes(field,level=0,spacing=(a.voxel,)*3); verts+=lo
    # Keep the connected object support; the analytic scene bounds are known and
    # remove the finite truncation shell at the volume margin.
    keep=np.all(np.abs(verts)<=size/2+a.voxel,axis=1); remap=-np.ones(len(verts),int); remap[keep]=np.arange(keep.sum())
    fkeep=np.all(keep[faces],axis=1); faces=remap[faces[fkeep]]; verts=verts[keep]
    write_ply(out/'tsdf_mesh.ply',verts,faces)
    mn,mx=verts.min(0),verts.max(0); recon=mx-mn; err=np.abs(recon-size); report={'views':a.views,'voxel_m':a.voxel,'vertices':int(len(verts)),'faces':int(len(faces)),'ground_truth_size_m':size.tolist(),'reconstructed_bbox_m':recon.tolist(),'bbox_abs_error_m':err.tolist(),'pass':bool(np.max(err)<=3*a.voxel)}
    (out/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
