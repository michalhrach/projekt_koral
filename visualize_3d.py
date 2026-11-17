"""3D izoplocha koncentrace V -> skutecny otacejici se GIF (nikoli 2D relief)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

base=Path('frames3d')
N,frames,iso=base.joinpath('config.txt').read_text().split()
N,frames,iso=int(N),int(frames),float(iso)
files=sorted(base.glob('frame_*.bin'))
if len(files)!=frames: raise RuntimeError(f'Ocekavano {frames} snimku, nalezeno {len(files)}')
out=Path('vysledky3d');out.mkdir(exist_ok=True)
fig=plt.figure(figsize=(6,6),dpi=95,facecolor='#101821')
ax=fig.add_subplot(111,projection='3d',computed_zorder=False)
fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.96)
writer=PillowWriter(fps=8)
with writer.saving(fig,str(out/'brain_coral_3d.gif'),dpi=95):
    for j,f in enumerate(files):
        v=np.fromfile(f,dtype=np.float32)
        if v.size != N**3: raise RuntimeError(f'Neplatny snimek {f}')
        volume=v.reshape((N,N,N))
        ax.cla()
        ax.set_facecolor('#101821')
        ax.set_xlim(0,N);ax.set_ylim(0,N);ax.set_zlim(0,N)
        ax.set_box_aspect((1,1,1),zoom=1.04)
        ax.set_axis_off()
        ax.view_init(elev=23,azim=35 + 310*j/max(1,frames-1))
        if np.min(volume)<iso<np.max(volume):
            verts,faces,_,_=marching_cubes(volume,iso,step_size=3,allow_degenerate=False)
            # Omez mnozstvi trojuhelniku kvuli rychlosti animace.
            tris=verts[faces]
            normal=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0])
            size=np.linalg.norm(normal,axis=1)
            normal/=np.maximum(size[:,None],1e-10)
            lamp=np.array([.35,-.4,.85])
            lamp/=np.linalg.norm(lamp)
            brightness=np.clip(.45+.55*np.abs(normal@lamp),.2,1.)
            basecolor=np.array([.93,.66,.39])
            colors=np.column_stack([brightness[:,None]*basecolor,np.ones(len(brightness))])
            mesh=Poly3DCollection(tris,facecolors=colors,edgecolors='none',linewidth=0,alpha=1,zsort='average')
            ax.add_collection3d(mesh)
        ax.set_title(f'Gray–Scott 3D   |   snimek {j+1}/{frames}',color='white',fontsize=12,pad=4)
        writer.grab_frame()
        print(f'3D GIF: {j+1}/{frames}',flush=True)
plt.close(fig)
print('Hotovo:',out/'brain_coral_3d.gif')
