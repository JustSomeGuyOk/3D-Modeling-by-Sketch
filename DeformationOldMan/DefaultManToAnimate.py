import igl
import numpy as np
import meshplot as mp
from meshplot import subplot, plot
import os

root_folder = os.getcwd()

# [4. Geometry Deformation] --------------------------------------
vv, ff = igl.read_triangle_mesh(os.path.join(root_folder, "data", "DefaultManToAnimate.obj"))

# Read .dmat values to move plots
s = igl.read_dmat(os.path.join(root_folder, "data", "HandlesforDefaultMan.dmat"))
b = np.array([[t[0] for t in [(i, s[i]) for i in range(0, vv.shape[0])] if t[1] >= 0]]).T

u_bc = np.zeros((b.shape[0], vv.shape[1]))
v_bc = np.zeros((b.shape[0], vv.shape[1]))

for bi in range(b.shape[0]):
    v_bc[bi] = vv[b[bi]]

    if s[b[bi]] == 0: # Don't move handle 0
        u_bc[bi] = vv[b[bi]]
    elif s[b[bi]] == 1: # Move handle 1 (+ up, - down) (forward/backward, up/down, left/right)
        u_bc[bi] = vv[b[bi]] + np.array([[0, 20, -10]])
    else: # Move other handles forward
        u_bc[bi] = vv[b[bi]] + np.array([[0, 0, 0]])

p = subplot(vv, ff, s, shading={"wireframe": False, "colormap": "tab10"}, s=[1, 4, 0])
for i in range(3):
    u_bc_anim = v_bc + i*0.6 * (u_bc - v_bc)
    d_bc = u_bc_anim - v_bc
    d = igl.harmonic_weights(vv, ff, b, d_bc, 2)
    u = vv + d
    subplot(u, ff, s, shading={"wireframe": False, "colormap": "tab10"}, s=[1, 4, i+1], data=p)

#ret = igl.write_triangle_mesh(os.path.join(root_folder, "data", "decimated-max.obj"), u, ff)
mp.offline()
plot(u, ff, shading={"wireframe": False})