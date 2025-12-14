import igl
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse.linalg import spsolve
import meshplot as mp
from meshplot import subplot, plot
import os

root_folder = os.getcwd()

# put in border vertices
input = np.array([[1.866025, 1.5],
       [1.5     , 1.866025],
       [1.      , 2.      ],
       [0.5     , 1.866025],
       [0.133975, 1.5     ],
       [0.      , 1.      ],
       [0.133975, 0.5     ],
       [0.5     , 0.133975],
       [1.      , 0.      ],
       [1.5     , 0.133975],
       [1.866025, 0.5     ],
       [2.      , 1.      ]])

# [1. Mesh Generation] --------------------------------------
# generate vertices within border
border = len(input) - 1
grid = np.copy(input)
centroid = grid.mean(axis=0)  # get center point value of the border
grid = np.vstack((grid, centroid))

for i in range(border):
    x = (input[i][0] + centroid[0]) / 2  # from border to the center, generate half length points
    y = (input[i][1] + centroid[1]) / 2
    grid = np.vstack((grid, (x, y)))

# triangulation
tri = Delaunay(grid)  # generate triangles from the vertex points (x and y only)
z = np.zeros((int(len(grid)), 1))  # get a list of 0's at the length of V
v = np.append(grid, z, axis=1)  # add the list of Z's axis of 0's on V's X and Y
f = tri.simplices  # generate a triangle face out of the new vertex list

# clean faces generated outside border
for i in range(len(v) - 1):
    if f[i][0] <= border and f[i][1] <= border and f[i][2] <= border:  # if faces xyz is <= border value
        f = np.delete(f, i, axis=0)  # delete bc face is outside border

# [2. Mesh Inflation] --------------------------------------
# the border value is used to inflate remainings
for i in range(len(v)):
    if i > border:  # if increment of i is more than 11
        v[i][2] = 0.5  # row 12, column 2

zvertices = np.copy(v)
zface = np.copy(f)

# reverse vertices
for i in range(len(zvertices)):
    zvertices[i][2] = zvertices[i][2] - (zvertices[i][2] + zvertices[i][2])

# reverse face normals
for i in range(len(zface)):
    dummy = np.zeros(shape=(1, 3))
    dummy[0][0] = zface[i][0]
    zface[i][0] = zface[i][2]
    zface[i][2] = dummy[0][0]

# update vertex indices pointed to the newly stacked values
for i in range(len(zface)):
    zface[i] = zface[i] + len(v)

# join the vertices
v = np.vstack([v, zvertices])
f = np.vstack([f, zface])

duplicatedborder = round(int(len(v) / 2))  # 27, 54 vertexes in total.
facecounter = int(0)  # points to the original first 11 values

# go through shared vertex points and update faces pointing to them (original first 11 values)
for i in range(duplicatedborder,
               duplicatedborder + border + 1):  # vertex range from 0 to 11 (12?), num of loops     -> correct
    for j in range(len(f)):  # compare vertex values i with f's entire column of
        if i == f[j][0]:  # x axis
            f[j][0] = facecounter
        elif i == f[j][1]:  # y axis
            f[j][1] = facecounter
        elif i == f[j][2]:  # z axis
            f[j][2] = facecounter

    facecounter = facecounter + 1  # add from 0 and then increment vertex indice to 28

# delete unecessary vertices
duplicatedborder2 = round(int(len(v) / 2)) + 1
if v[duplicatedborder2][0] == v[1][0] and v[duplicatedborder2][1] == v[1][1]:
    v = np.delete(v, np.s_[int(duplicatedborder2) - 1:int(duplicatedborder) + border + 1], axis=0)
    # same thing but with the indices and the [] position

oldborder = int(((len(v)) + border) / 2) + 1  # do not round this

# final index updating                       -> this thing deletes the final 11 faces
for i in range(oldborder, len(v) + 1):  # 25 to 36
    for j in range(len(f)):  # compare values i with f's entire list
        if i + border + 1 == f[j][0]:  # x axis
            f[j][0] = i
        elif i + border + 1 == f[j][1]:  # y axis
            f[j][1] = i
        elif i + border + 1 == f[j][2]:  # z axis
            f[j][2] = i
# this should be 37 verts but has 48
# 6 -> 11's values are being overlapped by 42-47 (they've moved for some reason), 6 values

ret = igl.write_triangle_mesh(os.path.join(root_folder, "data", "1.obj"), v, f)

# [3. Geometry Smoothing] --------------------------------------
l = igl.cotmatrix(v, f)  # laplacian? lambda?
n = igl.per_vertex_normals(v, f) * 0.5 + 0.5
c = np.linalg.norm(n, axis=1)
vs = [v]  # vertices source
cs = [c]  # cotangent source?

for i in range(100):
    m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    s = (m - 0.001 * l)  # massmatrix
    b = m.dot(v)
    v = spsolve(s, m.dot(v))
    n = igl.per_vertex_normals(v, f) * 0.5 + 0.5
    c = np.linalg.norm(n, axis=1)  # cotangent
    vs.append(v)
    cs.append(c)

p = subplot(vs[0], f, c, shading={"wireframe": False}, s=[1, 4, 0])
writeMesh = igl.write_triangle_mesh(os.path.join(root_folder, "data", "1.obj"), vs[100], f)

# [4. Geometry Deformation] --------------------------------------
vv, ff = igl.read_triangle_mesh(os.path.join(root_folder, "data", "1.obj"))
#vv[:,[0, 2]] = vv[:,[2, 0]] # Swap X and Z axes
u = vv.copy()

#create a .dmat file
borderToMove = 3	#input
dmatTop = [1, int(len(vv))]
dmatArray = []
for i in range(len(vv)):
    dmatArray.insert(i, 0)

for i in range(len(vv)):
    if i == int(borderToMove):
        dmatArray[i] = 1
        dmatArray[i+1] = -1
        dmatArray[i-1] = -1

dmatCleanTop = str(dmatTop).strip('[]').replace(', ' , ' ')
dmatArray.insert(0, dmatCleanTop)
dmatData = '\n'.join(str(line) for line in dmatArray)
fname = os.path.join(root_folder, "data", "decimated-max-selection1.dmat")
myfile = open(fname,"w")
myfile.write(str(dmatData))
myfile.close()

# Read .dmat values to move plots
s = igl.read_dmat(os.path.join(root_folder, "data", "decimated-max-selection1.dmat"))
b = np.array([[t[0] for t in [(i, s[i]) for i in range(0, v.shape[0])] if t[1] >= 0]]).T

u_bc = np.zeros((b.shape[0], v.shape[1]))
v_bc = np.zeros((b.shape[0], v.shape[1]))

for bi in range(b.shape[0]):
    v_bc[bi] = vv[b[bi]]

    if s[b[bi]] == 0: # Don't move handle 0
        u_bc[bi] = vv[b[bi]]
    elif s[b[bi]] == 1: # Move handle 1 (+ up, - down) (forward/backward, up/down, left/right)
        u_bc[bi] = vv[b[bi]] + np.array([[-0.5, 0.5, 0]])
    else: # Move other handles forward
        u_bc[bi] = vv[b[bi]] + np.array([[-0.25, 0.25, 0]])

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