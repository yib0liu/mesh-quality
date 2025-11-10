import numpy as np
import json
import meshio

try:
	with open("../polyfem/data/standard/neohookean2.json") as f:
		param=json.load(f)
except:
    param=None

if param:
	E=param["materials"]["E"]
	nu=param["materials"]["nu"]
else:
	E=20000
	nu=0.3
      
x,y,z=4,3,1
force=100

L=x
A=x*z
P=force * A  # ?
k=5/6
G=E/(2*(1+nu))  # ?
I=y*z**3/12  # ?

def W_L(P,L,k,A,G,E,I):
	return P*L/(k*A*G)+P*L**3/(3*E*I)

wL_timo=W_L(P,L,k,A,G,E,I)

### 
# vtu_path = "../simout/linear/cube/sim_surf.vtu"
vtu_path = "../simout/linear/prism/sim_surf.vtu"
print(vtu_path)

m = meshio.read(vtu_path)
pts = m.points
sol = m.point_data["solution"]  # (N, 3)
fid = m.point_data["sidesets"]

# print(sol.shape)
# print(fid.shape)
# print(fid)

# print(pts[:,2].mean())

li=[]
for i in range(len(sol)):
	if fid[i]==3:
		li.append(sol[i])
li=np.array(li)
# print(li.shape)
# print(li)
wL_sim=li[:,2].mean()

print(f"Timo: {wL_timo:.3f}")
print(f"Sim: {wL_sim:.3f}")
print(f"err: {wL_timo-wL_sim:.3f}")
