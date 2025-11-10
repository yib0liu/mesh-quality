import sys
# https://github.com/yib0liu/legacy-quadpy commit 5213fed9155e3cf0e62900a7483ecf9abb23d48a
sys.path.insert(0, "/home/liuyibo/myrepo/prism/legacy-quadpy/src/")

import numpy as np
from numpy.polynomial.legendre import leggauss
import sympy as sp
import quadpy
import plotly.graph_objects as go
import meshio
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ref_prism =np.array(
            [[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 1], [0, 1, 1]]],
            dtype=float,
        )
ref_tet = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
ref_pyramid = np.array([
    [0,0,0],
    [0,1,0],
    [1,1,0],
    [1,0,0],
    [0,0,1]
])

def barycentric_to_cartesian(bary_points, vertices):
    xyz = vertices.T @ bary_points
    return xyz.T

# def vsample(point_xyz,ctype,physical_nodes: np.array):    
#     px,py,pz=point_xyz

#     x, y, z = sp.symbols('x y z', real=True)

#     prism_phi = [
#         (1 - x - y)*(1 - z),
#         x*(1 - z),
#         y*(1 - z),
#         (1 - x - y)*z,
#         x*z,
#         y*z
#     ]
    
#     pyramid_phi = [
#         (-x*y+(z-1)*(-x-y-z+1))/(z-1),
#         x*(y+z-1)/(z-1),
#         y*(x+z-1)/(z-1),
#         -x*y/(z-1),
#         z
#     ]
    
#     tet_phi = [
#         -x-y-z+1,
#         x,
#         y,
#         z
#     ]
    
#     phis=None
#     if ctype=="Prism":
#         phis=prism_phi
#     elif ctype=="Pyramid":
#         phis=pyramid_phi
#     else:
#         phis=tet_phi

#     dphidx = [sp.diff(phi, x) for phi in phis]
#     dphidy = [sp.diff(phi, y) for phi in phis]
#     dphidz = [sp.diff(phi, z) for phi in phis]

#     dphidx_funcs = [sp.lambdify((x, y, z), dphi, "numpy") for dphi in dphidx]
#     dphidy_funcs = [sp.lambdify((x, y, z), dphi, "numpy") for dphi in dphidy]
#     dphidz_funcs = [sp.lambdify((x, y, z), dphi, "numpy") for dphi in dphidz]

#     dxdx = np.zeros((3,3))
#     for i in range(len(phis)):
#         grad_phi_i=np.array([
#                 dphidx_funcs[i](px, py, pz),
#                 dphidy_funcs[i](px, py, pz),
#                 dphidz_funcs[i](px, py, pz)
#             ])  
#         dxdx += np.outer(physical_nodes[i], grad_phi_i)

#     detJ = np.linalg.det(dxdx)
#     # print(detJ)

#     return detJ


def evalcell(phys_pts, ctype="Tet",test=False):
    
    if ctype=="Prism":
        scheme = quadpy.w3.felippa_3()
        points=scheme.transform(phys_pts)[0]
        weights=scheme.transform_w(phys_pts)

    elif ctype=="Pyramid":
        scheme = quadpy.p3.felippa_5()
        points=scheme.transform(phys_pts)
        weights=scheme.transform_w(phys_pts)

    elif ctype=="Tet":
        scheme = quadpy.t3.get_good_scheme(3)
        points=scheme.points
        weights=scheme.weights
        points=barycentric_to_cartesian(points,phys_pts)

    x, y, z = sp.symbols('x y z')
    poly = (x**0 + y**0 + z**0) / 3
    Vsamples=np.array([weights[i] * poly.subs([(x, points[i, 0]),
                           (y, points[i, 1]), (z, points[i, 2])]) for i in range(len(points))])
    # Vsamples=np.array([vsample(points[i],ctype,phys_pts) * weights[i] for i in range(len(points))])
    V=sum(Vsamples) 
    if ctype=="Prism":
        V = V
    elif ctype=="Tet":
        V = V/6
    elif ctype=="Pyramid":
        V = V

    if test:
        print(scheme)
        print(points.shape)
        print(points)
        print(weights.shape)
        print(weights)
        print("sum(weights) =", sum(scheme.weights))
        print("polynomial: ",poly)
        print("Vsmaple: ", Vsamples)
        print("V: ",V)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        if ctype=="Pyramid":
            faces = [
                [phys_pts[0], phys_pts[1], phys_pts[2], phys_pts[3]],
                [phys_pts[0], phys_pts[1], phys_pts[4]],        
                [phys_pts[1], phys_pts[2], phys_pts[4]],
                [phys_pts[2], phys_pts[3], phys_pts[4]],
                [phys_pts[3], phys_pts[0], phys_pts[4]]
            ]

        else:
            faces = []

        poly3d = Poly3DCollection(faces, alpha=0.2, edgecolor="k", facecolor="cyan")
        ax.add_collection3d(poly3d)

        # ax.scatter(a[:, 0], a[:, 1], a[:, 2], color="r", s=60, label="Magic points")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="b", s=60, label="Quadrature points")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_zlim(-1.2, 1.2)
        ax.legend()
        ax.view_init(elev=20, azim=35)
        plt.tight_layout()
        plt.show()

    a=False
    b=False
    if V>0:
        a=True
    if np.all(Vsamples>0):
        b=True
    return a,b

if __name__ == "__main__":
    test=False

    if test:
        a,b=evalcell(ref_pyramid, "Pyramid",test=True)
        a,b=evalcell(ref_tet, "Tet",test=True)
        a,b=evalcell(ref_prism, "Prism",test=True)
        mesh = meshio.read(f"data2/output1.msh_new",file_format="gmsh")
        nodes=mesh.points
        points=nodes[mesh.cells_dict["wedge"][0]].reshape(2,3,3)
        a,b=evalcell(points, "Prism",test=True)

    else:
        for n in [1,2,4]:
            cnt1=0
            cnt2=0
            cnt3=0
            cnt4=0
            cnt5=0
            cnt6=0
            mesh = meshio.read(f"data2/output{n}.msh_new",file_format="gmsh")
            print(f"file: data2/output{n}.msh_new")
            nodes=mesh.points
            
            for elem in tqdm(mesh.cells_dict["tetra"]):
                a,b=evalcell(nodes[elem], "Tet")
                cnt1+=a
                cnt4+=b
            print("Tet")
            print("V>0 ", round(100*cnt1/len(mesh.cells_dict["tetra"])), "%")
            print("all det>0 ", round(100*cnt4/len(mesh.cells_dict["tetra"])), "%")
            
            for elem in tqdm(mesh.cells_dict["pyramid"]):
                a,b=evalcell(nodes[elem], "Pyramid")
                cnt2+=a
                cnt5+=b
            print("Pyramid")
            print("V>0 ", round(100*cnt2/len(mesh.cells_dict["pyramid"])), "%")
            print("all det>0 ", round(100*cnt5/len(mesh.cells_dict["pyramid"])), "%")
            
            for elem in tqdm(mesh.cells_dict["wedge"]):
                a,b=evalcell(nodes[elem].reshape(2,3,3), "Prism")
                cnt3+=a
                cnt6+=b
            print("Prism")
            print("V>0 ", round(100*cnt3/len(mesh.cells_dict["wedge"])), "%")
            print("all det>0 ", round(100*cnt6/len(mesh.cells_dict["wedge"])), "%")
            
            print("Overall")
            print("V>0 ",round(100*(cnt1+cnt2+cnt3)/(len(mesh.cells_dict["wedge"])+len(mesh.cells_dict["tetra"])+len(mesh.cells_dict["pyramid"]))), "%")        
            print("all det>0 ", round(100*(cnt4+cnt5+cnt6)/(len(mesh.cells_dict["wedge"])+len(mesh.cells_dict["tetra"])+len(mesh.cells_dict["pyramid"]))), "%")