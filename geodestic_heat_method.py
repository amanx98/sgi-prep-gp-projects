import numpy as np
import trimesh
import gpytoolbox as gpy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

np.set_printoptions(precision=4, suppress=True)

mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
V, F = mesh.vertices, mesh.faces
n, m = V.shape[0], F.shape[0]
print(f"{n} vertices, {m} faces")

# source = the vertex closest to the "north pole"
source = int(np.argmax(V[:, 2]))
print(f"source vertex index: {source}, position: {V[source]}")