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


L = gpy.cotangent_laplacian(V, F)
M = gpy.massmatrix(V, F)
G = gpy.grad(V, F)
face_areas = mesh.area_faces

# Confirm the block-stacking convention: grad(x-coordinate) must be
# tangent to the surface, i.e. orthogonal to every face normal.
gx_flat = G @ V[:, 0]
grad_x = np.stack([gx_flat[0:m], gx_flat[m:2*m], gx_flat[2*m:3*m]], axis=1)
check = np.abs(np.sum(grad_x * mesh.face_normals, axis=1)).max()
print(f"max |grad(x) . normal| = {check:.2e}  (should be ~0 if the block-stack ordering assumption is right)")

A3 = sp.diags(np.tile(face_areas, 3))
L_from_G = G.T @ A3 @ G
max_diff = np.abs((L_from_G - L)).max()
print(f"max |G^T diag(A3) G - L| = {max_diff:.2e}  (should be ~0)")

# time step scaled to the mesh's edge length, per the heat-method paper's guidance
edges = mesh.edges_unique
h = np.mean(np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1))
t = 10 * h**2

u0 = np.zeros(n)
u0[source] = 1.0

A = (M + t * L).tocsc()
u = spla.spsolve(A, M @ u0)
print(f"heat value range after diffusion: [{u.min():.4f}, {u.max():.4f}]")

# gradient of u, per face
gu_flat = G @ u
grad_u = np.stack([gu_flat[0:m], gu_flat[m:2*m], gu_flat[2*m:3*m]], axis=1)

# normalize and negate -> unit vector field pointing away from the source
norms = np.linalg.norm(grad_u, axis=1, keepdims=True)
norms[norms < 1e-12] = 1e-12   # avoid divide-by-zero on degenerate faces
X = -grad_u / norms

# divergence of X as a Poisson right-hand side, using the identity confirmed above
X_stacked = np.concatenate([X[:, 0], X[:, 1], X[:, 2]])
b = G.T @ (A3 @ X_stacked)

# solve L*phi = b, pinning one row to fix the additive constant
L_mod = L.tolil()
L_mod[0, :] = 0
L_mod[0, 0] = 1
b_mod = b.copy()
b_mod[0] = 0

phi = spla.spsolve(L_mod.tocsc(), b_mod)
phi = phi - phi[source]   # distance from the source itself = 0
print(f"distance range: [{phi.min():.4f}, {phi.max():.4f}]")

cos_angle = np.clip(V @ V[source], -1, 1)
true_dist = np.arccos(cos_angle)   # great-circle distance on a unit sphere

err = np.abs(phi - true_dist)
print(f"mean absolute error: {err.mean():.4f} radians")
print(f"max absolute error:  {err.max():.4f} radians")
print(f"(for reference, the sphere's max possible distance, pole-to-pole, is pi = {np.pi:.4f})")

def heat_geodesic(V, F, source, t_mult=10.0):
    # Full pipeline, parameterized by t_mult, for easy experimentation.
    n, m = V.shape[0], F.shape[0]
    L = gpy.cotangent_laplacian(V, F)
    M = gpy.massmatrix(V, F)
    G = gpy.grad(V, F)
    mesh_tmp = trimesh.Trimesh(vertices=V, faces=F, process=False)
    face_areas = mesh_tmp.area_faces
    A3 = sp.diags(np.tile(face_areas, 3))

    edges = mesh_tmp.edges_unique
    h = np.mean(np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1))
    t = t_mult * h**2

    u0 = np.zeros(n); u0[source] = 1.0
    u = spla.spsolve((M + t * L).tocsc(), M @ u0)

    gu_flat = G @ u
    grad_u = np.stack([gu_flat[0:m], gu_flat[m:2*m], gu_flat[2*m:3*m]], axis=1)
    norms = np.linalg.norm(grad_u, axis=1, keepdims=True); norms[norms < 1e-12] = 1e-12
    X = -grad_u / norms
    X_stacked = np.concatenate([X[:, 0], X[:, 1], X[:, 2]])
    b = G.T @ (A3 @ X_stacked)

    L_mod = L.tolil(); L_mod[0, :] = 0; L_mod[0, 0] = 1
    b_mod = b.copy(); b_mod[0] = 0
    phi = spla.spsolve(L_mod.tocsc(), b_mod)
    return phi - phi[source]

for t_mult in [1, 5, 10, 20]:
    phi_t = heat_geodesic(V, F, source, t_mult=t_mult)
    err_t = np.abs(phi_t - true_dist)
    print(f"t_mult={t_mult:>3}:  mean_err={err_t.mean():.4f}   max_err={err_t.max():.4f}")

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
p1 = ax1.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F,
                       cmap="viridis", edgecolor="none")
p1.set_array(phi[F].mean(axis=1))
ax1.set_title("Heat-method geodesic distance")
ax1.set_box_aspect([1, 1, 1]); ax1.axis("off")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
p2 = ax2.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F,
                       cmap="viridis", edgecolor="none")
p2.set_array(true_dist[F].mean(axis=1))
ax2.set_title("Ground truth (great-circle distance)")
ax2.set_box_aspect([1, 1, 1]); ax2.axis("off")

plt.tight_layout()
plt.show()

