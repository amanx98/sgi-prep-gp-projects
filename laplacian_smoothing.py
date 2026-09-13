"""
SGI Prep Project 1: Mesh Laplacian Smoothing + Curvature Estimation
---------------------------------------------------------------------
Goal: implement the cotangent Laplacian from scratch, use it to (a) smooth
a noisy mesh and (b) estimate curvature, and sanity-check both against a
known ground truth (a sphere has constant mean curvature).

Why a noisy icosphere: a perfect sphere has mean curvature = 1/radius
everywhere (constant), and Gaussian curvature = 1/radius^2 everywhere.
That gives us a ground truth to validate against -- if your curvature
numbers don't converge toward that as noise decreases, something in the
Laplacian construction is wrong. This is a good habit for the geodesic
project too: always have a check, not just a "does it look ok" glance.
"""

import numpy as np
import trimesh
import gpytoolbox as gpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def make_noisy_icosphere(subdivisions=3, radius=1.0, noise_std=0.03, seed=0):
    """Create an icosphere and perturb vertices along their normals."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    V = mesh.vertices.copy()
    F = mesh.faces.copy()

    rng = np.random.default_rng(seed)
    normals = V / np.linalg.norm(V, axis=1, keepdims=True)
    noise = rng.normal(0, noise_std, size=(V.shape[0], 1))
    V_noisy = V + noise * normals
    return V_noisy, F, V  # return clean V too, as ground truth


def cotangent_laplacian(V, F):
    """
    Build the symmetric cotangent Laplacian L (V x V sparse matrix) and
    the diagonal mass matrix M, using gpytoolbox's implementation.
    Doing this via gpytoolbox rather than fully by hand for now --
    once this pipeline works end to end, come back and hand-derive the
    per-triangle cotangent weights yourself so it's not a black box.
    """
    L = gpy.cotangent_laplacian(V, F)
    M = gpy.massmatrix(V, F)
    return L, M


def smooth_mesh(V, F, L, M, num_iters=1, step=0.5, implicit=True):
    """
    Laplacian smoothing. Implicit (backward Euler) is unconditionally
    stable -- won't blow up regardless of step size, unlike explicit
    smoothing which can. That stability is exactly why implicit is the
    standard choice in geometry processing.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    V_smooth = V.copy()
    n = V.shape[0]

    
    if implicit:
        # (M + step*L) V_new = M V_old
        A = M + step * L
        for _ in range(num_iters):
            for d in range(3):
                V_smooth[:, d] = spla.spsolve(A.tocsc(), M @ V_smooth[:, d])
    else:
        Minv = sp.diags(1.0 / M.diagonal())
        for _ in range(num_iters):
            V_smooth = V_smooth - step * (Minv @ (L @ V_smooth))

    return V_smooth


def mean_curvature(V, F, L, M):
    """
    Mean curvature via the Laplace-Beltrami operator applied to vertex
    positions: (1/2) * |M^-1 L V| gives mean curvature magnitude at each
    vertex (this is the classic "mean curvature normal" identity).
    """
    import scipy.sparse as sp
    # Same sign-convention note as above: with gpytoolbox's positive
    # semi-definite L (= -Delta), the identity Delta x = 2 H n becomes
    # L x = -2 H n. We only need the magnitude here so the sign doesn't
    # matter for H itself, but it would matter if you used the normal
    # direction n later (e.g. to color inward vs outward curving regions).
    Minv = sp.diags(1.0 / M.diagonal())
    HN = Minv @ (L @ V)  # mean curvature normal vectors, per vertex
    H = 0.5 * np.linalg.norm(HN, axis=1)
    return H


def plot_comparison(V_noisy, V_smooth, F, V_clean, H, out_path):
    fig = plt.figure(figsize=(15, 5))

    for i, (V, title) in enumerate([
        (V_noisy, "Noisy input"),
        (V_smooth, "After implicit smoothing"),
        (V_clean, "Ground truth sphere"),
    ]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F,
                         cmap="viridis", edgecolor="none", alpha=0.9)
        ax.set_title(title)
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved comparison figure to {out_path}")


if __name__ == "__main__":
    print("Generating noisy icosphere...")
    V_noisy, F, V_clean = make_noisy_icosphere(subdivisions=3, radius=1.0, noise_std=0.03)
    print(f"  {V_noisy.shape[0]} vertices, {F.shape[0]} faces")

    print("Building cotangent Laplacian...")
    L, M = cotangent_laplacian(V_noisy, F)

    print("Smoothing (implicit, 3 iterations)...")
    V_smooth = smooth_mesh(V_noisy, F, L, M, num_iters=3, step=0.001, implicit=True)

    # sanity check: smoothing should reduce distance-to-sphere-surface error
    err_before = np.mean(np.abs(np.linalg.norm(V_noisy, axis=1) - 1.0))
    err_after = np.mean(np.abs(np.linalg.norm(V_smooth, axis=1) - 1.0))
    print(f"\nMean radial error before smoothing: {err_before:.5f}")
    print(f"Mean radial error after smoothing:  {err_after:.5f}")
    print("(should be smaller after -- smoothing pulled points back toward the true sphere)")

    print("\nEstimating mean curvature on smoothed mesh...")
    L2, M2 = cotangent_laplacian(V_smooth, F)
    H = mean_curvature(V_smooth, F, L2, M2)
    print(f"Mean curvature: mean={H.mean():.4f}, std={H.std():.4f}")
    print("(ground truth for unit sphere: mean curvature = 1.0 everywhere)")

    plot_comparison(V_noisy, V_smooth, F, V_clean, H, "comparison.png")
