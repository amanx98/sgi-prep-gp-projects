# SGI Prep: Geometry Processing Projects

A hands-on collection of discrete differential geometry and 3D geometry processing implementations built from first principles on triangle meshes and point clouds.

This repository tracks a 5-part project series designed for Summer Geometry Initiative (SGI) preparation, progressing from fundamental differential operators to geodesic distance fields, registration, and spectral analysis.

---

## 📌 Project Roadmap

- [x] **Project 1: Mesh Laplacian Smoothing & Curvature Estimation**
  - Constructing discrete cotangent Laplacian ($L$) and lumped mass ($M$) matrices.
  - Solving implicit (backward Euler) heat diffusion for unconditionally stable surface denoising.
  - Discrete mean curvature estimation ($H$) via the Laplace-Beltrami operator validated against analytical ground truth on a sphere.
- [ ] **Project 2: Geodesic Distance via the Heat Method**
  - Short-time heat diffusion on surface meshes ($u_t = \Delta u$).
  - Vector field extraction and normalization ($X = -\nabla u / \|\nabla u\|$).
  - Solving the Poisson equation ($L\phi = \text{div}(X)$) to retrieve all-pairs-to-source geodesic distances without graph traversal.
- [ ] **Project 3: Rigid Point Cloud Registration (ICP)**
  - Point-to-point and point-to-plane Iterative Closest Point (ICP).
  - SVD-based optimal rotation and translation estimation.
- [ ] **Project 4: Parameterization & Harmonic Maps** *(Upcoming)*
- [ ] **Project 5: Spectral Shape Analysis & Geodesics** *(Upcoming)*

---

## 🛠️ Project 1 Details: Laplacian Smoothing & Curvature Estimation

### Key Mathematical Principles
1. **Discrete Cotangent Laplacian ($L$):**
   Approximates the continuous Laplace-Beltrami operator $\Delta$ across discrete triangle meshes:
   $$(Lf)_i = \sum_{j \in \mathcal{N}(i)} \frac{\cot\alpha_{ij} + \cot\beta_{ij}}{2} (f_j - f_i)$$
   *Note on sign convention:* Evaluated as positive semi-definite ($L = -\Delta$) to maintain consistency with `gpytoolbox` and sparse Cholesky/LU solvers[cite: 1].

2. **Implicit Smoothing (Backward Euler):**
   Solves the linear system per spatial coordinate $d \in \{x, y, z\}$:
   $$(M + tL) V_{\text{new}} = M V_{\text{old}}$$
   Guarantees unconditional numerical stability regardless of the step size $t$[cite: 1].

3. **Mean Curvature Estimation:**
   Leverages the differential geometric identity relating surface position $\mathbf{x}$ to mean curvature $H$ and surface normal $\mathbf{n}$:
   $$\Delta \mathbf{x} = 2H\mathbf{n} \implies H = \frac{1}{2} \|M^{-1} L V\|$$
   Benchmarked against a unit sphere where $H = 1.0$ everywhere[cite: 1].

---

## 📦 Requirements & Environment Setup

### Prerequisites
- Python 3.10+
- A virtual environment tool (`venv` or `conda`)

### Installation
Clone the repository:
```bash
git clone [https://github.com/amanx98/sgi-prep-gp-projects.git](https://github.com/amanx98/sgi-prep-gp-projects.git)
cd sgi-prep-gp-projects