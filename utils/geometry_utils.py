import igl
import numpy as np

from scipy.sparse import diags, coo_matrix, identity
from scipy.sparse import csc_matrix as sp_csc
import scipy.sparse.linalg as sla
import potpourri3d as pp3d
import trimesh
import pyvista as pv
import torch

def compute_laplace_beltrami(vertices, faces):
    grad = igl.grad(vertices, faces)
    d_area = igl.doublearea(vertices, faces)
    d_area = diags(np.hstack([d_area, d_area, d_area]) * 0.5)
    mass = sp_csc(d_area)

    LBO = grad.T @ mass @ grad

    return LBO, grad, mass

def compute_finsler_laplace_beltrami(vertices, faces, alpha=0.0, tau=0.0):
    """
    Compute the Finsler-Laplace-Beltrami operator using Randers metric.

    Parameters:
    - vertices: mesh vertices
    - faces: mesh faces
    - alpha: anisotropy parameter
    - tau: parameter for the asymmetric component

    Returns:
    - FLBO: Finsler-Laplace-Beltrami operator
    """

    b1, b2, normals = igl.local_basis(vertices, faces)

    grad = igl.grad(vertices, faces)
    d_area = igl.doublearea(vertices, faces)
    d_area = diags(np.hstack([d_area, d_area, d_area]) * 0.5)
    mass = sp_csc(d_area)

    u_M, u_m, k1, k2 = igl.principal_curvature(vertices, faces)

    u_M = igl.average_onto_faces(faces, u_M)
    #u_m = igl.average_onto_faces(faces, u_m)

    # Define the anisotropic Riemannian metric using eq. (35)
    D_alpha = np.zeros((faces.shape[0], 2, 2))
    D_alpha[:, 0, 0] = 1 / (1 + alpha)
    D_alpha[:, 1, 1] = 1

    # For each face, compute the shear matrix H_alpha_theta using eq. (36)
    H = np.zeros((faces.shape[0], 3, 3))
    M_inv = np.zeros((faces.shape[0], 3, 3))

    for i in range(faces.shape[0]):
        # Create local frame
        frame = np.column_stack([b1[i], b2[i], normals[i]])

        # Extend D_alpha to 3D
        D3 = np.eye(3)
        D3[0:2, 0:2] = D_alpha[i]

        # Compute H_alpha_theta = frame @ D_alpha @ frame.T
        H[i] = frame @ D3 @ frame.T

        # M is inverse of H (eq. 37)
        M_inv[i] = np.linalg.inv(H[i])

    # Define the asymmetric component omega using eq. (38)
    omega = tau * u_M

    # Compute dual metric components
    M_star = np.zeros_like(M_inv)
    omega_star = np.zeros_like(omega)

    for i in range(faces.shape[0]):
        # Compute alpha_i = 1 - <omega, M^-1 omega>
        omega_i = omega[i]

        # When tau=0, omega_i is all zeros, so this gives alpha_i = 1
        alpha_i = 1 - omega_i @ M_inv[i] @ omega_i
        alpha_i = max(alpha_i, 1e-16)  # Safety check

        # Compute M* using equation (22)
        M_inv_omega = M_inv[i] @ omega_i

        # When tau=0, M_inv_omega is zero, so this simplifies to M* = M^-1
        M_star[i] = (1 / alpha_i ** 2) * (alpha_i * M_inv[i] + np.outer(M_inv_omega, M_inv_omega))

        # Compute omega* using equation (23)
        # When tau=0, this gives omega* = 0
        omega_star[i] = -(1 / alpha_i) * M_inv[i] @ omega_i

    # Compute Finsler diffusivity D_F* = M* - omega* omega*^T
    D_F = np.zeros_like(M_star)
    for i in range(faces.shape[0]):
        # When tau=0, omega_star is zero, so D_F = M*
        # When alpha=0 and tau=0, M* = M^-1 = identity
        D_F[i] = M_star[i] - np.outer(omega_star[i], omega_star[i])

    # Construct the diffusivity matrix
    n = grad.shape[0]  # This should be 3 * number of faces

    # Build a block diagonal matrix with D_F blocks - fixed indexing
    rows, cols, data = [], [], []
    for i in range(faces.shape[0]):
        for di in range(3):
            for dj in range(3):
                # Correctly calculate the indices in the gradient space
                row_idx = i * 3 + di
                col_idx = i * 3 + dj

                if row_idx < n and col_idx < n:
                    rows.append(row_idx)
                    cols.append(col_idx)
                    # Use the diffusivity tensor for this face
                    data.append(D_F[i, di, dj])

    diffusivity = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()

    # Construct the Finsler-Laplace-Beltrami operator
    FLBO = grad.T @ mass @ diffusivity @ grad

    return FLBO, grad, mass, diffusivity


def spectral_solve(L, massvec_np, k_eig=8, eps=1e-12):
    # === Compute the eigenbasis
    massvec_np += eps * np.mean(massvec_np)
    if k_eig > 0:

        # Prepare matrices
        L_eigsh = (L + identity(L.shape[0]) * eps).tocsc()
        massvec_eigsh = massvec_np
        Mmat = diags(massvec_eigsh)
        eigs_sigma = eps

        failcount = 0
        while True:
            try:
                # We would be happy here to lower tol or maxiter since we don't need these to be super precise, but for some reason those parameters seem to have no effect
                evals_np, evecs_np = sla.eigsh(L_eigsh, k=k_eig, M=Mmat, sigma=eigs_sigma, tol=1e-20)

                # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
                evals_np = np.clip(evals_np, a_min=0., a_max=float('inf'))

                break
            except Exception as e:
                print(e)
                if (failcount > 3):
                    raise ValueError("failed to compute eigendecomp")
                failcount += 1
                print("--- decomp failed; adding eps ===> count: " + str(failcount))
                L_eigsh = L_eigsh + identity(L.shape[0]) * (eps * 10 ** failcount)

    return evals_np, evecs_np


def consistent_spectral_decomposition(L, M, num_eigenvectors):
    """Compute consistent eigenvectors of the generalized eigenvalue problem."""
    # Set seed for repeatability
    np.random.seed(42)

    # Create initial vector (ones vector normalized by mass matrix)
    v0 = np.ones(L.shape[0])
    if isinstance(M, np.ndarray):
        M_diag = diags(M)
    else:
        M_diag = M
    v0 = v0 / np.sqrt(v0.T @ (M_diag @ v0))

    # Solve eigenvalue problem
    eigenvalues, eigenvectors = sla.eigsh(L, k=num_eigenvectors, M=M_diag,
                                      sigma=1e-8, which='LM', v0=v0)

    # Sort by eigenvalues
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Enforce consistent orientation
    for i in range(eigenvectors.shape[1]):
        max_idx = np.argmax(np.abs(eigenvectors[:, i]))
        if eigenvectors[max_idx, i] < 0:
            eigenvectors[:, i] = -eigenvectors[:, i]

    return eigenvalues, eigenvectors


def spectral_graph(evecs_np, V, M):

    M = diags(M)

    return evecs_np.T @ M @ V


def reconstruct_mesh_from_spectral_coeffs(eigvecs, coeffs):

    return eigvecs @ coeffs


def sbs_view(mesh1, mesh2):
    # Create PyVista meshes for visualization
    faces1 = np.hstack((np.full((mesh1.faces.shape[0], 1), 3, dtype=np.int64), mesh1.faces))
    faces2 = np.hstack((np.full((mesh2.faces.shape[0], 1), 3, dtype=np.int64), mesh2.faces))

    original_mesh = pv.PolyData(mesh1.vertices, faces1)
    reconstructed_mesh = pv.PolyData(mesh2.vertices, faces2)

    # Initialize the PyVista plotter with two subplots
    plotter = pv.Plotter(shape=(1, 2), window_size=(800, 400))

    # Plot the original mesh in the first subplot
    plotter.subplot(0, 0)
    plotter.add_mesh(original_mesh, color='lightblue', show_edges=True)
    plotter.add_text("Mesh 1", font_size=10)

    # Plot the reconstructed mesh in the second subplot
    plotter.subplot(0, 1)
    plotter.add_mesh(reconstructed_mesh, color='lightgreen', show_edges=True)
    plotter.add_text("Mesh 2", font_size=10)

    # Display the visualization
    plotter.show()


def compute_face_normals(vertices, faces):
    """
    Compute face normals for triangular mesh.

    Args:
        vertices: (N, 3) array of vertex positions
        faces: (F, 3) array of face indices

    Returns:
        (F, 3) array of normalized face normals
    """
    # Get triangle vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute face normals using cross product
    normals = np.cross(v1 - v0, v2 - v0)

    # Normalize
    norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norm_lengths + 1e-12)  # Add small epsilon to avoid division by zero

    return normals


def compute_vertex_normals(vertices, faces):

    n_vertices = len(vertices)
    vertex_normals = np.zeros((n_vertices, 3))

    # Get triangle vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute face normals and areas
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.linalg.norm(face_normals, axis=1)

    # Normalize face normals
    face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12)

    # Accumulate area-weighted normals at vertices
    for i in range(len(faces)):
        for j in range(3):
            vertex_idx = faces[i, j]
            vertex_normals[vertex_idx] += face_areas[i] * face_normals[i]

    # Normalize vertex normals
    vertex_normals = vertex_normals / (np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-12)

    return vertex_normals

def compute_vertex_areas(vertices, faces):
    """
    Compute area associated with each vertex using Voronoi area approximation.

    Args:
        vertices: (N, 3) array of vertex positions
        faces: (F, 3) array of face indices

    Returns:
        (N,) array of vertex areas
    """
    n_vertices = len(vertices)
    vertex_areas = np.zeros(n_vertices)

    # Get triangle vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute face areas
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    # Distribute 1/3 of each face area to each vertex (simple approximation)
    for i in range(len(faces)):
        for j in range(3):
            vertex_areas[faces[i, j]] += face_areas[i] / 3.0

    return vertex_areas

def face_to_vertex_signal(V, F, f, weights="area"):
    """
    Project a piecewise-constant per-face signal onto vertices (piecewise-linear),
    by computing a weighted average of incident face values at each vertex.

    Parameters
    ----------
    V : (N, 3) float
        Vertex positions.
    F : (M, 3) int
        Triangulation (indices into V).
    f : (M,) or (M, K) float
        Per-face signal (scalar or K-channel).
    weights : {"area", "uniform"} or array-like of shape (M,), optional
        - "area": weight each face by its area (L² projection with lumped mass).
        - "uniform": equal weight for each incident face.
        - array: custom nonnegative weights per face.

    Returns
    -------
    v : (N,) or (N, K) float
        Per-vertex signal.

    Notes
    -----
    With weights="area", this computes v = M_lumped^{-1} B f where
    B accumulates (area/3) * f to each face's three vertices—i.e., the standard
    L² projection of a piecewise-constant field onto P1 hat functions.
    """
    V = np.asarray(V)
    F = np.asarray(F, dtype=int)
    f = np.asarray(f)

    # Ensure f has shape (M, K)
    if f.ndim == 1:
        f = f[:, None]
    elif f.shape[0] != F.shape[0] and f.shape[1] == F.shape[0]:
        # Allow (K, M) input; transpose to (M, K)
        f = f.T
    if f.shape[0] != F.shape[0]:
        raise ValueError("f must have length equal to number of faces (M).")

    M = F.shape[0]
    N = V.shape[0]
    K = f.shape[1]

    # Face weights
    if isinstance(weights, str):
        if weights == "area":
            e1 = V[F[:, 1]] - V[F[:, 0]]
            e2 = V[F[:, 2]] - V[F[:, 0]]
            A = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)  # (M,)
        elif weights == "uniform":
            A = np.ones(M, dtype=V.dtype)
        else:
            raise ValueError("weights must be 'area', 'uniform', or an array of length M.")
    else:
        A = np.asarray(weights).reshape(-1)
        if A.shape[0] != M:
            raise ValueError("custom weights must have length M.")

    # Each face contributes weight/3 to each of its three vertices
    w = A / 3.0  # (M,)

    # Denominator: per-vertex total weight
    den = np.zeros(N, dtype=V.dtype)
    np.add.at(den, F[:, 0], w)
    np.add.at(den, F[:, 1], w)
    np.add.at(den, F[:, 2], w)

    # Numerator: accumulate weighted face values to vertices
    num = np.zeros((N, K), dtype=f.dtype)
    contrib = (w[:, None] * f)  # (M, K)
    np.add.at(num, F[:, 0], contrib)
    np.add.at(num, F[:, 1], contrib)
    np.add.at(num, F[:, 2], contrib)

    # Final per-vertex values
    with np.errstate(divide="ignore", invalid="ignore"):
        v = num / den[:, None]

    # Handle isolated vertices (den == 0): set to 0
    if np.any(den == 0):
        v[den == 0] = 0.0

    # Return 1D if input was scalar
    return v.ravel() if K == 1 else v


def face_to_vertex_signal_torch(V, F, f, weights="area"):
    """
    Project a per-face (piecewise-constant) signal onto vertices via
    an L²-style area-weighted average (lumped mass).

    Parameters
    ----------
    V : (N, d) float tensor
        Vertex positions (d = 2 or 3).
    F : (M, 3) long tensor
        Triangle indices into V.
    f : (M,) or (M, K) float tensor
        Per-face signal (scalar or K-channel).
    weights : {"area","uniform"} or (M,) float tensor
        Face weights. "area" uses triangle area; "uniform" uses 1.0.

    Returns
    -------
    v : (N,) or (N, K) float tensor
        Per-vertex signal.

    Notes
    -----
    With weights="area", this computes the standard L² projection of a
    piecewise-constant field onto P1 hat functions (lumped mass):
      accumulate (area/3)*f to each vertex of a face, then divide by the
      per-vertex accumulated (area/3).
    """
    if F.dtype != torch.long:
        raise ValueError("F must be a LongTensor of vertex indices.")
    if V.device != F.device or V.device != f.device:
        raise ValueError("V, F, and f must be on the same device.")

    # Ensure f has shape (M, K)
    if f.dim() == 1:
        f = f[:, None]
    elif f.size(0) != F.size(0) and f.size(1) == F.size(0):
        f = f.t()
    if f.size(0) != F.size(0):
        raise ValueError("f must have first dimension equal to number of faces (M).")

    M = F.size(0)
    N = V.size(0)
    K = f.size(1)

    # Build face weights
    if isinstance(weights, str):
        if weights == "area":
            if V.size(1) == 3:
                e1 = V[F[:, 1]] - V[F[:, 0]]
                e2 = V[F[:, 2]] - V[F[:, 0]]
                A = 0.5 * torch.linalg.vector_norm(torch.cross(e1, e2, dim=1), dim=1)
            elif V.size(1) == 2:
                v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
                A = 0.5 * torch.abs(
                    (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
                    - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
                )
            else:
                raise ValueError("V must have 2 or 3 columns for area weights.")
        elif weights == "uniform":
            A = torch.ones(M, device=V.device, dtype=V.dtype)
        else:
            raise ValueError("weights must be 'area', 'uniform', or a tensor of shape (M,).")
    else:
        A = torch.as_tensor(weights, device=V.device, dtype=V.dtype)
        if A.numel() != M:
            raise ValueError("Custom weights must have length M.")

    w = A / 3.0  # per-face weight contribution per vertex, shape (M,)

    # Denominator: per-vertex total weight
    den = torch.zeros(N, device=V.device, dtype=V.dtype)
    for j in range(3):
        den.index_add_(0, F[:, j], w)

    # Numerator: accumulate weighted face values to vertices
    num = torch.zeros((N, K), device=f.device, dtype=f.dtype)
    contrib = (w[:, None].to(dtype=f.dtype)) * f  # (M, K)
    for j in range(3):
        num.index_add_(0, F[:, j], contrib)

    # Safe division; isolated vertices (den==0) → 0
    v = torch.zeros_like(num)
    mask = den > 0
    v[mask] = num[mask] / den[mask].unsqueeze(-1)

    return v.squeeze(1) if K == 1 else v


if __name__ == "__main__":
    mesh = trimesh.load("./data/MANO.ply")
    V, F = mesh.vertices, mesh.faces

    L, grad, mass = compute_laplace_beltrami(V, F)
    M = pp3d.vertex_areas(V, F)

    evals_np, evecs_np = spectral_solve(L, M, k_eig=128)

    spectral_coeffs = spectral_graph(evecs_np, V, M)

    reconstructed_mesh = reconstruct_mesh_from_spectral_coeffs(evecs_np, spectral_coeffs)

    reconstructed_mesh = trimesh.Trimesh(reconstructed_mesh, mesh.faces)

    sbs_view(mesh, reconstructed_mesh)



