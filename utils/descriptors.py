import numpy as np
import scipy.spatial
import torch
import sklearn.neighbors
from scipy.spatial.distance import cdist
import robust_laplacian
import potpourri3d as pp3d
import open3d as o3d
from pykeops.torch import Vi, Vj
from .geometry_utils import compute_vertex_areas, compute_vertex_normals, compute_face_normals, face_to_vertex_signal_torch
import igl

def compute_pc(V, F):
    u_M, u_m, k1, k2 = igl.principal_curvature(V, F)
    pc = np.vstack([k1, k2]).swapaxes(0, 1)

    return pc

def compute_fpfh(vertices: np.ndarray,
                                    faces: np.ndarray,
                                    radius: float = 0.1) -> np.ndarray:
    """
    Compute FPFH descriptors using Open3D (similar to SHOT, easier to install).
    Open3D doesn't have SHOT, but FPFH is a similar local descriptor.
    This version uses all mesh vertices instead of sampling.
    """

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # Create point cloud from mesh vertices (no sampling)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices  # Use all vertices
    pcd.normals = mesh.vertex_normals  # Use computed vertex normals

    # Compute FPFH for all vertices
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    )
    descriptors = np.asarray(fpfh.data).T

    return descriptors

def compute_wks(evals, evecs, energies, sigma=None):
    """
    Compute the Wave Kernel Signature.

    Inputs:
      - evals: (K) eigenvalues
      - evecs: (V,K) eigenvector values
      - energies: (E) energy values
      - sigma: width of the energy window (if None, computed automatically)

    Outputs:
      - (V,E) WKS values
    """

    # expand batch if needed
    if len(evals.shape) == 1:
        expand_batch = True
        evals = evals.unsqueeze(0)
        evecs = evecs.unsqueeze(0)
        energies = energies.unsqueeze(0)
    else:
        expand_batch = False

    # Ensure positive eigenvalues for log
    evals = torch.clamp(evals, min=1e-6)
    log_evals = torch.log(evals)

    # Set sigma based on the average spacing between log eigenvalues if not provided
    if sigma is None:
        log_eval_range = torch.max(log_evals) - torch.min(log_evals)
        sigma = log_eval_range / 10.0

    # Compute Gaussian weights
    # (e - log(λ_i))^2 / (2*σ^2)
    diff = energies.unsqueeze(-1) - log_evals.unsqueeze(1)  # (B,E,K)
    gaussian = torch.exp(-(diff ** 2) / (2 * sigma ** 2))  # (B,E,K)

    # Normalize the Gaussian weights
    gaussian_sum = torch.sum(gaussian, dim=-1, keepdim=True)  # (B,E,1)
    gaussian_normalized = gaussian / gaussian_sum  # (B,E,K)

    # Compute WKS
    gaussian_normalized = gaussian_normalized.unsqueeze(1)  # (B,1,E,K)
    squared_evecs = (evecs * evecs).unsqueeze(2)  # (B,V,1,K)

    terms = gaussian_normalized * squared_evecs  # (B,V,E,K)
    wks = torch.sum(terms, dim=-1)  # (B,V,E)

    if expand_batch:
        return wks.squeeze(0).detach().cpu().numpy()
    else:
        return wks.detach().cpu().numpy()


def compute_wks_autoscale(evals, evecs, count, sigma_factor=1.0):
    """
    Compute the Wave Kernel Signature with automatically generated energy values.

    Inputs:
      - evals: (K) eigenvalues
      - evecs: (V,K) eigenvector values
      - count: number of energy values to use
      - sigma_factor: factor to multiply the default sigma

    Outputs:
      - (V,count) WKS values
    """
    # Ensure positive eigenvalues for log
    evals = torch.clamp(evals, min=1e-6)
    log_evals = torch.log(evals)

    # Create energies that span the range of log eigenvalues
    log_eval_min = torch.min(log_evals)
    log_eval_max = torch.max(log_evals)

    # Create evenly spaced energy values
    energies = torch.linspace(log_eval_min, log_eval_max, steps=count,
                              device=evals.device, dtype=evals.dtype)

    # Compute sigma based on the range of log eigenvalues
    log_eval_range = log_eval_max - log_eval_min
    sigma = sigma_factor * log_eval_range / count

    return compute_wks(evals, evecs, energies, sigma)

def compute_hks(evals, evecs, scales):
    """
    Inputs:
      - evals: (K) eigenvalues
      - evecs: (V,K) values
      - scales: (S) times
    Outputs:
      - (V,S) hks values
    """

    # expand batch
    if len(evals.shape) == 1:
        expand_batch = True
        evals = evals.unsqueeze(0)
        evecs = evecs.unsqueeze(0)
        scales = scales.unsqueeze(0)
    else:
        expand_batch = False

    # TODO could be a matmul
    power_coefs = torch.exp(-evals.unsqueeze(1) * scales.unsqueeze(-1)).unsqueeze(1)  # (B,1,S,K)
    terms = power_coefs * (evecs * evecs).unsqueeze(2)  # (B,V,S,K)

    out = torch.sum(terms, dim=-1)  # (B,V,S)

    if expand_batch:
        return out.squeeze(0).detach().cpu().numpy()
    else:
        return out.detach().cpu().numpy()


def compute_hks_autoscale(evals, evecs, count):
    # these scales roughly approximate those suggested in the hks paper
    scales = torch.logspace(-2, 0., steps=count, device=evals.device, dtype=evals.dtype)
    return compute_hks(evals, evecs, scales)


def gaussian_normal_kernel(vertices1, normals1, vertices2, normals2, sigma_pos=1.0):

    # Compute pairwise squared distances
    dist_squared = cdist(vertices1, vertices2, metric='sqeuclidean')

    # Gaussian kernel for positions
    gaussian_part = np.exp(-dist_squared / (2 * sigma_pos**2))

    dot_products = np.dot(normals1, normals2.T)
    normal_part = np.abs(dot_products) #np.exp(-(2 * (1 - np.abs(dot_products))) / (2 * 0.5**2))

    return gaussian_part * normal_part

def Kernel_varifold_unoriented(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp() * ((u * v)**2).sum()
    return (K * b).sum_reduction(axis=1)

def get_center_length_normal(F, V):
    V0, V1, V2 = (
        V.index_select(0, F[:, 0]),
        V.index_select(0, F[:, 1]),
        V.index_select(0, F[:, 2]),
    )
    centers, normals = (V0 + V1 + V2) / 3, 0.5 * torch.cross(V1 - V0, V2 - V0)
    length = (normals**2).sum(dim=1)[:, None].clamp_(min=1e-12).sqrt()
    return centers, length, normals / length


def compute_intrinsic_vs(vertices, faces, sigma_pos):
    import potpourri3d as pp3d

    solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
    signal = np.zeros(vertices.shape[0])
    for i in range(vertices.shape[0]):
        geodesic_dists = solver.compute_distance(50)
        pos_kernel = np.exp(-geodesic_dists / (2 * sigma_pos**2)).sum()
        signal = np.zeros


def anti_gaussian_normal_kernel(vertices1, normals1, vertices2, normals2, sigma_pos=1.0):

    # Compute pairwise squared distances
    dist_squared = cdist(vertices1, vertices2, metric='sqeuclidean')

    # Gaussian kernel for positions
    gaussian_part = np.exp(-dist_squared / (2 * sigma_pos**2))

    dot_products = np.dot(normals1, normals2.T)
    normal_part = 1 - np.abs(dot_products) #np.exp(-(2 * (1 - np.abs(dot_products))) / (2 * 0.5**2))

    return gaussian_part * normal_part


def compute_surface_signal(vertices, faces, sigma_pos=1.0):

    # Compute vertex normals and areas
    vertex_normals = compute_vertex_normals(vertices, faces)
    vertex_areas = compute_vertex_areas(vertices, faces)

    # Handle single or multiple scales
    sigma_values = np.atleast_1d(sigma_pos)
    n_scales = len(sigma_values)
    n_vertices = len(vertices)

    if n_scales == 1:
        # Single scale case
        kernel_matrix = gaussian_normal_kernel(
            vertices, vertex_normals,
            vertices, vertex_normals,
            sigma_values[0]
        )
        signal = np.dot(kernel_matrix, vertex_areas)
        return signal, vertex_normals, vertex_areas
    else:
        # Multi-scale case
        signals = np.zeros((n_vertices, n_scales))

        for i, sigma in enumerate(sigma_values):
            kernel_matrix = gaussian_normal_kernel(
                vertices, vertex_normals,
                vertices, vertex_normals,
                sigma
            )
            signals[:, i] = np.dot(kernel_matrix, vertex_areas)

        return signals, vertex_normals, vertex_areas

def compute_multiscale_signal(vertices, faces, sigma_range=None, n_scales=5):

    sigma_min, sigma_max = sigma_range

    # Create logarithmically spaced sigma values
    if n_scales == 1:
        sigma_values = np.array([np.sqrt(sigma_min * sigma_max)])
    else:
        log_sigma_min = np.log(sigma_min)
        log_sigma_max = np.log(sigma_max)
        sigma_values = np.exp(np.linspace(log_sigma_min, log_sigma_max, n_scales))

    signals, vertex_normals, vertex_areas = compute_surface_signal(vertices, faces, sigma_values)

    return signals

def compute_surface_signal_keops(V, F, K_list, normalized=False):
    C, L, Nn = get_center_length_normal(F, V)
    n_scales = len(K_list)
    n_vertices = len(F)

    signals = torch.zeros((n_vertices, n_scales)).to("cuda:0")

    for i, K in enumerate(K_list):
        signal = K(C, C, Nn, Nn, L)
        if normalized:
            signal = signal / (signal.max() - signal.min()) #(L * signal).sum()

        signals[:, i] = signal.squeeze(1)

    signals = face_to_vertex_signal_torch(V, F, signals)
    return signals


def compute_surface_signal2(vertices, faces, sigma_pos=1.0):

    # Compute vertex normals and areas
    vertex_normals = compute_vertex_normals(vertices, faces)
    vertex_areas = compute_vertex_areas(vertices, faces)

    # Handle single or multiple scales
    sigma_values = np.atleast_1d(sigma_pos)
    n_scales = len(sigma_values)
    n_vertices = len(vertices)

    if n_scales == 1:
        # Single scale case
        kernel_matrix = anti_gaussian_normal_kernel(
            vertices, vertex_normals,
            vertices, vertex_normals,
            sigma_values[0]
        )
        signal = np.dot(kernel_matrix, vertex_areas)
        return signal, vertex_normals, vertex_areas
    else:
        # Multi-scale case
        signals = np.zeros((n_vertices, n_scales))

        for i, sigma in enumerate(sigma_values):
            kernel_matrix = anti_gaussian_normal_kernel(
                vertices, vertex_normals,
                vertices, vertex_normals,
                sigma
            )
            signals[:, i] = np.dot(kernel_matrix, vertex_areas)

        return signals, vertex_normals, vertex_areas

def compute_multiscale_signal2(vertices, faces, sigma_range=None, n_scales=5):

    sigma_min, sigma_max = sigma_range

    # Create logarithmically spaced sigma values
    if n_scales == 1:
        sigma_values = np.array([np.sqrt(sigma_min * sigma_max)])
    else:
        log_sigma_min = np.log(sigma_min)
        log_sigma_max = np.log(sigma_max)
        sigma_values = np.exp(np.linspace(log_sigma_min, log_sigma_max, n_scales))

    signals, vertex_normals, vertex_areas = compute_surface_signal2(vertices, faces, sigma_values)

    return signals



if __name__ == "__main__":
    import trimesh
    import time
    mesh = trimesh.load("../datasets/MANO_ALIGNED/01_01r.ply")
    V, F = mesh.vertices, mesh.faces

    res = compute_intrinsic_vs(V, F, sigma_pos=0.1)