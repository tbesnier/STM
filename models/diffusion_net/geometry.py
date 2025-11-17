import scipy
import scipy.sparse.linalg as sla
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import os.path
from multiprocessing import Pool

import numpy as np
import scipy.spatial
import torch
import sklearn.neighbors

import robust_laplacian
import potpourri3d as pp3d

import models.diffusion_net.utils as utils
from .utils import toNP
from collections import defaultdict


def norm(x, highdim=False):
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return torch.norm(x, dim=len(x.shape) - 1)


def norm2(x, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return dot(x, x)


def normalize(x, divide_eps=1e-6, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    if (len(x.shape) == 1):
        raise ValueError("called normalize() on single vector of dim " +
                         str(x.shape) + " are you sure?")
    if (not highdim and x.shape[-1] > 4):
        raise ValueError("called normalize() with large last dimension " +
                         str(x.shape) + " are you sure?")
    return x / (norm(x, highdim=highdim) + divide_eps).unsqueeze(-1)


def face_coords(verts, faces):
    coords = verts[faces]
    return coords


def cross(vec_A, vec_B):
    return torch.cross(vec_A, vec_B, dim=-1)


def dot(vec_A, vec_B):
    return torch.sum(vec_A * vec_B, dim=-1)


# Given (..., 3) vectors and normals, projects out any components of vecs
# which lies in the direction of normals. Normals are assumed to be unit.

def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots.unsqueeze(-1)


def face_area(verts, faces):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)
    return 0.5 * norm(raw_normal)


def face_normals(verts, faces, normalized=True):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)

    if normalized:
        return normalize(raw_normal)

    return raw_normal


def neighborhood_normal(points):
    # points: (N, K, 3) array of neighborhood psoitions
    # points should be centered at origin
    # out: (N,3) array of normals
    # numpy in, numpy out
    (u, s, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:, 2, :]
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def mesh_vertex_normals(verts, faces):
    # numpy in / out
    face_n = toNP(face_normals(torch.tensor(verts), torch.tensor(faces)))  # ugly torch <---> numpy

    vertex_normals = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_n)
    l = np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    vertex_normals = vertex_normals / np.clip(l, a_min=1e-10, a_max=None)

    return vertex_normals


def vertex_normals(verts, faces, n_neighbors_cloud=30):
    verts_np = toNP(verts)

    if faces.numel() == 0:  # point cloud

        _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
        neigh_points = verts_np[neigh_inds, :]
        neigh_points = neigh_points - verts_np[:, np.newaxis, :]
        normals = neighborhood_normal(neigh_points)

    else:  # mesh

        normals = mesh_vertex_normals(verts_np, toNP(faces))

        # if any are NaN, wiggle slightly and recompute
        bad_normals_mask = np.isnan(normals).any(axis=1, keepdims=True)
        if bad_normals_mask.any():
            bbox = np.amax(verts_np, axis=0) - np.amin(verts_np, axis=0)
            scale = np.linalg.norm(bbox) * 1e-4
            wiggle = (np.random.RandomState(seed=777).rand(*verts.shape) - 0.5) * scale
            wiggle_verts = verts_np + bad_normals_mask * wiggle
            normals = mesh_vertex_normals(wiggle_verts, toNP(faces))

        # if still NaN assign random normals (probably means unreferenced verts in mesh)
        bad_normals_mask = np.isnan(normals).any(axis=1)
        if bad_normals_mask.any():
            normals[bad_normals_mask, :] = (np.random.RandomState(seed=777).rand(*verts.shape) - 0.5)[bad_normals_mask,
                                           :]
            normals = normals / np.linalg.norm(normals, axis=-1)[:, np.newaxis]

    normals = torch.from_numpy(normals).to(device=verts.device, dtype=verts.dtype)

    if torch.any(torch.isnan(normals)): raise ValueError("NaN normals :(")

    return normals


def build_tangent_frames(verts, faces, normals=None):
    V = verts.shape[0]
    dtype = verts.dtype
    device = verts.device

    if normals == None:
        vert_normals = vertex_normals(verts, faces)  # (V,3)
    else:
        vert_normals = normals

        # = find an orthogonal basis

    basis_cand1 = torch.tensor([1, 0, 0]).to(device=device, dtype=dtype).expand(V, -1)
    basis_cand2 = torch.tensor([0, 1, 0]).to(device=device, dtype=dtype).expand(V, -1)

    basisX = torch.where((torch.abs(dot(vert_normals, basis_cand1))
                          < 0.9).unsqueeze(-1), basis_cand1, basis_cand2)
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = cross(vert_normals, basisX)
    frames = torch.stack((basisX, basisY, vert_normals), dim=-2)

    if torch.any(torch.isnan(frames)):
        raise ValueError("NaN coordinate frame! Must be very degenerate")

    return frames


def build_grad_point_cloud(verts, frames, n_neighbors_cloud=30):
    verts_np = toNP(verts)
    frames_np = toNP(frames)

    _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
    neigh_points = verts_np[neigh_inds, :]
    neigh_vecs = neigh_points - verts_np[:, np.newaxis, :]

    # TODO this could easily be way faster. For instance we could avoid the weird edges format and the corresponding pure-python loop via some numpy broadcasting of the same logic. The way it works right now is just to share code with the mesh version. But its low priority since its preprocessing code.

    edge_inds_from = np.repeat(np.arange(verts.shape[0]), n_neighbors_cloud)
    edges = np.stack((edge_inds_from, neigh_inds.flatten()))
    edge_tangent_vecs = edge_tangent_vectors(verts, frames, edges)

    return build_grad(verts_np, torch.tensor(edges), edge_tangent_vecs)


def edge_tangent_vectors(verts, frames, edges):
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]

    compX = dot(edge_vecs, basisX)
    compY = dot(edge_vecs, basisY)
    edge_tangent = torch.stack((compX, compY), dim=-1)

    return edge_tangent


def build_grad(verts, edges, edge_tangent_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator. Given real inputs at vertices, produces a complex (vector value) at vertices giving the gradient. All values pointwise.
    - edges: (2, E)
    """

    edges_np = toNP(edges)
    edge_tangent_vectors_np = toNP(edge_tangent_vectors)

    # TODO find a way to do this in pure numpy?

    # Build outgoing neighbor lists
    N = verts.shape[0]
    vert_edge_outgoing = [[] for i in range(N)]
    for iE in range(edges_np.shape[1]):
        tail_ind = edges_np[0, iE]
        tip_ind = edges_np[1, iE]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(iE)

    # Build local inversion matrix for each vertex
    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5
    for iV in range(N):
        n_neigh = len(vert_edge_outgoing[iV])

        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [iV]
        for i_neigh in range(n_neigh):
            iE = vert_edge_outgoing[iV][i_neigh]
            jV = edges_np[1, iE]
            ind_lookup.append(jV)

            edge_vec = edge_tangent_vectors[iE][:]
            w_e = 1.

            lhs_mat[i_neigh][:] = w_e * edge_vec
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

        lhs_T = lhs_mat.T
        lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(2)) @ lhs_T

        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(iV)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)), shape=(
            N, N)).tocsc()

    return mat


def compute_operators(verts, faces, k_eig, normals=None):
    """
    Builds spectral operators for a mesh/point cloud. Constructs mass matrix, eigenvalues/vectors for Laplacian, and gradient matrix.

    See get_operators() for a similar routine that wraps this one with a layer of caching.

    Torch in / torch out.

    Arguments:
      - vertices: (V,3) vertex positions
      - faces: (F,3) list of triangular faces. If empty, assumed to be a point cloud.
      - k_eig: number of eigenvectors to use

    Returns:
      - frames: (V,3,3) X/Y/Z coordinate frame at each vertex. Z coordinate is normal (e.g. [:,2,:] for normals)
      - massvec: (V) real diagonal of lumped mass matrix
      - L: (VxV) real sparse matrix of (weak) Laplacian
      - evals: (k) list of eigenvalues of the Laplacian
      - evecs: (V,k) list of eigenvectors of the Laplacian 
      - gradX: (VxV) sparse matrix which gives X-component of gradient in the local basis at the vertex
      - gradY: same as gradX but for Y-component of gradient

    PyTorch doesn't seem to like complex sparse matrices, so we store the "real" and "imaginary" (aka X and Y) gradient matrices separately, rather than as one complex sparse matrix.

    Note: for a generalized eigenvalue problem, the mass matrix matters! The eigenvectors are only othrthonormal with respect to the mass matrix, like v^H M v, so the mass (given as the diagonal vector massvec) needs to be used in projections, etc.
    """

    device = verts.device
    dtype = verts.dtype
    V = verts.shape[0]
    is_cloud = True  #faces.numel() == 0

    eps = 1e-12

    verts_np = toNP(verts).astype(np.float64)
    faces_np = toNP(faces)
    frames = build_tangent_frames(verts, faces, normals=normals)
    frames_np = toNP(frames)

    # Build the scalar Laplacian
    if is_cloud:
        L, M = robust_laplacian.point_cloud_laplacian(verts_np)
        massvec_np = M.diagonal()
    else:
        #L, M = robust_laplacian.mesh_laplacian(verts_np, faces_np)
        #massvec_np = M.diagonal()
        L = pp3d.cotan_laplacian(verts_np, faces_np, denom_eps=eps)
    massvec_np = pp3d.vertex_areas(verts_np, faces_np)
    massvec_np += eps * np.mean(massvec_np)

    if (np.isnan(L.data).any()):
        raise RuntimeError("NaN Laplace matrix")
    if (np.isnan(massvec_np).any()):
        raise RuntimeError("NaN mass matrix")

    # Read off neighbors & rotations from the Laplacian
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col

    # === Compute the eigenbasis
    if k_eig > 0:

        # Prepare matrices
        L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * eps).tocsc()
        massvec_eigsh = massvec_np
        Mmat = scipy.sparse.diags(massvec_eigsh)
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
                L_eigsh = L_eigsh + scipy.sparse.identity(L.shape[0]) * (eps * 10 ** failcount)


    else:  # k_eig == 0
        evals_np = np.zeros((0))
        evecs_np = np.zeros((verts.shape[0], 0))

    # == Build gradient matrices

    # For meshes, we use the same edges as were used to build the Laplacian. For point clouds, use a whole local neighborhood
    if is_cloud:
        grad_mat_np = build_grad_point_cloud(verts, frames)
    else:
        edges = torch.tensor(np.stack((inds_row, inds_col), axis=0), device=device, dtype=faces.dtype)
        edge_vecs = edge_tangent_vectors(verts, frames, edges)
        grad_mat_np = build_grad(verts, edges, edge_vecs)

    # Split complex gradient in to two real sparse mats (torch doesn't like complex sparse matrices)
    gradX_np = np.real(grad_mat_np)
    gradY_np = np.imag(grad_mat_np)

    # === Convert back to torch
    massvec = torch.from_numpy(massvec_np).to(device=device, dtype=dtype)
    L = utils.sparse_np_to_torch(L).to(device=device, dtype=dtype)
    evals = torch.from_numpy(evals_np).to(device=device, dtype=dtype)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=dtype)
    gradX = utils.sparse_np_to_torch(gradX_np).to(device=device, dtype=dtype)
    gradY = utils.sparse_np_to_torch(gradY_np).to(device=device, dtype=dtype)

    return frames, massvec, L, evals, evecs, gradX, gradY


def get_all_operators(verts_list, faces_list, k_eig, op_cache_dir=None, normals=None):
    N = len(verts_list)

    frames = [None] * N
    massvec = [None] * N
    L = [None] * N
    evals = [None] * N
    evecs = [None] * N
    gradX = [None] * N
    gradY = [None] * N

    inds = [i for i in range(N)]
    # process in random order
    # random.shuffle(inds)

    for num, i in enumerate(inds):
        print("get_all_operators() processing {} / {} {:.3f}%".format(num, N, num / N * 100))
        if normals is None:
            outputs = get_operators(verts_list[i], faces_list[i], k_eig, op_cache_dir)
        else:
            outputs = get_operators(verts_list[i], faces_list[i], k_eig, op_cache_dir, normals=normals[i])
        frames[i] = outputs[0]
        massvec[i] = outputs[1]
        L[i] = outputs[2]
        evals[i] = outputs[3]
        evecs[i] = outputs[4]
        gradX[i] = outputs[5]
        gradY[i] = outputs[6]

    return frames, massvec, L, evals, evecs, gradX, gradY


def get_operators(verts, faces, k_eig=128, op_cache_dir=None, normals=None, overwrite_cache=False):
    """
    See documentation for compute_operators(). This essentailly just wraps a call to compute_operators, using a cache if possible.
    All arrays are always computed using double precision for stability, then truncated to single precision floats to store on disk, and finally returned as a tensor with dtype/device matching the `verts` input.
    """

    device = verts.device
    dtype = verts.dtype
    verts_np = toNP(verts)
    faces_np = toNP(faces)
    is_cloud = faces.numel() == 0

    if (np.isnan(verts_np).any()):
        raise RuntimeError("tried to construct operators from NaN verts")

    # Check the cache directory
    # Note 1: Collisions here are exceptionally unlikely, so we could probably just use the hash...
    #         but for good measure we check values nonetheless.
    # Note 2: There is a small possibility for race conditions to lead to bucket gaps or duplicate
    #         entries in this cache. The good news is that that is totally fine, and at most slightly
    #         slows performance with rare extra cache misses.
    found = False
    if op_cache_dir is not None:
        utils.ensure_dir_exists(op_cache_dir)
        hash_key_str = str(utils.hash_arrays((verts_np, faces_np)))
        # print("Building operators for input with hash: " + hash_key_str)

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                op_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")

            try:
                # print('loading path: ' + str(search_path))
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]
                cache_k_eig = npzfile["k_eig"].item()

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts, cache_verts)) or (not np.array_equal(faces, cache_faces)):
                    i_cache_search += 1
                    print("hash collision! searching next.")
                    continue

                # print("  cache hit!")

                # If we're overwriting, or there aren't enough eigenvalues, just delete it; we'll create a new
                # entry below more eigenvalues
                if overwrite_cache:
                    print("  overwriting cache by request")
                    os.remove(search_path)
                    break

                if cache_k_eig < k_eig:
                    print("  overwriting cache --- not enough eigenvalues")
                    os.remove(search_path)
                    break

                if "L_data" not in npzfile:
                    print("  overwriting cache --- entries are absent")
                    os.remove(search_path)
                    break

                def read_sp_mat(prefix):
                    data = npzfile[prefix + "_data"]
                    indices = npzfile[prefix + "_indices"]
                    indptr = npzfile[prefix + "_indptr"]
                    shape = npzfile[prefix + "_shape"]
                    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
                    return mat

                # This entry matches! Return it.
                frames = npzfile["frames"]
                mass = npzfile["mass"]
                L = read_sp_mat("L")
                evals = npzfile["evals"][:k_eig]
                evecs = npzfile["evecs"][:, :k_eig]
                gradX = read_sp_mat("gradX")
                gradY = read_sp_mat("gradY")

                frames = torch.from_numpy(frames).to(device=device, dtype=dtype)
                mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
                L = utils.sparse_np_to_torch(L).to(device=device, dtype=dtype)
                evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
                evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
                gradX = utils.sparse_np_to_torch(gradX).to(device=device, dtype=dtype)
                gradY = utils.sparse_np_to_torch(gradY).to(device=device, dtype=dtype)

                found = True

                break

            except FileNotFoundError:
                print("  cache miss -- constructing operators")
                break

            except Exception as E:
                print("unexpected error loading file: " + str(E))
                print("-- constructing operators")
                break

    if not found:

        # No matching entry found; recompute.
        frames, mass, L, evals, evecs, gradX, gradY = compute_operators(verts, faces, k_eig, normals=normals)

        dtype_np = np.float32

        # Store it in the cache
        if op_cache_dir is not None:
            L_np = utils.sparse_torch_to_np(L).astype(dtype_np)
            gradX_np = utils.sparse_torch_to_np(gradX).astype(dtype_np)
            gradY_np = utils.sparse_torch_to_np(gradY).astype(dtype_np)

            np.savez(search_path,
                     verts=verts_np.astype(dtype_np),
                     frames=toNP(frames).astype(dtype_np),
                     faces=faces_np,
                     k_eig=k_eig,
                     mass=toNP(mass).astype(dtype_np),
                     L_data=L_np.data.astype(dtype_np),
                     L_indices=L_np.indices,
                     L_indptr=L_np.indptr,
                     L_shape=L_np.shape,
                     evals=toNP(evals).astype(dtype_np),
                     evecs=toNP(evecs).astype(dtype_np),
                     gradX_data=gradX_np.data.astype(dtype_np),
                     gradX_indices=gradX_np.indices,
                     gradX_indptr=gradX_np.indptr,
                     gradX_shape=gradX_np.shape,
                     gradY_data=gradY_np.data.astype(dtype_np),
                     gradY_indices=gradY_np.indices,
                     gradY_indptr=gradY_np.indptr,
                     gradY_shape=gradY_np.shape,
                     )

    return frames, mass, L, evals, evecs, gradX, gradY


def to_basis(values, basis, massvec):
    """
    Transform data in to an orthonormal basis (where orthonormal is wrt to massvec)
    Inputs:
      - values: (B,V,D)
      - basis: (B,V,K)
      - massvec: (B,V)
    Outputs:
      - (B,K,D) transformed values
    """
    basisT = basis.transpose(-2, -1)
    return torch.matmul(basisT, values * massvec.unsqueeze(-1))


def from_basis(values, basis):
    """
    Transform data out of an orthonormal basis
    Inputs:
      - values: (K,D)
      - basis: (V,K)
    Outputs:
      - (V,D) reconstructed values
    """
    if values.is_complex() or basis.is_complex():
        return utils.cmatmul(utils.ensure_complex(basis), utils.ensure_complex(values))
    else:
        return torch.matmul(basis, values)


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
        return out.squeeze(0)
    else:
        return out


def compute_hks_autoscale(evals, evecs, count):
    # these scales roughly approximate those suggested in the hks paper
    scales = torch.logspace(-2, 0., steps=count, device=evals.device, dtype=evals.dtype)
    return compute_hks(evals, evecs, scales)


def face_geometry_torch(vertices: torch.Tensor,
                        faces: torch.LongTensor):
    B, V, _ = vertices.shape
    F = faces.shape[0]
    v0 = vertices[:, faces[:, 0], :]
    v1 = vertices[:, faces[:, 1], :]
    v2 = vertices[:, faces[:, 2], :]
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    areas = 0.5 * torch.norm(cross, dim=-1)
    denom = torch.clamp(2.0 * areas, min=1e-12).unsqueeze(-1)
    normals = cross / denom
    centroids = (v0 + v1 + v2) / 3.0
    return centroids, normals, areas

def face_adjacency(faces: torch.LongTensor):
    faces_np = faces.cpu().numpy()
    edge2faces = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces_np):
        for e in ((a, b), (b, c), (c, a)):
            edge2faces[tuple(sorted(e))].append(fi)
    neigh = [[] for _ in range(len(faces_np))]
    for flist in edge2faces.values():
        if len(flist) == 2:
            f1, f2 = flist
            neigh[f1].append(f2)
            neigh[f2].append(f1)
    return [torch.LongTensor(n) for n in neigh]

def varifold_signature_local_torch(vertices: torch.Tensor,
                                   faces: torch.LongTensor,
                                   sigma: float,
                                   angular_power: int = 2):
    device, dtype = vertices.device, vertices.dtype
    centroids, normals, areas = face_geometry_torch(vertices, faces)
    neighbours = face_adjacency(faces)
    B, F, _ = centroids.shape
    signature = torch.zeros((B, F), device=device, dtype=dtype)
    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)

    for i in range(F):
        idx = torch.cat([torch.tensor([i], device=device).long(),
                         neighbours[i].to(device)], dim=0)
        diffs = centroids[:, i:i+1, :] - centroids[:, idx, :]
        spatial_k = torch.exp(-torch.sum(diffs * diffs, dim=-1) * inv_two_sigma2)

        if angular_power == 0:
            angular_k = 1.0
        else:
            cos_theta = torch.clamp(
                torch.sum(normals[:, i:i+1, :] * normals[:, idx, :], dim=-1),
                -1.0, 1.0
            )
            angular_k = cos_theta.pow(angular_power)

        area_w = areas[:, idx]
        signature[:, i] = torch.sum(spatial_k * angular_k * area_w, dim=-1)

    return signature  # (B, F)

def face_to_vertex_average(signature: torch.Tensor,
                           faces: torch.LongTensor,
                           V: int) -> torch.Tensor:
    B, F = signature.shape
    device, dtype = signature.device, signature.dtype
    faces_flat = faces.view(-1)
    sig_flat = signature.unsqueeze(-1).expand(-1, -1, 3).reshape(B, -1)
    idx = faces_flat.unsqueeze(0).expand(B, -1)

    vertex_sum = torch.zeros((B, V), device=device, dtype=dtype)
    vertex_sum = vertex_sum.scatter_add(1, idx, sig_flat)

    counts = torch.zeros((V,), device=device, dtype=dtype)
    counts = counts.scatter_add(0, faces_flat, torch.ones_like(faces_flat, dtype=dtype))
    counts = torch.clamp(counts, min=1.0)

    return vertex_sum / counts.unsqueeze(0)  # (B, V)

def varifold_signature_multiscale(vertices: torch.Tensor,
                                  faces: torch.LongTensor,
                                  sigmas: list[float],
                                  angular_power: int = 2) -> torch.Tensor:
    """
    vertices: (B, V, 3)
    faces:    (F, 3)
    sigmas:   list of S bandwidths
    → returns (B, V, S)
    """
    B, V, _ = vertices.shape
    S = len(sigmas)
    device, dtype = vertices.device, vertices.dtype

    out = torch.zeros((B, V, S), device=device, dtype=dtype)
    for s, sigma in enumerate(sigmas):
        sig_faces = varifold_signature_local_torch(vertices, faces, sigma, angular_power)  # (B, F)
        sig_verts = face_to_vertex_average(sig_faces, faces, V)                             # (B, V)
        out[:, :, s] = sig_verts

    return out

def normalize_positions(pos, faces=None, method='mean', scale_method='max_rad'):
    # center and unit-scale positions

    if method == 'mean':
        # center using the average point position
        pos = (pos - torch.mean(pos, dim=-2, keepdim=True))
    elif method == 'bbox':
        # center via the middle of the axis-aligned bounding box
        bbox_min = torch.min(pos, dim=-2).values
        bbox_max = torch.max(pos, dim=-2).values
        center = (bbox_max + bbox_min) / 2.
        pos -= center.unsqueeze(-2)
    else:
        raise ValueError("unrecognized method")

    if scale_method == 'max_rad':
        scale = torch.max(norm(pos), dim=-1, keepdim=True).values.unsqueeze(-1)
        pos = pos / scale
    elif scale_method == 'area':
        if faces is None:
            raise ValueError("must pass faces for area normalization")
        coords = pos[faces]
        vec_A = coords[:, 1, :] - coords[:, 0, :]
        vec_B = coords[:, 2, :] - coords[:, 0, :]
        face_areas = torch.norm(torch.cross(vec_A, vec_B, dim=-1), dim=1) * 0.5
        total_area = torch.sum(face_areas)
        scale = (1. / torch.sqrt(total_area))
        pos = pos * scale
    else:
        raise ValueError("unrecognized scale method")
    return pos


# Finds the k nearest neighbors of source on target.
# Return is two tensors (distances, indices). Returned points will be sorted in increasing order of distance.
def find_knn(points_source, points_target, k, largest=False, omit_diagonal=False, method='brute'):
    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError("omit_diagonal can only be used when source and target are same shape")

    if method != 'cpu_kd' and points_source.shape[0] * points_target.shape[0] > 1e8:
        method = 'cpu_kd'
        print("switching to cpu_kd knn")

    if method == 'brute':

        # Expand so both are NxMx3 tensor
        points_source_expand = points_source.unsqueeze(1)
        points_source_expand = points_source_expand.expand(-1, points_target.shape[0], -1)
        points_target_expand = points_target.unsqueeze(0)
        points_target_expand = points_target_expand.expand(points_source.shape[0], -1, -1)

        diff_mat = points_source_expand - points_target_expand
        dist_mat = norm(diff_mat)

        if omit_diagonal:
            torch.diagonal(dist_mat)[:] = float('inf')

        result = torch.topk(dist_mat, k=k, largest=largest, sorted=True)
        return result

    elif method == 'cpu_kd':

        if largest:
            raise ValueError("can't do largest with cpu_kd")

        points_source_np = toNP(points_source)
        points_target_np = toNP(points_target)

        # Build the tree
        kd_tree = sklearn.neighbors.KDTree(points_target_np)

        k_search = k + 1 if omit_diagonal else k
        _, neighbors = kd_tree.query(points_source_np, k=k_search)

        if omit_diagonal:
            # Mask out self element
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

            # make sure we mask out exactly one element in each row, in rare case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            neighbors = neighbors[mask].reshape((neighbors.shape[0], neighbors.shape[1] - 1))

        inds = torch.tensor(neighbors, device=points_source.device, dtype=torch.int64)
        dists = norm(points_source.unsqueeze(1).expand(-1, k, -1) - points_target[inds])

        return dists, inds

    else:
        raise ValueError("unrecognized method")


def farthest_point_sampling(points, n_sample):
    # Torch in, torch out. Returns a |V| mask with n_sample elements set to true.

    N = points.shape[0]
    if (n_sample > N): raise ValueError("not enough points to sample")

    chosen_mask = torch.zeros(N, dtype=torch.bool, device=points.device)
    min_dists = torch.ones(N, dtype=points.dtype, device=points.device) * float('inf')

    # pick the centermost first point
    points = normalize_positions(points)
    i = torch.min(norm2(points), dim=0).indices
    chosen_mask[i] = True

    for _ in range(n_sample - 1):
        # update distance
        dists = norm2(points[i, :].unsqueeze(0) - points)
        min_dists = torch.minimum(dists, min_dists)

        # take the farthest
        i = torch.max(min_dists, dim=0).indices.item()
        chosen_mask[i] = True

    return chosen_mask


def geodesic_label_errors(target_verts, target_faces, pred_labels, gt_labels, normalization='diameter',
                          geodesic_cache_dir=None):
    """
    Return a vector of distances between predicted and ground-truth lables (normalized by geodesic diameter or area)

    This method is SLOW when it needs to recompute geodesic distances.
    """

    # move all to numpy cpu
    target_verts = toNP(target_verts)
    target_faces = toNP(target_faces)

    pred_labels = toNP(pred_labels)
    gt_labels = toNP(gt_labels)

    dists = get_all_pairs_geodesic_distance(target_verts, target_faces, geodesic_cache_dir)

    result_dists = dists[pred_labels, gt_labels]

    if normalization == 'diameter':
        geodesic_diameter = np.max(dists)
        normalized_result_dists = result_dists / geodesic_diameter
    elif normalization == 'area':
        total_area = torch.sum(face_area(torch.tensor(target_verts), torch.tensor(target_faces)))
        normalized_result_dists = result_dists / torch.sqrt(total_area)
    else:
        raise ValueError('unrecognized normalization')

    return normalized_result_dists


# This function and the helper class below are to support parallel computation of all-pairs geodesic distance
def all_pairs_geodesic_worker(verts, faces, i):
    import igl

    N = verts.shape[0]

    # TODO: this re-does a ton of work, since it is called independently each time. Some custom C++ code could surely make it faster.
    sources = np.array([i])[:, np.newaxis]
    targets = np.arange(N)[:, np.newaxis]
    dist_vec = igl.exact_geodesic(verts, faces, sources, targets)

    return dist_vec


class AllPairsGeodesicEngine(object):
    def __init__(self, verts, faces):
        self.verts = verts
        self.faces = faces

    def __call__(self, i):
        return all_pairs_geodesic_worker(self.verts, self.faces, i)


def get_all_pairs_geodesic_distance(verts_np, faces_np, geodesic_cache_dir=None):
    """
    Return a gigantic VxV dense matrix containing the all-pairs geodesic distance matrix. Internally caches, recomputing only if necessary.

    (numpy in, numpy out)
    """

    # need libigl for geodesic call
    try:
        import igl
    except ImportError as e:
        raise ImportError(
            "Must have python libigl installed for all-pairs geodesics. `conda install -c conda-forge igl`")

    # Check the cache
    found = False
    if geodesic_cache_dir is not None:
        utils.ensure_dir_exists(geodesic_cache_dir)
        hash_key_str = str(utils.hash_arrays((verts_np, faces_np)))
        # print("Building operators for input with hash: " + hash_key_str)

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                geodesic_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")

            try:
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts_np, cache_verts)) or (not np.array_equal(faces_np, cache_faces)):
                    i_cache_search += 1
                    continue

                # This entry matches! Return it.
                found = True
                result_dists = npzfile["dist"]
                break

            except FileNotFoundError:
                break

    if not found:

        print("Computing all-pairs geodesic distance (warning: SLOW!)")

        # Not found, compute from scratch
        # warning: slowwwwwww

        N = verts_np.shape[0]

        try:
            pool = Pool(None)  # on 8 processors
            engine = AllPairsGeodesicEngine(verts_np, faces_np)
            outputs = pool.map(engine, range(N))
        finally:  # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()

        result_dists = np.array(outputs)

        # replace any failed values with nan
        result_dists = np.nan_to_num(result_dists, nan=np.nan, posinf=np.nan, neginf=np.nan)

        # we expect that this should be a symmetric matrix, but it might not be. Take the min of the symmetric values to make it symmetric
        result_dists = np.fmin(result_dists, np.transpose(result_dists))

        # on rare occaisions MMP fails, yielding nan/inf; set it to the largest non-failed value if so
        max_dist = np.nanmax(result_dists)
        result_dists = np.nan_to_num(result_dists, nan=max_dist, posinf=max_dist, neginf=max_dist)

        print("...finished computing all-pairs geodesic distance")

        # put it in the cache if possible
        if geodesic_cache_dir is not None:
            print("saving geodesic distances to cache: " + str(geodesic_cache_dir))

            # TODO we're potentially saving a double precision but only using a single
            # precision here; could save storage by always saving as floats
            np.savez(search_path,
                     verts=verts_np,
                     faces=faces_np,
                     dist=result_dists
                     )

    return result_dists


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
        return wks.squeeze(0)
    else:
        return wks


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


from scipy.spatial import cKDTree
from typing import Tuple, Optional


def compute_shot_descriptors_open3d(vertices: np.ndarray,
                                    faces: np.ndarray,
                                    radius: float = 0.1,
                                    keypoint_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute FPFH descriptors using Open3D (similar to SHOT, easier to install).
    Open3D doesn't have SHOT, but FPFH is a similar local descriptor.
    This version uses all mesh vertices instead of sampling.
    """
    import open3d as o3d

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # Create point cloud from mesh vertices (no sampling)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices  # Use all vertices
    pcd.normals = mesh.vertex_normals  # Use computed vertex normals

    # Build KDTree for the full point cloud
    #pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    if keypoint_indices is not None:
        # Compute FPFH only for specified keypoints
        keypoint_pcd = o3d.geometry.PointCloud()
        keypoint_pcd.points = o3d.utility.Vector3dVector(vertices[keypoint_indices])
        keypoint_pcd.normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals)[keypoint_indices])

        # Compute FPFH for keypoints using full point cloud as search surface
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            keypoint_pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
        )
        descriptors = np.asarray(fpfh.data).T
    else:
        # Compute FPFH for all vertices
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
        )
        descriptors = np.asarray(fpfh.data).T

    return descriptors

from scipy.spatial import cKDTree

# ------------------------------
# Utility: vertex normals (area-weighted)
# ------------------------------
def compute_vertex_normals(V, F):
    """
    V: (n,3) float32/64
    F: (m,3) int
    Returns: (n,3) unit vertex normals
    """
    fn = np.cross(V[F[:,1]] - V[F[:,0]], V[F[:,2]] - V[F[:,0]])  # (m,3), area-weighted face normals
    vn = np.zeros_like(V, dtype=np.float64)
    for i in range(3):
        np.add.at(vn, F[:, i], fn)
    # normalize
    norms = np.linalg.norm(vn, axis=1)
    norms[norms == 0.0] = 1.0
    vn /= norms[:, None]
    return vn

# ------------------------------
# Local Reference Frame (LRF)
# ------------------------------
def compute_lrf(p, nbrs, radius):
    """
    p: (3,) vertex position
    nbrs: (k,3) neighbor positions (within 'radius' of p), INCLUDING or EXCLUDING p (either is fine)
    radius: support radius
    Returns: (3,3) rotation matrix whose columns are x,y,z unit axes of the LRF.
             z-axis ~ smallest variation (approx normal); sign disambiguation applied.
             If degenerate, returns identity.
    """
    if nbrs.shape[0] < 5:
        return np.eye(3)

    # vectors from p
    d = nbrs - p[None, :]
    r = np.linalg.norm(d, axis=1)
    # weights: linearly decrease to boundary (SHOT uses distance weighting)
    w = np.clip(1.0 - r / max(radius, 1e-12), 0.0, 1.0)
    if np.all(w == 0):
        return np.eye(3)

    # weighted scatter (centered at p)
    # S = sum_i w_i * d_i * d_i^T
    S = (d * w[:, None]).T @ d

    # eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(S)  # returns ascending eigenvalues
    # Smallest eigenvalue ~ surface normal direction (z)
    z = eigvecs[:, 0]
    x = eigvecs[:, 2]  # largest variance
    y = np.cross(z, x)
    # Re-orthonormalize in case of numerical issues
    x /= np.linalg.norm(x) + 1e-12
    z /= np.linalg.norm(z) + 1e-12
    y = np.cross(z, x); y /= np.linalg.norm(y) + 1e-12
    x = np.cross(y, z); x /= np.linalg.norm(x) + 1e-12

    # Sign disambiguation: align each axis with the "majority" of neighbor offsets
    def fix_sign(axis):
        proj = d @ axis
        score = np.sum(np.sign(proj) * (proj ** 2) * w)  # emphasize larger projections; weight by distance weights
        return axis if score >= 0 else -axis

    x = fix_sign(x)
    y = fix_sign(y)
    z = fix_sign(z)

    R = np.stack([x, y, z], axis=1)
    return R

# ------------------------------
# Soft binning (linear) helper
# ------------------------------
def linear_bin_and_weight(x, nbins, wrap=False):
    """
    x in [0, 1) (or any real if wrap=True; we wrap then clamp)
    Returns: (i0, i1, w0, w1) for linear interpolation between adjacent bins.
    """
    if wrap:
        x = x % 1.0
    else:
        x = np.clip(x, 0.0, np.nextafter(1.0, -1.0))  # keep below 1 so ceil-1 valid
    fx = x * nbins
    i0 = int(np.floor(fx))
    t = fx - i0
    i1 = (i0 + 1) % nbins if wrap else min(i0 + 1, nbins - 1)
    return i0, i1, (1.0 - t), t

# ------------------------------
# SHOT for a single vertex
# ------------------------------
def shot_at_vertex(i, V, normals, tree, radius,
                   spatial_bins=(8, 2, 2), angle_bins=11):
    """
    Compute SHOT at vertex i.
    Returns: (S,) where S = 8*2*2*angle_bins (default 352)
    """
    center = V[i]
    key_n = normals[i]
    idx = tree.query_ball_point(center, radius)
    if len(idx) <= 1:
        return np.zeros(spatial_bins[0] * spatial_bins[1] * spatial_bins[2] * angle_bins, dtype=np.float64)

    nbr_pos = V[idx]
    nbr_norm = normals[idx]

    # LRF
    R = compute_lrf(center, nbr_pos, radius)
    # transform neighbor offsets into LRF
    d = (nbr_pos - center[None, :]) @ R   # (k,3) columns are x,y,z in LRF
    rho = np.linalg.norm(d, axis=1)
    valid = (rho > 1e-12) & (rho <= radius)
    if not np.any(valid):
        return np.zeros(spatial_bins[0] * spatial_bins[1] * spatial_bins[2] * angle_bins, dtype=np.float64)

    d = d[valid]; rho = rho[valid]
    nn = nbr_norm[valid]

    # spherical coords in LRF:
    # azimuth ∈ [0, 2π), elevation ∈ [0, π] (polar angle from +z)
    x, y, z = d[:,0], d[:,1], d[:,2]
    az = np.arctan2(y, x)  # [-pi, pi]
    az = np.where(az < 0, az + 2*np.pi, az)  # [0, 2pi)
    el = np.arccos(np.clip(z / rho, -1.0, 1.0))  # [0, pi]

    # spatial normalized coordinates in [0,1)
    az_u = az / (2*np.pi)                     # 8 bins, wrap-around
    el_u = el / np.pi                         # 2 bins
    r_u  = rho / radius                       # 2 bins

    # deviation angle between keypoint normal and neighbor normal
    cosang = np.einsum('ij,j->i', nn, key_n)  # dot with key_n
    cosang = np.clip(cosang, -1.0, 1.0)
    theta = np.arccos(cosang)                 # [0, pi]
    ang_u = theta / np.pi                     # map to [0,1] for binning over angle_bins

    A, E, Rr = spatial_bins
    S = A * E * Rr * angle_bins
    desc = np.zeros(S, dtype=np.float64)

    # weights: neighbors closer to center contribute slightly more
    base_w = (1.0 - r_u)  # simple linear; >=0

    for j in range(d.shape[0]):
        i_az0, i_az1, w_az0, w_az1 = linear_bin_and_weight(az_u[j], A, wrap=True)
        i_el0, i_el1, w_el0, w_el1 = linear_bin_and_weight(el_u[j], E, wrap=False)
        i_r0,  i_r1,  w_r0,  w_r1  = linear_bin_and_weight(r_u[j],  Rr, wrap=False)
        i_a0,  i_a1,  w_a0,  w_a1  = linear_bin_and_weight(ang_u[j], angle_bins, wrap=False)

        # quadrilinear interpolation across (az, el, r, angle)
        # 2×2×2×2 = 16 contributions
        w_spatial = np.array([
            (w_az0*w_el0*w_r0),
            (w_az1*w_el0*w_r0),
            (w_az0*w_el1*w_r0),
            (w_az1*w_el1*w_r0),
            (w_az0*w_el0*w_r1),
            (w_az1*w_el0*w_r1),
            (w_az0*w_el1*w_r1),
            (w_az1*w_el1*w_r1),
        ])
        az_idx = np.array([i_az0, i_az1, i_az0, i_az1, i_az0, i_az1, i_az0, i_az1])
        el_idx = np.array([i_el0, i_el0, i_el1, i_el1, i_el0, i_el0, i_el1, i_el1])
        r_idx  = np.array([i_r0,  i_r0,  i_r0,  i_r0,  i_r1,  i_r1,  i_r1,  i_r1 ])

        for k in range(8):
            spatial_linear_index = ((r_idx[k] * E + el_idx[k]) * A + az_idx[k])  # [0, 31]
            base_idx = spatial_linear_index * angle_bins

            # split angle weight across two neighboring bins
            w0 = base_w[j] * w_spatial[k] * w_a0
            w1 = base_w[j] * w_spatial[k] * w_a1

            desc[base_idx + i_a0] += w0
            desc[base_idx + i_a1] += w1

    # L2 normalize descriptor (as in PCL)
    nrm = np.linalg.norm(desc)
    if nrm > 1e-12:
        desc /= nrm
    return desc

# ------------------------------
# Public API
# ------------------------------
def compute_shot_descriptors(V, F, radius,
                             spatial_bins=(8,2,2), angle_bins=11,
                             normals=None):
    """
    Compute SHOT descriptors for all vertices.
    V: (n,3) vertices
    F: (m,3) faces
    radius: support radius for descriptor/LRF
    spatial_bins: (azimuth=8, elevation=2, radial=2) → 32 subvolumes
    angle_bins: 11 → total dims 32*angle_bins (default 352)
    normals: optional (n,3) precomputed unit normals; if None, computed area-weighted from F
    n_jobs: placeholder (simple serial version; set >1 to parallelize yourself with joblib/multiprocessing)
    Returns:
        descriptors: (n, 32*angle_bins) float64
    """
    if normals is None:
        normals = compute_vertex_normals(V, F)

    tree = cKDTree(V)
    n = V.shape[0]
    S = spatial_bins[0]*spatial_bins[1]*spatial_bins[2]*angle_bins
    out = np.zeros((n, S), dtype=np.float64)

    for i in range(n):
        out[i] = shot_at_vertex(i, V, normals, tree, radius, spatial_bins, angle_bins)
    return out


