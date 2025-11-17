import numpy
import igl
import numpy as np
import torch
import time

from scipy.sparse import diags, coo_matrix, identity
from scipy.sparse import csc_matrix as sp_csc

USE_TORCH_SPARSE = True  ## This uses TORCH_SPARSE instead of TORCH.SPARSE

# This four are mutually exclusive
USE_CUPY = False  ## This uses CUPY LU decomposition on GPU
USE_CHOLESPY_GPU = True  ## This uses cholesky decomposition on GPU
USE_CHOLESPY_CPU = False  ## This uses cholesky decomposition on CPU
USE_SCIPY = False  ## This uses CUPY LU decomposition on CPU

# If USE_SCIPY = True, wether or not to use enhanced backend
USE_SCIKITS_UMFPACK = False  ## This uses UMFPACK backend for scipy instead of naive scipy.

if USE_CHOLESPY_GPU or USE_CHOLESPY_CPU:
    from cholespy import CholeskySolverD, MatrixType

if USE_CUPY and torch.cuda.is_available():
    from cupyx.scipy.sparse.linalg import spsolve_triangular
    from cupyx.scipy.sparse import csr_matrix
    import cupy
    from torch.utils.dlpack import to_dlpack, from_dlpack

if USE_SCIPY:
    if USE_SCIKITS_UMFPACK:
        from scikits.umfpack import splu as scipy_splu
    else:
        import scipy.sparse.linalg as lg

        lg.use_solver(useUmfpack=False)

        from scipy.sparse.linalg import splu as scipy_splu
        from scipy.sparse.linalg import spsolve_triangular, spsolve

if USE_TORCH_SPARSE:
    import torch_sparse

USE_UGLY_PATCH_FOR_CUPY_ERROR = False


class SparseMat:
    '''
    Sparse matrix object represented in the COO format
    Refacto : consider killing this object, byproduct of torch_sparse instead of torch.sparse (new feature)
    '''

    @staticmethod
    def from_M(M, ttype):
        return SparseMat(M[0], M[1], M[2], M[3], ttype)

    @staticmethod
    def from_coo(coo, ttype):
        inds = numpy.vstack((coo.row, coo.col))
        return SparseMat(inds, coo.data, coo.shape[0], coo.shape[1], ttype)

    def __init__(self, inds, vals, n, m, ttype):
        self.n = n
        self.m = m
        self.vals = vals
        self.inds = inds
        assert (inds.shape[0] == 2)
        assert (inds.shape[1] == vals.shape[0])
        assert (np.max(inds[0, :]) <= n)
        assert (np.max(inds[1, :] <= m))
        self.vals = torch.from_numpy(self.vals).type(ttype).contiguous()
        self.inds = torch.from_numpy(self.inds).type(torch.int64).contiguous()

    def to_coo(self):
        return coo_matrix((self.vals, (self.inds[0, :], self.inds[1, :])), shape=(self.n, self.m))

    def to_csc(self):
        return sp_csc((self.vals, (self.inds[0, :], self.inds[1, :])), shape=(self.n, self.m))

    def to_cholesky(self):
        return CholeskySolverD(self.n, self.inds[0, :], self.inds[1, :], self.vals, MatrixType.COO)

    def to(self, device):
        self.vals = self.vals.to(device)
        self.inds = self.inds.to(device)
        return self

    def multiply_with_dense(self, dense):
        if USE_TORCH_SPARSE:
            res = torch_sparse.spmm(self.inds, self.vals, self.n, self.m, dense)
            # 1000 for loop on the above line takes 0.13 sec. Fast but annoying to have this dependency
        else:
            # Somehow this is not implemented for now?
            # 1000 for loop on the above line takes 10 sec on the CPU. It is not implemented on gpu yet Slower but no dependency
            if self.vals.device.type == 'cpu':
                tensor_zero_hack = torch.FloatTensor(
                    [0]).double()  # This line was somehow responsible for a nasty NAN bug
            else:
                tensor_zero_hack = torch.cuda.FloatTensor([0]).to(dense.get_device()).double()
            # beware with addmm, it is experimental and gave me a NaN bug!
            res = torch.sparse.addmm(tensor_zero_hack, torch.sparse_coo_tensor(self.inds.double(), self.vals.double()),
                                     (dense.double())).type_as(self.vals)
            # 1000 for loop on the above line takes 0.77 sec. Slower but no dependency
        return res.contiguous()


class PoissonSystemMatrices:
    '''
    Holds the matrices needed to perform gradient and poisson computations
    Logic : this class is supposed is supposed to hold everything needed to compute Poisson Solver
    Refacto : merge with Poisson Solver
    Only accept SparseMat representation
    '''

    def __init__(self, V, F, grad, rhs, w, ttype, is_sparse=True, lap=None, cpuonly=False):
        self.dim = 3
        self.is_sparse = is_sparse
        self.w = w
        self.rhs = rhs
        self.igl_grad = grad
        self.ttype = ttype
        self.__splu_L = None
        self.__splu_U = None
        self.__splu_perm_c = None
        self.__splu_perm_r = None
        self.lap = lap
        self.__V = V
        self.__F = F
        self.cpuonly = cpuonly
        self.cpu_splu = None

    def create_poisson_solver(self):
        return PoissonSolver(self.igl_grad, self.w, self.rhs, None, self.lap)


def _convert_sparse_igl_grad_to_our_convention(input):
    '''
    The grad operator computed from igl.grad() results in a matrix of shape (3*#tri x #verts).
    It is packed such that all the x-coordinates are placed first, followed by y and z. As shown below

    ----------           ----------
    | x1 ...             | x1 ...
    | x2 ...             | y1 ...
    | x3 ...             | z1 ...
    | .                  | .
    | .                  | .
    | y1 ...             | x2 ...
    | y2 ...      ---->  | y2 ...
    | y3 ...             | z2 ...
    | .                  | .
    | .                  | .
    | z1 ...             | x3 ...
    | z2 ...             | y3 ...
    | z3 ...             | z3 ...
    | .                  | .
    | .                  | .
    ----------           ----------

    Note that this functionality cannot be computed trivially if because igl.grad() is a sparse tensor and as such
    slicing is not well defined for sparse matrices. the following code performs the above conversion and returns a
    torch.sparse tensor.
    Set check to True to verify the results by converting the matrices to dense and comparing it.
    '''
    assert type(input) == sp_csc, 'Input should be a scipy csc sparse matrix'
    T = input.tocoo()

    r_c_data = np.hstack((T.row[..., np.newaxis], T.col[..., np.newaxis],
                          T.data[..., np.newaxis]))  # horizontally stack row, col and data arrays
    r_c_data = r_c_data[r_c_data[:, 0].argsort()]  # sort along the row column

    # Separate out x, y and z blocks
    '''
    Note that for the grad operator there are exactly 3 non zero elements in a row
    '''
    L = T.shape[0]
    Tx = r_c_data[:L, :]
    Ty = r_c_data[L:2 * L, :]
    Tz = r_c_data[2 * L:3 * L, :]

    # align the y,z rows with x so that they too start from 0
    Ty[:, 0] -= Ty[0, 0]
    Tz[:, 0] -= Tz[0, 0]

    # 'strech' the x,y,z rows so that they can be interleaved.
    Tx[:, 0] *= 3
    Ty[:, 0] *= 3
    Tz[:, 0] *= 3

    # interleave the y,z into x
    Ty[:, 0] += 1
    Tz[:, 0] += 2

    Tc = np.zeros((input.shape[0] * 3, 3))
    Tc[::3] = Tx
    Tc[1::3] = Ty
    Tc[2::3] = Tz

    indices = Tc[:, :-1].astype(int)
    data = Tc[:, -1]

    return (indices.T, data, input.shape[0], input.shape[1])


class PoissonSolver:
    '''
    an object to compute gradients and solve poisson
    '''

    def __init__(self, grad, W, rhs, my_splu, lap=None):
        self.W = torch.from_numpy(W).double()
        self.grad = grad
        self.rhs = rhs
        self.my_splu = my_splu
        self.lap = lap
        self.sparse_grad = grad
        self.sparse_rhs = rhs

    def to(self, device):
        self.W = self.W.to(device)
        self.sparse_grad = self.sparse_grad.to(device)
        self.sparse_rhs = self.sparse_rhs.to(device)
        if USE_CUPY or USE_CHOLESPY_GPU:
            self.lap = self.lap.to(device)
        return self

    def jacobians_from_vertices(self, V):
        res = _multiply_sparse_2d_by_dense_3d(self.sparse_grad, V).type_as(V)
        res = res.unsqueeze(2)
        return res.view(V.shape[0], -1, 3, 3).transpose(2, 3)

    def restrict_jacobians(self, D):
        assert isinstance(D, torch.Tensor) and len(D.shape) in [3, 4]
        assert D.shape[-1] == 3 and D.shape[-2] == 3
        assert isinstance(self.W, torch.Tensor) and len(self.W.shape) == 3
        assert self.W.shape[-1] == 2 and self.W.shape[-2] == 3

        if len(D.shape) == 4:
            DW = torch.einsum("abcd,bde->abce", (D, self.W.type_as(D)))
        else:
            DW = torch.einsum("abcd,bde->abce", (D.unsqueeze(0), self.W)).squeeze(0)

        if len(DW.shape) > 4:
            DW = DW.squeeze(0)
        return DW

    def restricted_jacobians_from_vertices(self, V):
        return self.restrict_jacobians(self.jacobians_from_vertices(V))

    def solve_poisson(self, jacobians):
        assert (len(jacobians.shape) == 4)
        assert (jacobians.shape[2] == 3 and jacobians.shape[3] == 3)

        if self.my_splu is None:
            self.my_splu = self.lap.to_cholesky()

        sol = _predicted_jacobians_to_vertices_via_poisson_solve(self.my_splu, self.sparse_rhs,
                                                                 jacobians.transpose(2, 3).reshape(jacobians.shape[0],
                                                                                                   -1, 3, 1).squeeze(
                                                                     3).contiguous())
        #c = torch.mean(sol, axis=1).unsqueeze(1)  ## Beware the predicted mesh is centered here.

        return sol# - c


def compute_finsler_laplace_beltrami(vertices, faces, b1, b2, b3, alpha=0.0, tau=0.0):
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
    # First, compute the standard gradient operator
    grad = igl.grad(vertices, faces)
    # Compute double area
    d_area = igl.doublearea(vertices, faces)
    d_area = np.hstack((d_area, d_area, d_area))
    mass = sp_csc(diags(d_area))

    # Calculate face normals and principal directions
    normals = b3

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


def poisson_system_matrices_from_mesh(V, F, dim=3, ttype=torch.float64, is_sparse=True, cpuonly=False):
    '''
    compute poisson matricees for a given mesh
    :param V vertices
    :param F faces
    :param dim: for now always 3 :)
    :param ttype the type of tensor (e.g., float,double)
    :param is_sparse: for now always true
    :return: a PoissonMatricese object holding the computed matrices
    '''

    assert type(dim) == int and dim in [2, 3], f'Only two and three dimensional meshes are supported'
    assert type(is_sparse) == bool
    vertices = V
    faces = F
    is_sparse = is_sparse

    b1, b2, b3 = igl.local_basis(V, F)

    grad = igl.grad(vertices, faces)
    d_area = igl.doublearea(vertices, faces)
    d_area = np.hstack((d_area, d_area, d_area))
    mass = sp_csc(diags(d_area))

    laplace = grad.T @ mass @ grad

    # FLBO with α=0, τ=0
    #laplace, grad, mass = compute_finsler_laplace_beltrami(vertices, faces, b1, b2, b3, alpha=0.15, tau=0.0)

    # These should be very similar (up to numerical precision)
    #difference = (laplace - finsler_lbo).todense()
    #print(f"Max difference: {np.abs(difference).max()}")

    # laplace, grad, mass, diffusivity = compute_finsler_laplace_beltrami(vertices, faces, alpha=0, tau=0.0)
    #mass = _get_mass_matrix(vertices, faces, is_sparse)

    laplace = laplace[1:, 1:]
    rhs = grad.T @ mass  # needs to add diffusivity in the Finsler setting
    w = np.stack((b1, b2), axis=-1)

    rhs = rhs[1:, :]

    if is_sparse:
        laplace = laplace.tocoo()
        rhs = rhs.tocoo()
        grad = grad.tocsc()

    grad = SparseMat.from_M(_convert_sparse_igl_grad_to_our_convention(grad), torch.float64)
    poissonbuilder = PoissonSystemMatrices(V=V, F=F, grad=grad,
                                           rhs=SparseMat.from_coo(rhs, torch.float64), w=w,
                                           ttype=ttype, is_sparse=is_sparse,
                                           lap=SparseMat.from_coo(laplace, torch.float64),
                                           cpuonly=cpuonly)
    return poissonbuilder


def _get_mass_matrix(vertices, faces, is_sparse):
    d_area = igl.doublearea(vertices, faces)
    d_area = np.hstack((d_area, d_area, d_area))
    if is_sparse:
        return sp_csc(diags(d_area))
    return diags(d_area)


class SPLUSolveLayer(torch.autograd.Function):
    '''
    Implements the SPLU solve as a differentiable layer, with a forward and backward function
    '''

    @staticmethod
    def forward(ctx, solver, b):
        '''
        override forward function
        :param ctx: context object (to keep the lu object for the backward pass)
        :param lu: splu object
        :param b: right hand side, could be a vector or matrix
        :return: the vector or matrix x which holds lu.solve(b) = x
        '''
        assert isinstance(b, torch.Tensor)
        assert b.shape[-1] >= 1 and b.shape[-1] <= 3, f'got shape {b.shape} expected last dim to be in range 1-3'
        b = b.contiguous()
        ctx.solver = solver

        vertices = SPLUSolveLayer.solve(solver, b).type_as(b)

        assert not torch.isnan(vertices).any(), "Nan in the forward pass of the POISSON SOLVE"
        return vertices

    def backward(ctx, grad_output):
        '''
        overrides backward function
        :param grad_output: the gradient to be back-propped
        :return: the outgoing gradient to be back-propped
        '''

        assert isinstance(grad_output, torch.Tensor)
        assert grad_output.shape[-1] >= 1 and grad_output.shape[
            -1] <= 3, f'got shape {grad_output.shape} expected last dim to be in range 1-3'
        # when backpropping, if a layer is linear with matrix M, x ---> Mx, then the backprop of gradient g is M^Tg
        # in our case M = A^{-1}, so the backprop is to solve x = A^-T g.
        # Because A is symmetric we simply solve A^{-1}g without transposing, but this will break if A is not symmetric.
        grad_output = grad_output.contiguous()
        grad = SPLUSolveLayer.solve(ctx.solver,
                                    grad_output)
        # At this point we perform a NAN check because the backsolve sometimes returns NaNs.
        assert not torch.isnan(grad).any(), "Nan in the backward pass of the POISSON SOLVE"

        if USE_CUPY:
            mempool = cupy.get_default_memory_pool()
            pinned_mempool = cupy.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            del ctx.lu

        return None, grad

    @staticmethod
    def solve(solver, b):
        '''
        solve the linear system defined by an SPLU object for a given right hand side. if the RHS is a matrix, solution will also be a matrix.
        :param solver: the splu object (LU decomposition) or cholesky object
        :param b: the right hand side to solve for
        :return: solution x which satisfies Ax = b where A is the poisson system lu describes
        '''

        if USE_CUPY:
            b_cupy = b #cupy.fromDlpack(to_dlpack(b))
            with cupy.cuda.Device(solver.device()):
                # this will hold the solution
                sol = cupy.ndarray(b_cupy.shape)
                for i in range(b_cupy.shape[2]):  # b may have multiple columns, solve for each one
                    b2d = b_cupy[..., i]  # cupy.expand_dims(b_cpu[...,i],2)
                    s = solver.solve(b2d.T).T
                    sol[:, :, i] = s
            # # # convert back to torch
            res = from_dlpack(sol.toDlpack())

            return res.type_as(b.type())

        elif USE_SCIPY:

            assert (b.shape[0] == 1), "Need to code parrallel implem on the first dim"
            sol = solver.solve(b[0].double().cpu().numpy())
            res = torch.from_numpy(sol).to(b.device).reshape(b.shape)
            return res.type_as(b).contiguous()

        elif USE_CHOLESPY_GPU:

            b = b.double().contiguous()
            c = b.permute(1, 2, 0).contiguous()
            c = c.view(c.shape[0], -1)
            x = torch.zeros_like(c)
            solver.solve(c, x)
            x = x.view(b.shape[1], b.shape[2], b.shape[0])
            x = x.permute(2, 0, 1).contiguous()

            return x.contiguous()

        elif USE_CHOLESPY_CPU:
            assert (b.shape[0] == 1), "Need to code parrallel implem on the first dim"
            b = b.squeeze()
            b_cpu = b.cpu()
            x = torch.zeros_like(b_cpu)
            solver.solve(b_cpu, x)
            return x.contiguous().to(b.device).unsqueeze(0)


def _predicted_jacobians_to_vertices_via_poisson_solve(Lap, rhs, jacobians):
    '''
    convert the predictions to the correct convention and feed it to the poisson solve
    '''

    def _batch_rearrange_input(input):
        assert isinstance(input, torch.Tensor) and len(input.shape) in [2, 3]
        P = torch.zeros(input.shape).type_as(input)
        if len(input.shape) == 3:
            # Batched input
            k = input.shape[1] // 3
            P[:, :k, :] = input[:, ::3]
            P[:, k:2 * k, :] = input[:, 1::3]
            P[:, 2 * k:, :] = input[:, 2::3]

        else:
            k = input.shape[0] // 3
            P[:k, :] = input[::3]
            P[k:2 * k, :] = input[1::3]
            P[2 * k:, :] = input[2::3]

        return P

    def _list_rearrange_input(input):
        assert isinstance(input, list) and all([isinstance(x, torch.Tensor) and len(x.shape) in [2, 3] for x in input])
        P = []
        for p in input:
            P.append(_batch_rearrange_input(p))
        return P

    if isinstance(jacobians, list):
        P = _list_rearrange_input(jacobians)
    else:
        P = _batch_rearrange_input(jacobians)

    assert isinstance(P, torch.Tensor) and len(P.shape) in [2, 3]
    assert len(P.shape) == 3

    P = P.double()
    input_to_solve = _multiply_sparse_2d_by_dense_3d(rhs, P)

    out = SPLUSolveLayer.apply(Lap, input_to_solve)

    out = torch.cat([torch.zeros(out.shape[0], 1, out.shape[2]).type_as(out), out], dim=1)
    out = out# - torch.mean(out, axis=1, keepdim=True) #center ??

    return out.type_as(jacobians)


def _multiply_sparse_2d_by_dense_3d(mat, B):
    ret = []
    for i in range(B.shape[0]):
        C = mat.multiply_with_dense(B[i, ...])
        ret.append(C)
    ret = torch.stack(tuple(ret))
    return ret


class MyCuSPLU:
    '''
    implmentation of SPLU on the gpu via CuPy
    '''

    def __init__(self, L, U, perm_c=None, perm_r=None):
        self.__orgL = L
        self.__orgU = U
        self.L = None
        self.U = None
        self.perm_c = perm_c
        self.perm_r = perm_r
        self.__device = None

    def to(self, device):
        self.__device = device.index
        with cupy.cuda.Device(self.__device):
            self.L = csr_matrix(self.__orgL)
            self.U = csr_matrix(self.__orgU)
        return self

    def device(self):
        return self.__device

    def solve(self, b):
        """ an attempt to use SuperLU data to efficiently solve
            Ax = Pr.T L U Pc.T x = b
             - note that L from SuperLU is in CSC format solving for c
               results in an efficiency warning
            Pr . A . Pc = L . U
            Lc = b      - forward solve for c
             c = Ux     - then back solve for x
        """

        assert self.__device is not None, "need to explicitly call to() before solving"
        if USE_UGLY_PATCH_FOR_CUPY_ERROR:
            with cupy.cuda.Device(0):
                b[:1, :1].copy()[:, :1]

        with cupy.cuda.Device(self.__device):
            b = cupy.array(b)
            if self.perm_r is not None:
                b_old = b.copy()
                b[self.perm_r] = b_old

        assert b.device.id == self.__device, "got device" + str(b.device.id) + "instead of" + str(self.__device)
        try:  # unit_diagonal is a new kw
            c = spsolve_triangular(self.L, b, lower=True, unit_diagonal=True, overwrite_b=True)
        except TypeError:
            c = spsolve_triangular(self.L, b, lower=True, overwrite_b=True)
        px = spsolve_triangular(self.U, c, lower=False, overwrite_b=True)

        if self.perm_c is None:
            return px
        px = px[self.perm_c]

        return px


class MyCuSPLU_CPU:
    '''
    implmentation of SPLU on the gpu via CuPy
    '''

    def __init__(self, L, U, perm_c=None, perm_r=None):
        self.__orgL = L
        self.__orgU = U
        self.L = L
        self.U = U
        self.perm_c = perm_c
        self.perm_r = perm_r
        self.__device = 'cpu'

    def device(self):
        return self.__device

    def solve(self, b):
        """ an attempt to use SuperLU data to efficiently solve
            Ax = Pr.T L U Pc.T x = b
             - note that L from SuperLU is in CSC format solving for c
               results in an efficiency warning
            Pr . A . Pc = L . U
            Lc = b      - forward solve for c
             c = Ux     - then back solve for x
        """

        # Could be done on GPU
        if self.perm_r is not None:
            b_old = b.copy()
            b[self.perm_r] = b_old

        st = time.time()

        try:  # unit_diagonal is a new kw
            c = spsolve(self.L, b, permc_spec="NATURAL")
        except TypeError:
            c = spsolve(self.L, b, permc_spec="NATURAL")
        px = spsolve(self.U, c, permc_spec="NATURAL")
        print(f"time for spsolve_triangular CPU: {time.time() - st}")

        if self.perm_c is None:
            return px
        px = px[self.perm_c]

        return px
