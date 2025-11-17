from multiprocessing import process
import warnings

warnings.filterwarnings("ignore")

from scipy.sparse import load_npz
from PoissonSystem import poisson_system_matrices_from_mesh, SparseMat
import os
import trimesh
from easydict import EasyDict
import numpy
import numpy as np
import scipy
import scipy.sparse
import igl
from scipy.sparse import save_npz
from time import time
import torch

NUM_SAMPLES = 1024
WKS_DIM = 100


class MeshProcessor:
    '''
    Extracts all preprocessing-related data (sample points  for pointnet; wave-kernel-signature, etc.)
    '''

    def __init__(self, vertices, faces, ttype, from_file=False,
                 cpuonly=False, load_wks_samples=False, load_wks_centroids=False,
                 compute_splu=True, load_splu=False):
        '''
        :param vertices:
        :param faces:
        :param ttype: the torch data type to use (float, half, double)
        :param source_dir: the directory to load the preprocessed data from; if given, will try to load the data before computing, if not given, always compute
        '''

        self.ttype = ttype
        self.num_samples = NUM_SAMPLES
        self.vertices = vertices[:, :, :3].squeeze().detach().cpu().numpy()
        self.faces = faces.squeeze().to(dtype=torch.int32).detach().cpu().numpy()
        self.normals = vertices[:, :, 3:].squeeze().detach().cpu().numpy() #igl.per_vertex_normals(self.vertices, self.faces)
        # self.__use_wks = use_wks
        self.samples = EasyDict()
        self.samples.xyz = None
        self.samples.normals = None
        self.samples.wks = None
        self.centroids = EasyDict()
        self.centroids.points_and_normals = None
        self.centroids.wks = None
        self.diff_ops = EasyDict()
        self.diff_ops.splu = EasyDict()
        self.diff_ops.splu.L = None
        self.diff_ops.splu.U = None
        self.diff_ops.splu.perm_c = None
        self.diff_ops.splu.perm_r = None
        self.diff_ops.frames = None
        self.diff_ops.rhs = None
        self.diff_ops.grad = None
        self.diff_ops.poisson_sys_mat = None
        self.faces_wks = None
        self.vert_wks = None
        self.diff_ops.poisson = None
        self.from_file = from_file
        self.cpuonly = cpuonly
        self.load_wks_samples = load_wks_samples
        self.load_wks_centroids = load_wks_centroids
        self.compute_splu = compute_splu
        self.load_splu = load_splu

    @staticmethod
    def meshprocessor_from_directory(source_dir, ttype, cpuonly=False, load_wks_samples=False,
                                     load_wks_centroids=False):
        try:
            vertices = np.load(os.path.join(source_dir, "vertices.npy"))
            faces = np.load(os.path.join(source_dir, "faces.npy"))
        except:
            print(os.path.join(source_dir, "vertices.npy"))
            import traceback
            traceback.print_exc()
        return MeshProcessor(vertices, faces, ttype, source_dir, cpuonly=cpuonly, load_wks_samples=load_wks_samples,
                             load_wks_centroids=load_wks_centroids, compute_splu=False)

    @staticmethod
    def meshprocessor_from_file(fname, ttype, cpuonly=False, load_wks_samples=False, load_wks_centroids=False):
        if fname[-4:] == '.obj':
            V, _, _, F, _, _ = igl.read_obj(fname)
        elif fname[-4:] == '.off':
            V, F, _ = igl.read_off(fname)
        elif fname[-4:] == '.ply':
            V, F = igl.read_triangle_mesh(fname)
        return MeshProcessor(V, F, ttype, os.path.dirname(fname), True, cpuonly=cpuonly,
                             load_wks_samples=load_wks_samples, load_wks_centroids=load_wks_centroids,
                             compute_splu=False)

    @staticmethod
    def meshprocessor_from_array(vertices, faces, ttype, cpuonly=False, load_wks_samples=False,
                                 load_wks_centroids=False):
        return MeshProcessor(vertices, faces, ttype, cpuonly=cpuonly, load_wks_samples=load_wks_samples,
                             load_wks_centroids=load_wks_centroids, compute_splu=False)

    def get_vertices(self):
        return self.vertices

    def get_faces(self):
        return self.faces

    def get_samples(self):
        if self.samples.xyz is None:
            if not self.from_file:
                try:
                    self.load_samples()
                except Exception as e:
                    self.compute_samples()
                    self.save_samples()
        return self.samples

    def load_samples(self):
        if self.samples.xyz is None:
            self.samples.xyz = np.load(os.path.join(self.source_dir, 'samples.npy'))
        if self.samples.normals is None:
            self.samples.normals = np.load(os.path.join(self.source_dir, 'samples_normals.npy'))
        if self.load_wks_samples:
            if self.samples.wks is None:
                self.samples.wks = np.load(os.path.join(self.source_dir, 'samples_wks.npy'))
            if self.centroids.wks is None:
                self.centroids.wks = np.load(os.path.join(self.source_dir, 'centroid_wks.npy'))

    def save_samples(self):
        os.makedirs(self.source_dir, exist_ok=True)
        np.save(os.path.join(self.source_dir, 'samples.npy'), self.samples.xyz)
        np.save(os.path.join(self.source_dir, 'samples_normals.npy'), self.samples.normals)
        if self.load_wks_samples:
            np.save(os.path.join(self.source_dir, 'samples_wks.npy'), self.samples.wks)
            np.save(os.path.join(self.source_dir, 'centroid_wks.npy'), self.centroids.wks)

    def compute_samples(self):
        if self.load_wks_centroids or self.load_wks_centroids:
            self.computeWKS()
        pt_samples, normals_samples, wks_samples, bary = self.sample_points(self.num_samples)
        self.samples.xyz = pt_samples
        self.samples.normals = normals_samples
        self.samples.wks = wks_samples
        self.centroids.wks = self.faces_wks

    def get_centroids(self):
        if self.centroids.points_and_normals is None:
            if not self.from_file:
                self.compute_centroids()
        return self.centroids

    def compute_centroids(self):
        m = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
        self.centroids.points_and_normals = np.hstack((np.mean(m.triangles, axis=1), m.face_normals))

    def compute_centroids_batch(self):

        # Extract the vertices for each face
        v0 = self.vertices[np.arange(self.vertices.shape[0])[:, None], self.faces[..., 0]]
        v1 = self.vertices[np.arange(self.vertices.shape[0])[:, None], self.faces[..., 1]]
        v2 = self.vertices[np.arange(self.vertices.shape[0])[:, None], self.faces[..., 2]]

        # Compute the normals of each triangle
        normals = np.cross(v1 - v0, v2 - v0)

        # Normalize the normals
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        return (v0 + v1 + v2) / 3.0, normals

    def get_differential_operators(self):
        if self.diff_ops.grad is None:
            if not self.from_file:
                self.compute_differential_operators()
        if self.load_splu:
            self.get_poisson_system()
        return self.diff_ops

    def load_poisson_system(self):
        try:
            self.diff_ops.splu.L = load_npz(os.path.join(self.source_dir, 'lap_L.npz'))
            self.diff_ops.splu.U = load_npz(os.path.join(self.source_dir, 'lap_U.npz'))
            self.diff_ops.splu.perm_c = np.load(os.path.join(self.source_dir, 'lap_perm_c.npy'))
            self.diff_ops.splu.perm_r = np.load(os.path.join(self.source_dir, 'lap_perm_r.npy'))
        except:
            print(f"FAILED load poisson on: {os.path.join(self.source_dir)}")
            raise Exception("FAILED load poisson on: {os.path.join(self.source_dir)}")

    def load_differential_operators(self):
        self.diff_ops.rhs = SparseMat.from_coo(load_npz(os.path.join(self.source_dir, 'new_rhs.npz')),
                                               ttype=torch.float64)
        self.diff_ops.grad = SparseMat.from_coo(load_npz(os.path.join(self.source_dir, 'new_grad.npz')),
                                                ttype=torch.float64)
        self.diff_ops.frames = np.load(os.path.join(self.source_dir, 'w.npy'))
        self.diff_ops.laplacian = SparseMat.from_coo(load_npz(os.path.join(self.source_dir, 'laplacian.npz')),
                                                     ttype=torch.float64)

    def save_differential_operators(self):
        save_npz(os.path.join(self.source_dir, 'new_rhs.npz'), self.diff_ops.rhs.to_coo())
        save_npz(os.path.join(self.source_dir, 'new_grad.npz'), self.diff_ops.grad.to_coo())
        np.save(os.path.join(self.source_dir, 'w.npy'), self.diff_ops.frames)
        save_npz(os.path.join(self.source_dir, 'laplacian.npz'), self.diff_ops.laplacian.to_coo())

    def compute_differential_operators(self):
        '''
        process the given mesh
        '''
        poisson_sys_mat = poisson_system_matrices_from_mesh(V=self.vertices, F=self.faces, cpuonly=self.cpuonly)
        self.diff_ops.grad = poisson_sys_mat.igl_grad
        self.diff_ops.rhs = poisson_sys_mat.rhs
        self.diff_ops.laplacian = poisson_sys_mat.lap
        self.diff_ops.frames = poisson_sys_mat.w
        self.diff_ops.poisson_sys_mat = poisson_sys_mat

    def prepare_differential_operators_for_use(self, ttype):
        diff_ops = self.get_differential_operators()
        self.diff_ops.poisson_solver = diff_ops.poisson_sys_mat.create_poisson_solver()


