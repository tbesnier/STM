import numpy as np
import torch
import potpourri3d as pp3d

def norm(x, highdim=False):
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return torch.norm(x, dim=len(x.shape) - 1)

def compute_centroids(vertices: torch.Tensor, faces: torch.Tensor):
    """
    Compute centroids, face normals, and areas for a batch of triangular meshes.

    Args:
        vertices (torch.Tensor): Tensor of shape (B, V, 3) representing vertex positions for B meshes.
        faces (torch.Tensor): Tensor of shape (B, F, 3) representing indices of vertices forming faces for B meshes.

    Returns:
        centroids (torch.Tensor): Tensor of shape (B, F, 3) representing face centroids.
        normals (torch.Tensor): Tensor of shape (B, F, 3) representing face normals (not necessarily unit length).
        areas (torch.Tensor): Tensor of shape (B, F) representing face areas.
    """
    B = vertices.shape[0]

    # Gather vertex positions for each face correctly
    v0 = torch.gather(vertices, 1, faces[..., 0].unsqueeze(-1).expand(-1, -1, 3))
    v1 = torch.gather(vertices, 1, faces[..., 1].unsqueeze(-1).expand(-1, -1, 3))
    v2 = torch.gather(vertices, 1, faces[..., 2].unsqueeze(-1).expand(-1, -1, 3))

    # Compute centroids
    centroids = (v0 + v1 + v2) / 3.0

    return centroids

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

def cpt_geodesics_from_point(mesh, index):
    V, F = np.array(mesh.vertices), np.array(mesh.faces)
    solver = pp3d.MeshHeatMethodDistanceSolver(V, F)

    return solver.compute_distance(index)
