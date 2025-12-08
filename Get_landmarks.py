import numpy as np
import trimesh
import _pickle as pickle


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

def show_point_cloud_with_edges(points,
                                edges,
                                point_size=5,
                                edge_width=1.0,
                                point_color="tab:blue",
                                edge_color="tab:gray",
                                equal_axes=True, target_points=None):
    """
    Display a 3D point cloud and edges.

    Parameters
    ----------
    points : (N, 3) array-like
        Array of point positions (x, y, z).
    edges : (M, 2) array-like of int
        Each row [i, j] indicates an edge between points[i] and points[j].
    point_size : float, optional
        Size of the points in the scatter plot.
    edge_width : float, optional
        Line width of the edges.
    point_color : str or array-like, optional
        Color for the points (any Matplotlib color).
    edge_color : str or array-like, optional
        Color for the edges (any Matplotlib color).
    equal_axes : bool, optional
        If True, sets equal scale on x, y, z axes.
    """
    points = np.asarray(points)
    edges = np.asarray(edges, dtype=int)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be of shape (N, 3)")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must be of shape (M, 2) with integer indices")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=point_size, c="red")

    if target_points is not None:
        ax.quiver3D(points[:, 0],points[:, 1], points[:,2],
                    target_points[:, 0], target_points[:, 1], target_points[:, 2])

    # Plot edges
    for i, j in edges:
        xs = [points[i, 0], points[j, 0]]
        ys = [points[i, 1], points[j, 1]]
        zs = [points[i, 2], points[j, 2]]
        ax.plot(xs, ys, zs, linewidth=edge_width, color=edge_color)

    # Optionally set equal aspect ratio
    if equal_axes:
        _set_equal_3d_axes(ax, points)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.tight_layout()
    plt.show()


def _set_equal_3d_axes(ax, points):
    """
    Set equal aspect ratio for a 3D axis based on the extent of the points.
    """
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def show_point_cloud(points):
    """
    points: numpy array of shape (N, 3)
    """
    points = np.asarray(points)
    assert points.shape[1] == 3, "points should have shape (N, 3)"

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, s=15)  # s is point size

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')

    # Optionally set equal aspect ratio
    max_range = np.ptp(points, axis=0).max()
    mid_x = (x.max() + x.min()) / 2
    mid_y = (y.max() + y.min()) / 2
    mid_z = (z.max() + z.min()) / 2

    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    plt.show()


def load_static_embedding(static_embedding_path):
    with open(static_embedding_path, 'rb') as f:
        lmk_indexes_dict = pickle.load(f, encoding='latin1')
    lmk_face_idx = lmk_indexes_dict['lmk_face_idx'].astype(np.uint32)
    lmk_b_coords = lmk_indexes_dict['lmk_b_coords']
    return lmk_face_idx, lmk_b_coords


def mesh_points_by_barycentric_coordinates(mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords):
    # function: evaluation 3d points given mesh and landmark embedding
    # modified from https://github.com/Rubikplayer/flame-fitting/blob/master/fitting/landmarks.py
    dif1 = np.vstack([(mesh_verts[mesh_faces[lmk_face_idx], 0] * lmk_b_coords).sum(axis=1),
                      (mesh_verts[mesh_faces[lmk_face_idx], 1] * lmk_b_coords).sum(axis=1),
                      (mesh_verts[mesh_faces[lmk_face_idx], 2] * lmk_b_coords).sum(axis=1)]).T
    return dif1


def load_dynamic_contour(vertices, faces, contour_embeddings_path='None', static_embedding_path='None', angle=0):
    contour_embeddings_path = contour_embeddings_path
    dynamic_lmks_embeddings = np.load(contour_embeddings_path, allow_pickle=True, encoding='latin1').item()
    lmk_face_idx_static, lmk_b_coords_static = load_static_embedding(static_embedding_path)
    lmk_face_idx_dynamic = dynamic_lmks_embeddings['lmk_face_idx'][angle]
    lmk_b_coords_dynamic = dynamic_lmks_embeddings['lmk_b_coords'][angle]
    dynamic_lmks = mesh_points_by_barycentric_coordinates(vertices, faces, lmk_face_idx_dynamic, lmk_b_coords_dynamic)
    static_lmks = mesh_points_by_barycentric_coordinates(vertices, faces, lmk_face_idx_static, lmk_b_coords_static)
    total_lmks = np.vstack([dynamic_lmks, static_lmks])

    return total_lmks


def process_landmarks(landmarks):
    points = np.zeros(np.shape(landmarks))
    ## Centering
    mu_x = np.mean(landmarks[:, 0])
    mu_y = np.mean(landmarks[:, 1])
    mu_z = np.mean(landmarks[:, 2])
    mu = [mu_x, mu_y, mu_z]

    landmarks_gram = np.zeros(np.shape(landmarks))
    for j in range(np.shape(landmarks)[0]):
        landmarks_gram[j, :] = np.squeeze(landmarks[j, :])# - np.transpose(mu)

    #normFro = np.sqrt(np.trace(np.matmul(landmarks_gram, np.transpose(landmarks_gram))))
    land = landmarks_gram* 1.0# / normFro
    points[:, :] = land
    return points


def get_landmarks(vertices):
    angle = 0.0  # in degrees
    if angle < 0:
        angle = 39 - angle
    contour_embeddings_path = './data/flame_dynamic_embedding.npy'
    static_embedding_path = './data/flame_static_embedding.pkl'
    template_mesh = trimesh.load('./data/FLAME_sample.ply', process=False)
    faces = template_mesh.faces
    total_lmks = load_dynamic_contour(vertices, faces, contour_embeddings_path=contour_embeddings_path,
                                      static_embedding_path=static_embedding_path, angle=int(angle))
    total_lmks = process_landmarks(total_lmks)
    return total_lmks


def closest_points_indices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    A: (N, 3) point cloud
    B: (M, 3) point cloud, with M <= N

    Returns:
        indices: (M, 1) array of indices into A
                 indices[i, 0] is the index of the point in A
                 closest to B[i]
    """
    A = np.asarray(A)  # (N, 3)
    B = np.asarray(B)  # (M, 3)

    # Compute pairwise squared distances between B and A:
    # diff has shape (M, N, 3)
    diff = A[None, :, :] - B[:, None, :]
    dist2 = np.sum(diff ** 2, axis=2)  # (M, N)

    # For each point in B, find index of closest point in A
    idx = np.argmin(dist2, axis=1)  # (M,)

    # Return as (M, 1)
    return idx


if __name__ == "__main__":
    template_mesh = trimesh.load('./data/template.obj', process=False)

    neutral_mesh = trimesh.load("/media/tbesnier/T5 EVO/datasets/Face/COMA_full/FaceTalk_170725_00137_TA/bareteeth/bareteeth.000001.ply")
    neutral_mesh_no_eyes = trimesh.load("/media/tbesnier/T5 EVO/datasets/Face/COMA_noeyes/meshes/FaceTalk_170725_00137_TA/bareteeth/bareteeth.000001.ply")
    exp_mesh_no_eyes = trimesh.load("/media/tbesnier/T5 EVO/datasets/Face/COMA_noeyes/meshes/FaceTalk_170725_00137_TA/bareteeth/bareteeth.000030.ply")

    lmk_neutral = get_landmarks(neutral_mesh.vertices)
    #print(lmk_neutral.shape)
    #show_point_cloud(lmk_neutral)
    #show_point_cloud_with_mesh(lmk_neutral, neutral_mesh.vertices, neutral_mesh.faces)

    lmk_neutral_no_eyes_idx = closest_points_indices(np.array(neutral_mesh_no_eyes.vertices), lmk_neutral)
    np.save("./data/lmk_noeyes_idx", lmk_neutral_no_eyes_idx)
    lmk_neutral_no_eyes = np.array(neutral_mesh_no_eyes.vertices)[lmk_neutral_no_eyes_idx]
    lmk_exp_no_eyes = np.array(exp_mesh_no_eyes.vertices)[lmk_neutral_no_eyes_idx]
    edges = np.array([
        [0, 1], [1, 2], [2,3], [3,4], [4,5], [5,6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
        [17, 18], [18, 19], [19, 20], [20, 21], [22, 23], [23, 24], [24, 25], [25, 26],
        [27, 28], [28, 29], [29, 30], [31, 32], [32, 33], [33, 34], [34, 35],
        [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],
        [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],
        [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 60], [60, 48],
        [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60]
    ])
    show_point_cloud_with_edges(lmk_neutral_no_eyes, edges, point_size=15, target_points=lmk_exp_no_eyes - lmk_neutral_no_eyes)
    #show_point_cloud_with_mesh(lmk_neutral_no_eyes, neutral_mesh_no_eyes.vertices, neutral_mesh_no_eyes.faces)

    #delta_lmk = get_landmarks(exp_mesh.vertices) - get_landmarks(neutral_mesh.vertices)


