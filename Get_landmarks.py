import numpy as np
import trimesh
import _pickle as pickle


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

def show_point_cloud_with_mesh(points,
                               vertices=None,
                               faces=None,
                               point_size=15,
                               mesh_alpha=0.2):
    """
    Display a 3D point cloud and (optionally) a triangular mesh.

    Parameters
    ----------
    points : (N, 3) array
        Point cloud.
    vertices : (V, 3) array, optional
        Mesh vertex positions.
    faces : (F, 3) int array, optional
        Mesh triangle indices (each row: indices into `vertices`).
    point_size : float
        Size of the points in the scatter plot.
    mesh_alpha : float
        Transparency of the mesh (0 = fully transparent, 1 = opaque).
    """
    points = np.asarray(points)
    assert points.shape[1] == 3, "points should have shape (N, 3)"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # --- Plot point cloud ---
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=point_size, depthshade=True)

    # --- Plot mesh if given ---
    if vertices is not None and faces is not None:
        vertices = np.asarray(vertices)
        faces = np.asarray(faces, dtype=int)
        assert vertices.shape[1] == 3, "vertices should have shape (V, 3)"
        assert faces.shape[1] == 3, "faces should have shape (F, 3) for triangles"

        # Build a list of triangle vertex coordinates
        tri_verts = vertices[faces]  # (F, 3, 3)

        mesh = Poly3DCollection(tri_verts,
                                alpha=mesh_alpha,
                                facecolor='cyan',
                                edgecolor='k',
                                linewidths=0.2)
        ax.add_collection3d(mesh)

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud with Mesh')

    # --- Equal aspect ratio for all axes ---
    all_pts = points
    if vertices is not None:
        all_pts = np.vstack([points, vertices])

    x, y, z = all_pts[:, 0], all_pts[:, 1], all_pts[:, 2]
    max_range = np.ptp(all_pts, axis=0).max()
    mid_x = (x.max() + x.min()) / 2
    mid_y = (y.max() + y.min()) / 2
    mid_z = (z.max() + z.min()) / 2

    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    plt.tight_layout()
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

def get_landmarks_no_eyes(vertices):

    return []


if __name__ == "__main__":
    template_mesh = trimesh.load('./data/template.obj', process=False)

    neutral_mesh = trimesh.load("/media/tbesnier/T5 EVO/datasets/Face/COMA_full/FaceTalk_170725_00137_TA/bareteeth/bareteeth.000001.ply")
    neutral_mesh_no_eyes = trimesh.load("/media/tbesnier/T5 EVO/datasets/Face/COMA_noeyes/meshes/FaceTalk_170725_00137_TA/bareteeth/bareteeth.000001.ply")
    exp_mesh = trimesh.load("/media/tbesnier/T5 EVO/datasets/Face/COMA_full/FaceTalk_170725_00137_TA/bareteeth/bareteeth.000030.ply")

    lmk_neutral = get_landmarks(neutral_mesh.vertices)
    print(lmk_neutral.shape)
    show_point_cloud(lmk_neutral)
    #show_point_cloud_with_mesh(lmk_neutral, neutral_mesh.vertices, neutral_mesh.faces)

    lmk_neutral_no_eyes_idx = closest_points_indices(np.array(neutral_mesh_no_eyes.vertices), lmk_neutral)
    np.save("./data/lmk_noeyes_idx", lmk_neutral_no_eyes_idx)
    lmk_neutral_no_eyes = np.array(neutral_mesh_no_eyes.vertices)[lmk_neutral_no_eyes_idx]
    show_point_cloud(lmk_neutral_no_eyes)
    #show_point_cloud_with_mesh(lmk_neutral_no_eyes, neutral_mesh_no_eyes.vertices, neutral_mesh_no_eyes.faces)

    #delta_lmk = get_landmarks(exp_mesh.vertices) - get_landmarks(neutral_mesh.vertices)


