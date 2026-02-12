import numpy as np
import trimesh
import os

def rigid_align_landmarks(ref: np.ndarray, src: np.ndarray, return_transform: bool = False):
    """
    Rigidly align src landmarks to ref landmarks (rotation + translation, no scaling).

    Args:
        ref: (L, 3) reference landmarks
        src: (L, 3) source landmarks to align onto ref
        return_transform: if True, also return (R, t) such that src_aligned = src @ R.T + t

    Returns:
        src_aligned: (L, 3) aligned source landmarks
        (optional) R: (3, 3) rotation matrix
        (optional) t: (3,) translation vector
    """
    ref = np.asarray(ref, dtype=float)
    src = np.asarray(src, dtype=float)
    if ref.shape != src.shape or ref.shape[1] != 3:
        raise ValueError(f"Expected both inputs to have shape (L, 3). Got {ref.shape} and {src.shape}.")

    # 1) subtract centroids
    c_ref = ref.mean(axis=0)
    c_src = src.mean(axis=0)
    X = ref - c_ref
    Y = src - c_src

    # 2) compute optimal rotation via SVD
    H = Y.T @ X  # (3,3)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 3) fix improper rotation (reflection) if det(R) < 0
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 4) translation
    t = c_ref - R @ c_src

    # Apply transform (note: src points are row-vectors)
    src_aligned = (src @ R.T) + t

    if return_transform:
        return src_aligned, R, t
    return src_aligned

lmk_flame_no_eyes = np.load("./data/lmk_noeyes_idx.npy")
ref = trimesh.load("./data/template_flame_noeyes.ply").vertices[lmk_flame_no_eyes]

os.makedirs("D:/phd_data/ravdess/lmks_npy_aligned/Actor_01/", exist_ok=True)
L = os.listdir("D:/phd_data/ravdess/tracking_npy_aligned/Actor_01")
L.sort()
for k, src_seq_path in enumerate(L):

    src_seq = np.load(f"D:/phd_data/ravdess/tracking_npy_aligned/Actor_01/{src_seq_path}")
    src = src_seq[0]
    print(src_seq.shape)

    aligned, R, t = rigid_align_landmarks(ref, src, return_transform=True)
    aligned_seq = np.zeros((src_seq.shape[0]+1, src_seq.shape[1], src_seq.shape[2]))
    aligned_seq[0] = ref

    for idx, frame in enumerate(src_seq):
        aligned = (frame @ R.T) + t
        aligned_seq[idx+1] = aligned

    np.save(f"D:/phd_data/ravdess/lmks_npy_aligned/Actor_01/{src_seq_path}", aligned_seq)

#aligned = trimesh.Trimesh(vertices=aligned, faces=None, process=False)
#aligned.export("./test.ply")
