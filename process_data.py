import os

import numpy as np
import trimesh
# import pymeshlab as pm

# template = trimesh.load_mesh('./data/FLAME_sample.ply')
#
# seq = np.load(f"../datasets/VOCA_training/vertices_npy/FaceTalk_170725_00137_TA_sentence01.npy")#np.load("../datasets/multiface/Aligned_with_VOCA/vertices/20171024_SEN_are_you_looking_for_employment.npy")
# seq = seq.reshape(seq.shape[0], seq.shape[1]//3, 3)
# lmk_idx = np.load("./data/lmk_noeyes_idx.npy")
# lmk_seq = []
# for frame in range(seq.shape[0]):
#     mesh = trimesh.Trimesh(seq[frame], template.faces)
#     ms = pm.MeshSet()
#     ms.add_mesh(pm.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
#     ms.compute_selection_by_small_disconnected_components_per_face(nbfaceratio=0.3)
#     ms.apply_filter("meshing_remove_selected_vertices_and_faces")
#
#     mesh = trimesh.Trimesh(vertices=ms.current_mesh().vertex_matrix(), faces=ms.current_mesh().face_matrix())
#     mesh.export(f"./data/ex_vocaset/{str(frame).zfill(6)}.ply")
#
#     lmk_seq.append(ms.current_mesh().vertex_matrix()[lmk_idx])
#
# lmk_seq = np.array(lmk_seq)
# np.save("./data/ex_vocaset_lmk.npy", lmk_seq)

lmk_seq = np.load("../datasets/ravdess/tracking_npy_aligned/Actor_01/01-01-01-01-01-01-01_aligned.npy")

os.makedirs(f"./data/gt_lmk_seq_ravdess", exist_ok=True)
for i, frame in enumerate(lmk_seq):
    mesh = trimesh.Trimesh(frame, faces=None)
    mesh.export(f"./data/gt_lmk_seq_ravdess/{str(i).zfill(3)}.ply")

#!/usr/bin/env python3
# import numpy as np
#
# def rigid_align(A: np.ndarray, B: np.ndarray):
#     """
#     Best-fit rigid transform aligning B to A with pointwise correspondences.
#     Solves: A ≈ R @ B + t
#     A, B: (N,3)
#     Returns: R (3,3), t (3,)
#     """
#     if A.shape != B.shape or A.shape[1] != 3:
#         raise ValueError("A and B must be the same shape (N, 3).")
#
#     cA = A.mean(axis=0)
#     cB = B.mean(axis=0)
#     AA = A - cA
#     BB = B - cB
#
#     H = BB.T @ AA
#     U, _, Vt = np.linalg.svd(H)
#     R = Vt.T @ U.T
#
#     # Fix reflection if needed
#     if np.linalg.det(R) < 0:
#         Vt[-1, :] *= -1
#         R = Vt.T @ U.T
#
#     t = cA - R @ cB
#     return R, t
#
# def apply_transform(P: np.ndarray, R: np.ndarray, t: np.ndarray):
#     return (R @ P.T).T + t
#
# def write_ply_xyz(path: str, points: np.ndarray):
#     """
#     Write Nx3 points to an ASCII PLY file.
#     """
#     points = np.asarray(points, dtype=np.float64)
#     if points.ndim != 2 or points.shape[1] != 3:
#         raise ValueError("points must be (N,3)")
#
#     header = "\n".join([
#         "ply",
#         "format ascii 1.0",
#         f"element vertex {points.shape[0]}",
#         "property float x",
#         "property float y",
#         "property float z",
#         "end_header"
#     ])
#
#     with open(path, "w") as f:
#         f.write(header + "\n")
#         np.savetxt(f, points, fmt="%.8f %.8f %.8f")
#
# def main():
#     import argparse
#     parser = argparse.ArgumentParser(
#         description="Align last frame to first frame from a (T,N,3) .npy sequence and export aligned last frame as .ply."
#     )
#     parser.add_argument("--seq", type=str, default="../datasets/ravdess/tracking_npy/Actor_01/01-01-01-01-01-01-01_aligned.npy",
#                         help="Path to Nx3 numpy file for reference cloud A (.npy or .txt)")
#     parser.add_argument("--out_ply", default="last_aligned_to_first.ply", help="Output .ply for aligned last frame")
#     parser.add_argument("--save_T", default="T_last_to_first.txt", help="Output 4x4 transform matrix (txt)")
#     parser.add_argument("--i", type=int, default=0, help="Index of reference frame A (default: 0)")
#     parser.add_argument("--j", type=int, default=-1, help="Index of source frame B to align (default: -1)")
#     args = parser.parse_args()
#
#     seq = np.load(args.seq)
#     if seq.ndim != 3 or seq.shape[2] != 3:
#         raise SystemExit(f"Expected shape (T,N,3), got {seq.shape}")
#
#     T_frames, N, _ = seq.shape
#
#     A = np.asarray(seq[args.i], dtype=np.float64)
#     B = np.asarray(seq[args.j], dtype=np.float64)
#
#     R, t = rigid_align(A, B)
#     B_aligned = apply_transform(B, R, t)
#
#     # RMS alignment error on correspondences
#     rms = np.sqrt(np.mean(np.sum((A - B_aligned) ** 2, axis=1)))
#
#     # Save aligned last frame as PLY
#     write_ply_xyz(args.out_ply, B_aligned)
#
#     # Save transform matrix (B -> A)
#     T = np.eye(4, dtype=np.float64)
#     T[:3, :3] = R
#     T[:3, 3] = t
#     np.savetxt(args.save_T, T, fmt="%.10f")
#
#     print(f"Sequence shape: {seq.shape} (T={T_frames}, N={N})")
#     print(f"Aligned frame j={args.j} to frame i={args.i}")
#     print(f"RMS error: {rms:.6f}")
#     print(f"Wrote: {args.out_ply}")
#     print(f"Wrote transform: {args.save_T}")
#
# if __name__ == "__main__":
#     main()

