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

import pyvista as pv
m = pv.read("./data/ex_vocaset/000000.ply")
p = pv.Plotter()
p.add_mesh(m)
p.show()
print(p.camera_position)