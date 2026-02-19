import os
import trimesh
import numpy as np
import pymeshlab as pm

npy_path = "D:/phd_data/VOCA_training/vertices_npy"
FLAME_template = trimesh.load('./data/FLAME_sample.ply', process=False)

L = os.listdir(npy_path)
L.sort()
for i, seq in enumerate(L):
    npy = np.load(os.path.join(npy_path, seq))
    for j, frame in enumerate(npy):
        frame = frame.reshape((5023, 3))
        ms = pm.MeshSet()
        ms.add_mesh(pm.Mesh(vertex_matrix=frame, face_matrix=np.array(FLAME_template.faces)))
        ms.compute_selection_by_small_disconnected_components_per_face(nbfaceratio=0.3)
        ms.apply_filter("meshing_remove_selected_vertices_and_faces")
        mesh = trimesh.Trimesh(vertices=ms.current_mesh().vertex_matrix(),
                                      faces=ms.current_mesh().face_matrix())
        os.makedirs(os.path.join("D:/phd_data/VOCA_training/ply_no_eyes", seq[:-4]), exist_ok=True)
        mesh.export(os.path.join("D:/phd_data/VOCA_training/ply_no_eyes", seq[:-4], str(j).zfill(3) + ".ply"))