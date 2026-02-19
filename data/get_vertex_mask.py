import trimesh
import numpy as np

face_mesh = trimesh.load("D:/phd_data/VOCA_training/ply/FaceTalk_170725_00137_TA_sentence01/000.ply")  #trimesh.load('D:/phd_data/VOCA_training/ply_no_eyes/FaceTalk_170725_00137_TA_sentence01/000.ply', process=False)
vertices = face_mesh.vertices
mask_vertices = []
indices = []
for k in range(vertices.shape[0]):
    vertex = vertices[k]
    if vertex[1] >= 0.02 and vertex[2] > 0:
        mask_vertices.append(vertex)
        indices.append(k)

mask = trimesh.Trimesh(vertices=mask_vertices)
mask.export('voca/upper_face.ply')
np.save('voca/upper_face.npy', indices)

mask_vertices = []
indices = []
for k in range(vertices.shape[0]):
    vertex = vertices[k]
    if vertex[1] <= -0.02 and vertex[1] >= -0.08 and vertex[2] > 0.03:
        mask_vertices.append(vertex)
        indices.append(k)

mask = trimesh.Trimesh(vertices=mask_vertices)
mask.export('voca/mouth_indices.ply')
np.save('voca/mouth_indices.npy', indices)