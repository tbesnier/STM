import numpy as np
import trimesh

template = trimesh.load_mesh('../datasets/multiface/Aligned_with_VOCA/templates/20171024.ply')

seq = np.load("../datasets/multiface/Aligned_with_VOCA/vertices/20171024_SEN_are_you_looking_for_employment.npy")
for frame in range(seq.shape[0]):
    mesh = trimesh.Trimesh(seq[frame], template.faces)
    mesh.export(f"./data/ex_multiface/{str(frame).zfill(6)}.ply")