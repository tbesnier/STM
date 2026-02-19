import os

import trimesh
import numpy as np

dataset = "voca"

L = os.listdir(f"D:/EmoScanTalk/comparisons/ours/{dataset}/Meshes")
L.sort()
os.makedirs(f"D:/EmoScanTalk/comparisons/ours/{dataset}/npy", exist_ok=True)
for i, sent in enumerate(L):
    #if "FaceTalk_170809_00138_TA" in sent or "FaceTalk_170731_00024_TA" in sent:
    list_meshes = []
    M = os.listdir(os.path.join(f"D:/EmoScanTalk/comparisons/ours/{dataset}/Meshes", sent))
    M.sort()
    for j, frame in enumerate(M):
        mesh = trimesh.load(os.path.join(f"D:/EmoScanTalk/comparisons/ours/{dataset}/Meshes", sent, frame))
        list_meshes.append(mesh.vertices)

    list_meshes = np.array(list_meshes)
    np.save(f"D:/EmoScanTalk/comparisons/ours/{dataset}/npy/{sent}.npy", list_meshes)