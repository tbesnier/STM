import numpy as np
import os
import argparse
from scipy.io import loadmat
# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
from aitviewer.renderables.meshes import Meshes
import trimesh

from aitviewer.viewer import Viewer


def render(args):
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        'burlywood', 'cadetblue',
        'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue',
        'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
        'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey',
        'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', '#000000'
    ]

    meshes_dir = '../../papers/PaNDaS/comparison_videos/partial_MANO_scan' #'../Data/LISC/results_dual_MANO/MANO_anim'
    #meshes_dir = "/home/tbesnier/phd/projects/Data/VOCA/res/Results_Actor/Meshes50"
    mesh_list = [os.path.join(meshes_dir, os.listdir(meshes_dir)[i]) for i in range(len(os.listdir(meshes_dir)))]
    mesh_list.sort()

    # Number of frames.
    N = len(mesh_list)
    print(mesh_list)

    vertices = np.array([trimesh.load(mesh_list[i]).vertices for i in range(N)])
    c = tuple(int(colors[5][i:i + 2], 16) / 255.0 for i in (1, 3, 5))

    seq = Meshes(
        vertices,
        trimesh.load(mesh_list[0]).faces,
        name="Prediction PN",
        position=[0, 0, 0],
        flat_shading=True,
        color=tuple((c[0], c[1], c[2], 1))
    )

    viewer = Viewer()
    viewer.scene.add(seq)

    viewer.auto_set_camera_target = True
    #seq.norm_coloring = True

    viewer.scene.origin.enabled = False
    viewer.playback_fps = 30

    # Set the camera position.
    #viewer.scene.camera.position = (-0.014, 0.058, 0.210)
    #viewer.scene.camera.target = (-0.009, 0.039, -0.005)

    viewer.run_animations = True
    viewer.run()


def main():
    parser = argparse.ArgumentParser(description='3D Visualization for skeleton dual motions')

    parser.add_argument("--export_video", type=str, default="./anim.mp4")

    args = parser.parse_args()
    render(args)


if __name__ == "__main__":
    main()


