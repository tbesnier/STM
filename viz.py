import polyscope as ps
import polyscope.imgui as psim
import trimesh as tri
import numpy as np
import os


ui_int = 0
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
# 9
def register_surface(name, mesh, x=0.0, y=0.0, z=0.0, idx_color=9, transparency=1.0, disp_vectors=None, disp_heatmap=None, scale_vectors = 5):
    vertices, faces = np.array(mesh.vertices), np.array(mesh.faces)
    vertices = vertices + np.stack((x*np.ones((vertices.shape[0],1)), np.zeros((vertices.shape[0],1)), np.zeros((vertices.shape[0],1))), axis=1)[:,:,0]
    vertices = vertices + np.stack((np.zeros((vertices.shape[0],1)), y*np.ones((vertices.shape[0],1)), np.zeros((vertices.shape[0],1))), axis=1)[:,:,0]
    vertices = vertices + np.stack((np.zeros((vertices.shape[0],1)), np.zeros((vertices.shape[0],1)), z*np.ones((vertices.shape[0],1))), axis=1)[:,:,0]

    mesh = ps.register_surface_mesh(name, vertices, faces, edge_width=0.)
    mesh.set_color(tuple(int(colors[idx_color][i:i + 2], 16) / 255.0 for i in (1, 3, 5)))
    mesh.set_smooth_shade(False)
    mesh.set_transparency(transparency)

    if disp_vectors is not None:
        mesh.add_vector_quantity("displacement vectors", scale_vectors * disp_vectors, enabled=True,
                                 color=tuple(int(colors[-1][i:i + 2], 16) / 255.0 for i in (1, 3, 5)), vectortype="ambient")

    if disp_heatmap is not None:
        min_bound, max_bound = disp_heatmap.min(), disp_heatmap.max()  #
        mesh.add_scalar_quantity('Varifold signature', disp_heatmap, defined_on='vertices', enabled=True, cmap='jet', vminmax=(min_bound, max_bound))

    return mesh

# Define our callback function, which Polyscope will repeatedly execute while running the UI.
def callback():

    global ui_int, meshes, heatmap

    # == Settings

    # Note that it is a push/pop pair, with the matching pop() below.
    psim.PushItemWidth(150)

    # == Show text in the UI

    psim.TextUnformatted("Sequence of meshes")
    psim.TextUnformatted("Sequence length: {}".format(len(meshes)))
    psim.Separator()

    # Input Int Slider
    changed, ui_int = psim.SliderInt("Frame", ui_int, v_min=0, v_max=len(meshes)-1)
    if changed:
        ps.remove_all_structures()
        register_surface(name=f'Step {ui_int} Ours', mesh=meshes[ui_int], idx_color=9, disp_vectors=None, disp_heatmap=None)


if __name__ == '__main__':

    meshes_dir = '../datasets/preprocessed'

    l_mesh_dir = os.listdir(meshes_dir)
    l_mesh_dir.sort()

    meshes = [tri.load(os.path.join(meshes_dir, l_mesh_dir[i])) for i in range(len(l_mesh_dir)) if 'ply' in l_mesh_dir[i]]

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_ground_plane_height_factor(0)

    register_surface(name=f'Step {0} Ours', mesh=meshes[0], idx_color=0, disp_vectors=None, disp_heatmap=None)

    ps.set_user_callback(callback)
    ps.show()
