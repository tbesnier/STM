import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
from tqdm import tqdm
import trimesh
import Get_landmarks
import pymeshlab

import cholespy
from torch.utils.data.dataloader import default_collate
import models.poissonnet as poisson_net

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, subjects_dict, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        vertices = self.data[index]["vertices"]
        template = self.data[index]["template"]
        faces = self.data[index]["faces"]
        normals = self.data[index]["normals"]

        mass_template = self.data[index]["mass_template"]
        solver_template = self.data[index]["solver_template"]
        G_template = self.data[index]["G_template"]
        M_template = self.data[index]["M_template"]
        faces_template = self.data[index]["faces_template"]
        normals_template = self.data[index]["normals_template"]
        feats = self.data[index]["feats"]
        feats_temp = self.data[index]["feats_temp"]

        return (torch.FloatTensor(vertices), torch.FloatTensor(template), file_name, faces,
            mass_template.float(), solver_template[0], G_template.float(),
            M_template.float(), faces_template, torch.FloatTensor(normals), torch.FloatTensor(normals_template),
            torch.FloatTensor(feats), torch.FloatTensor(feats_temp))

    def __len__(self):
        return self.len


def read_data(args, flag=None):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    if flag is None:
        meshes_path = args.meshes_path
    else:
        meshes_path = args.meshes_path_remesh

    subject_id_list = []

    mass_template_dict = {}
    solver_template_dict = {}
    G_template_dict = {}
    M_template_dict = {}
    lmk_idx = np.load("./data/lmk_noeyes_idx.npy")


    for r, ds, fs in os.walk(meshes_path):
        for f in tqdm(fs):
            if f.endswith(".ply"):
                key = f
                data[key]["name"] = f

                subject_id = "_".join(key.split("_")[:4])

                template_mesh = trimesh.load(f"{meshes_path}/{subject_id}_neutral_no_eyes.ply", process=False)
                temp = np.array(template_mesh.vertices)
                faces_temp = np.array(template_mesh.faces)
                lmk_template = temp[lmk_idx]#Get_landmarks.get_landmarks(temp)#.reshape((-1))

                # ms = pymeshlab.MeshSet()
                # mesh = pymeshlab.Mesh(vertex_matrix = temp, face_matrix = template_mesh.faces)
                # ms.add_mesh(mesh)
                #
                # ms.compute_selection_by_small_disconnected_components_per_face(nbfaceratio = 0.3)
                # ms.meshing_remove_selected_vertices_and_faces()
                #
                # temp = ms.current_mesh().vertex_matrix()
                # faces_temp = ms.current_mesh().face_matrix()
                # template_mesh = trimesh.Trimesh(temp, faces_temp)

                data[key]["template"] = temp
                normals_template = np.array(template_mesh.vertex_normals)
                data[key]["normals_template"] = normals_template

                vertices_path_ = os.path.join(meshes_path, f)

                mass_src, solver_src, G_src, M_src = poisson_net.geometry.construct_mesh_operators(torch.FloatTensor(temp).to(device="cuda:0"),
                                                                                                   torch.tensor(faces_temp).to(device="cuda:0", dtype=torch.int64), high_precision=True)
                mass_template_dict[subject_id] = mass_src[0]
                solver_template_dict[subject_id] = solver_src
                G_template_dict[subject_id] = G_src[0]
                M_template_dict[subject_id] = M_src[0]

                data[key]["mass_template"] = mass_template_dict[subject_id]
                data[key]["solver_template"] = solver_template_dict[subject_id]
                data[key]["G_template"] = G_template_dict[subject_id]
                data[key]["M_template"] = M_template_dict[subject_id]
                data[key]["faces_template"] = torch.tensor(faces_temp).to(dtype=torch.int64)

                if not os.path.exists(vertices_path_):
                    del data[key]
                else:
                    mesh = trimesh.load(vertices_path_)
                    vertices = np.array(mesh.vertices)
                    faces = np.array(mesh.faces)
                    def_landmarks = vertices[lmk_idx]#Get_landmarks.get_landmarks(vertices)
                    # ms = pymeshlab.MeshSet()
                    # mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
                    # ms.add_mesh(mesh)
                    # ms.compute_selection_by_small_disconnected_components_per_face(nbfaceratio=0.3)
                    # ms.meshing_remove_selected_vertices_and_faces()
                    #vertices = ms.current_mesh().vertex_matrix()
                    #faces = ms.current_mesh().face_matrix()
                    mesh = trimesh.Trimesh(vertices, faces)
                    data[key]["vertices"] = vertices
                    data[key]["normals"] = np.array(mesh.vertex_normals)
                    data[key]["faces"] = torch.tensor(faces).to(dtype=torch.int64)

                target_feats = def_landmarks - lmk_template

                data[key]["feats"] = target_feats
                feats_temp = np.hstack([temp, normals_template])
                data[key]["feats_temp"] = feats_temp

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]
    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:4])
        if subject_id in subjects_dict["train"]:
            train_data.append(v)
        if subject_id in subjects_dict["val"]:
            valid_data.append(v)
        if subject_id in subjects_dict["test"]:
            test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data))

    return train_data, valid_data, test_data, subjects_dict


def custom_collate(batch):
    # Separate the CholeskySolverF objects from the rest
    filtered_batch = []
    solver_objects = []

    for item in batch:
        if isinstance(item, (list, tuple)):
            filtered_item = [x for x in item if not isinstance(x, cholespy.CholeskySolverF)]
            solver_objects.append([x for x in item if isinstance(x, cholespy.CholeskySolverF)])
            filtered_batch.append(filtered_item)
        else:
            filtered_batch.append(item)

    # Use default collate on the filtered batch
    collated = default_collate(filtered_batch)

    return collated, solver_objects[0]


def get_dataloader(args, flag=None):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args, flag)
    train_data = Dataset(train_data, subjects_dict, "train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                       num_workers=args.num_workers, collate_fn=custom_collate)
    valid_data = Dataset(valid_data, subjects_dict, "val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=True, collate_fn=custom_collate)
    test_data = Dataset(test_data, subjects_dict, "test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=True, drop_last=False, collate_fn=custom_collate)

    return dataset


if __name__ == "__main__":

    get_dataloader()
