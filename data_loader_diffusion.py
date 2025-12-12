import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
from tqdm import tqdm
import trimesh
import Get_landmarks
import models.diffusion_net as diffusion_net

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
        L_template = self.data[index]["L_template"]
        evals_template = self.data[index]["evals_template"]
        evecs_template = self.data[index]["evecs_template"]
        gradX_template = self.data[index]["gradX_template"]
        gradY_template = self.data[index]["gradY_template"]
        faces_template = self.data[index]["faces_template"]
        normals_template = self.data[index]["normals_template"]
        feats = self.data[index]["feats"]
        feats_temp = self.data[index]["feats_temp"]

        return (torch.FloatTensor(vertices), torch.FloatTensor(template), file_name, faces,
                torch.FloatTensor(np.array(mass_template)).float(),
                L_template.float(), torch.FloatTensor(np.array(evals_template)),
                torch.FloatTensor(np.array(evecs_template)), gradX_template.float(), gradY_template.float(), faces_template, torch.FloatTensor(normals), torch.FloatTensor(normals_template),
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
    L_template_dict = {}
    evals_template_dict = {}
    evecs_template_dict = {}
    gradX_template_dict = {}
    gradY_template_dict = {}
    lmk_idx = np.load("./data/lmk_noeyes_idx.npy")

    for r, ds, fs in os.walk(meshes_path):
        for f in tqdm(fs):
            if f.endswith(".ply"):
                key = f
                data[key]["name"] = f

                subject_id = "_".join(key.split("_")[:4])

                template_mesh = trimesh.load(f"{meshes_path}/{subject_id}_neutral_no_eyes.ply", process=False)
                temp = np.array(template_mesh.vertices)
                lmk_template = temp[lmk_idx]

                data[key]["template"] = temp
                normals_template = np.array(template_mesh.vertex_normals)
                data[key]["normals_template"] = normals_template

                vertices_path_ = os.path.join(meshes_path, f)

                frame, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
                    torch.tensor(temp), faces=torch.tensor(template_mesh.faces), k_eig=args.k_eig)
                mass_template_dict[subject_id] = mass
                L_template_dict[subject_id] = L
                evals_template_dict[subject_id] = evals
                evecs_template_dict[subject_id] = evecs
                gradX_template_dict[subject_id] = gradX
                gradY_template_dict[subject_id] = gradY

                data[key]["mass_template"] = mass_template_dict[subject_id]
                data[key]["L_template"] = L_template_dict[subject_id]
                data[key]["evals_template"] = evals_template_dict[subject_id]
                data[key]["evecs_template"] = evecs_template_dict[subject_id]
                data[key]["gradX_template"] = gradX_template_dict[subject_id]
                data[key]["gradY_template"] = gradY_template_dict[subject_id]
                data[key]["faces_template"] = torch.tensor(template_mesh.faces).to(dtype=torch.int64)

                if not os.path.exists(vertices_path_):
                    del data[key]
                else:
                    mesh = trimesh.load(vertices_path_)
                    vertices = np.array(mesh.vertices)
                    faces = np.array(mesh.faces)
                    def_landmarks = vertices[lmk_idx]

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


def get_dataloader(args, flag=None):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args, flag)
    train_data = Dataset(train_data, subjects_dict, "train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                       num_workers=args.num_workers)
    valid_data = Dataset(valid_data, subjects_dict, "val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=True)
    test_data = Dataset(test_data, subjects_dict, "test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=True, drop_last=False)

    return dataset


if __name__ == "__main__":

    get_dataloader()
