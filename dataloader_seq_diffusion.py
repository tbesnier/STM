from __future__ import annotations

import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import trimesh
import models.diffusion_net as diffusion_net
from pathlib import Path
from typing import List, Tuple, Union, Sequence, Any

DEFAULT_MESH_EXTS: Tuple[str, ...] = (".ply", ".obj", ".off", ".stl", ".glb", ".gltf")
exts: Tuple[str, ...] = tuple(DEFAULT_MESH_EXTS)


# ----------------------------
# Path helpers (fix Path/str issues)
# ----------------------------
PathLike = Union[str, Path]


def as_path(p: PathLike) -> Path:
    """Convert str/Path to Path (idempotent)."""
    return p if isinstance(p, Path) else Path(p)


def ensure_dir(p: PathLike, name: str) -> Path:
    """Convert to Path and validate it's an existing directory."""
    pp = as_path(p).expanduser()
    if not pp.exists():
        raise FileNotFoundError(f"{name} does not exist: {pp}")
    if not pp.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {pp}")
    return pp


def ensure_file(p: PathLike, name: str) -> Path:
    """Convert to Path and validate it's an existing file."""
    pp = as_path(p).expanduser()
    if not pp.exists():
        raise FileNotFoundError(f"{name} does not exist: {pp}")
    if not pp.is_file():
        raise FileNotFoundError(f"{name} is not a file: {pp}")
    return pp


# ----------------------------
# IO utilities
# ----------------------------
def iter_mesh_files(folder: Path, exts: Tuple[str, ...]) -> List[Path]:
    folder = ensure_dir(folder, "sequence_dir")
    exts_lc = tuple(e.lower() for e in exts)
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts_lc])


def iter_template_meshes(templates_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    templates_dir = ensure_dir(templates_dir, "templates_dir")
    files: List[Path] = []
    # glob is case-sensitive on some OS; include upper-case extensions too
    for ext in exts:
        files.extend(templates_dir.glob(f"*{ext}"))
        files.extend(templates_dir.glob(f"*{ext.upper()}"))
    return sorted(set(files), key=lambda p: p.name.lower())


def find_matching_sequence_dirs(seqs_root: Path, template_key: str) -> List[Path]:
    seqs_root = ensure_dir(seqs_root, "deformations_dir")
    key = template_key.lower()
    matches = [d for d in seqs_root.iterdir() if d.is_dir() and key in d.name.lower()]
    return sorted(matches, key=lambda p: p.name.lower())


# ----------------------------
# Dataset wrapper
# ----------------------------
class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_list, subjects_dict, data_type: str = "train"):
        self.data = data_list
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type

    def __getitem__(self, index):
        file_name = self.data[index]["name"]
        vertices = self.data[index]["seq_vertices"]
        template = self.data[index]["template"]
        faces = self.data[index]["seq_faces"]
        normals = self.data[index]["seq_normals"]

        mass_template = self.data[index]["mass_template"]
        L_template = self.data[index]["L_template"]
        evals_template = self.data[index]["evals_template"]
        evecs_template = self.data[index]["evecs_template"]
        gradX_template = self.data[index]["gradX_template"]
        gradY_template = self.data[index]["gradY_template"]
        faces_template = self.data[index]["faces_template"]
        normals_template = self.data[index]["normals_template"]
        feats = self.data[index]["seq_feats"]
        feats_temp = self.data[index]["feats_temp"]

        return (
            torch.FloatTensor(vertices),
            torch.FloatTensor(template),
            file_name,
            torch.tensor(faces).to(dtype=torch.int64),
            torch.FloatTensor(np.array(mass_template)).float(),
            L_template.float(),
            torch.FloatTensor(np.array(evals_template)),
            torch.FloatTensor(np.array(evecs_template)),
            gradX_template.float(),
            gradY_template.float(),
            torch.tensor(faces_template).to(dtype=torch.int64),
            torch.FloatTensor(normals),
            torch.FloatTensor(normals_template),
            torch.FloatTensor(feats),
            torch.FloatTensor(feats_temp),
        )

    def __len__(self):
        return self.len


# ----------------------------
# Main loader
# ----------------------------
def _subjects_dict_from_args(args) -> dict:
    return {
        "train": [i for i in str(args.train_subjects).split(" ") if i],
        "val": [i for i in str(args.val_subjects).split(" ") if i],
        "test": [i for i in str(args.test_subjects).split(" ") if i],
    }


def read_data(args, flag=None):
    print("Loading data...")
    data_store = defaultdict(dict)
    train_data, valid_data, test_data = [], [], []

    # ----------------------------
    # COMA
    # ----------------------------
    templates_dir_coma = ensure_dir(args.templates_dir_COMA, "templates_dir_COMA")
    deformations_dir_coma = ensure_dir(args.deformations_dir_COMA, "deformations_dir_COMA")

    # np.load accepts Path; keep it as Path for correctness
    lmk_idx = np.load(as_path("./data/lmk_noeyes_idx.npy"))

    template_files_coma = iter_template_meshes(templates_dir_coma, exts)

    for tmpl_path in template_files_coma:
        template_name = tmpl_path.stem

        template_mesh = trimesh.load(str(tmpl_path), process=False)
        temp = np.array(template_mesh.vertices)
        lmk_template = temp[lmk_idx]
        normals_template = np.array(template_mesh.vertex_normals)

        _, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
            torch.tensor(temp),
            faces=torch.tensor(template_mesh.faces),
            k_eig=args.k_eig,
        )

        seq_dirs = find_matching_sequence_dirs(deformations_dir_coma, template_name)
        for seq_dir in seq_dirs:
            frame_files = iter_mesh_files(seq_dir, exts)
            print(f"  Sequence dir: {seq_dir.name}  ({len(frame_files)} frames)")

            data_store[seq_dir.name]["name"] = seq_dir.name
            data_store[seq_dir.name]["template"] = temp
            data_store[seq_dir.name]["normals_template"] = normals_template

            data_store[seq_dir.name]["mass_template"] = mass
            data_store[seq_dir.name]["L_template"] = L
            data_store[seq_dir.name]["evals_template"] = evals
            data_store[seq_dir.name]["evecs_template"] = evecs
            data_store[seq_dir.name]["gradX_template"] = gradX
            data_store[seq_dir.name]["gradY_template"] = gradY
            data_store[seq_dir.name]["faces_template"] = torch.tensor(template_mesh.faces).to(dtype=torch.int64)

            feats_temp = np.hstack([temp, normals_template])
            data_store[seq_dir.name]["feats_temp"] = feats_temp

            target_vertices_seq, target_faces_seq, target_feats_seq, target_normals_seq = [], [], [], []
            for f in frame_files:
                mesh = trimesh.load(str(f), process=False)
                vertices = np.array(mesh.vertices)
                target_vertices_seq.append(vertices)

                faces = np.array(mesh.faces)
                target_faces_seq.append(faces)

                normals = np.array(mesh.vertex_normals)
                target_normals_seq.append(normals)

                def_landmarks = vertices[lmk_idx]
                target_feats = def_landmarks - lmk_template
                target_feats_seq.append(target_feats)

            data_store[seq_dir.name]["seq_vertices"] = np.array(target_vertices_seq)
            data_store[seq_dir.name]["seq_normals"] = np.array(target_normals_seq)
            data_store[seq_dir.name]["seq_faces"] = np.array(target_faces_seq)
            data_store[seq_dir.name]["seq_feats"] = np.array(target_feats_seq)

    subjects_dict = _subjects_dict_from_args(args)
    for k, v in data_store.items():
        subject_id = "_".join(k.split("_")[:4])
        if subject_id in subjects_dict["train"]:
            train_data.append(v)
        if subject_id in subjects_dict["val"]:
            valid_data.append(v)
        if subject_id in subjects_dict["test"]:
            test_data.append(v)

    # ----------------------------
    # ICT
    # ----------------------------
    templates_dir_ict = ensure_dir(args.templates_dir_ICT, "templates_dir_ICT")
    deformations_dir_ict = ensure_dir(args.deformations_dir_ICT, "deformations_dir_ICT")
    lmk_idx = np.load(as_path("./data/lmk_ds_ict.npy"))

    template_files_ict = iter_template_meshes(templates_dir_ict, exts)

    for tmpl_path in template_files_ict:
        template_name = tmpl_path.stem

        template_mesh = trimesh.load(str(tmpl_path), process=False)
        temp = np.array(template_mesh.vertices)
        lmk_template = temp[lmk_idx]
        normals_template = np.array(template_mesh.vertex_normals)

        _, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
            torch.tensor(temp),
            faces=torch.tensor(template_mesh.faces),
            k_eig=args.k_eig,
        )

        seq_dirs = find_matching_sequence_dirs(deformations_dir_ict, template_name)
        for seq_dir in seq_dirs:
            frame_files = iter_mesh_files(seq_dir, exts)
            print(f"  Sequence dir: {seq_dir.name}  ({len(frame_files)} frames)")

            data_store[seq_dir.name]["name"] = seq_dir.name
            data_store[seq_dir.name]["template"] = temp
            data_store[seq_dir.name]["normals_template"] = normals_template

            data_store[seq_dir.name]["mass_template"] = mass
            data_store[seq_dir.name]["L_template"] = L
            data_store[seq_dir.name]["evals_template"] = evals
            data_store[seq_dir.name]["evecs_template"] = evecs
            data_store[seq_dir.name]["gradX_template"] = gradX
            data_store[seq_dir.name]["gradY_template"] = gradY
            data_store[seq_dir.name]["faces_template"] = torch.tensor(template_mesh.faces).to(dtype=torch.int64)

            feats_temp = np.hstack([temp, normals_template])
            data_store[seq_dir.name]["feats_temp"] = feats_temp

            target_vertices_seq, target_faces_seq, target_feats_seq, target_normals_seq = [], [], [], []
            for f in frame_files:
                mesh = trimesh.load(str(f), process=False)
                vertices = np.array(mesh.vertices)
                target_vertices_seq.append(vertices)

                faces = mesh.faces
                target_faces_seq.append(faces)

                normals = np.array(mesh.vertex_normals)
                target_normals_seq.append(normals)

                def_landmarks = vertices[lmk_idx]
                target_feats = def_landmarks - lmk_template
                target_feats_seq.append(target_feats)

            data_store[seq_dir.name]["seq_vertices"] = np.array(target_vertices_seq)
            data_store[seq_dir.name]["seq_normals"] = np.array(target_normals_seq)
            data_store[seq_dir.name]["seq_faces"] = np.array(target_faces_seq)
            data_store[seq_dir.name]["seq_feats"] = np.array(target_feats_seq)

    # Re-use the same subjects_dict (assumes args.* subjects apply to both datasets)
    subjects_dict = _subjects_dict_from_args(args)

    for k, v in data_store.items():
        subject_id = "_".join(k.split("_")[:2])
        if subject_id in subjects_dict["train"]:
            train_data.append(v)
        if subject_id in subjects_dict["val"]:
            valid_data.append(v)
        if subject_id in subjects_dict["test"]:
            test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict

# ----------------------------
# Collate function (handles sparse tensors returned by diffusion operators)
# ----------------------------
def custom_collate(batch):
    """Custom collate_fn for PyTorch DataLoader.

    - Stacks dense tensors if they have identical shapes.
    - Keeps sparse tensors (e.g., Laplacians/grad operators) as a Python list (one per sample).
    - Keeps non-tensor fields (e.g., strings, numpy arrays, variable-length items) as a list.
    """
    if len(batch) == 0:
        return batch

    # batch is a list of tuples -> transpose to per-field lists
    fields = list(zip(*batch))
    out = []

    for items in fields:
        first = items[0]

        if isinstance(first, torch.Tensor):
            # Sparse tensors cannot be stacked by default collate
            if first.is_sparse or (hasattr(first, "layout") and str(first.layout).startswith("torch.sparse")):
                out.append(list(items))
                continue

            # Only stack if all tensors share the same shape
            same_shape = all(isinstance(x, torch.Tensor) and x.shape == first.shape for x in items)
            if same_shape:
                out.append(torch.stack(list(items), dim=0))
            else:
                out.append(list(items))
        else:
            out.append(list(items))

    return tuple(out)




def get_dataloader(args, flag=None):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args, flag)

    train_ds = Dataset(train_data, subjects_dict, "train")
    dataset["train"] = data.DataLoader(
        dataset=train_ds,
        batch_size=args.batch_size,
        collate_fn=custom_collate,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    valid_ds = Dataset(valid_data, subjects_dict, "val")
    dataset["valid"] = data.DataLoader(dataset=valid_ds, batch_size=1, collate_fn=custom_collate, shuffle=True)

    test_ds = Dataset(test_data, subjects_dict, "test")
    dataset["test"] = data.DataLoader(dataset=test_ds, batch_size=1, collate_fn=custom_collate, shuffle=True, drop_last=False)

    return dataset
