import sys, os, glob
import trimesh
import numpy as np
import torch
import torch.nn as nn
import argparse
import pickle
import pymeshlab as pm

from models.deformer_seq_diffusionnet import DiffusionNetAutoencoder

def infer_seq(args, seq="positions"):
    import models.diffusion_net as diffusion_net

    model = DiffusionNetAutoencoder(args).to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])

    reference = trimesh.load('./data/FLAME_sample.ply',
                             process=False)
    template_tri = reference.faces

    lmk_paths = os.path.join(args.data_path, "landmark_npy")
    L = os.listdir(lmk_paths)
    L.sort()

    with open("D:/phd_data/VOCA_training/templates.pkl", 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    model.eval()
    with torch.no_grad():
        for i, lmk_seq in enumerate(L):
            name = lmk_seq[:-4]
            identity = "_".join(name.split("_")[:-1])
            print(identity)

            if identity == "FaceTalk_170809_00138_TA" or identity == "FaceTalk_170731_00024_TA":
                ms = pm.MeshSet()
                ms.add_mesh(pm.Mesh(vertex_matrix=templates[identity], face_matrix=template_tri))
                ms.compute_selection_by_small_disconnected_components_per_face(nbfaceratio=0.3)
                ms.apply_filter("meshing_remove_selected_vertices_and_faces")
                source_mesh = trimesh.Trimesh(vertices=ms.current_mesh().vertex_matrix(),
                                       faces=ms.current_mesh().face_matrix())
                temp = np.array(source_mesh.vertices)
                faces_template = np.array(source_mesh.faces)
                target_seq = np.load(os.path.join(lmk_paths, lmk_seq))
                if target_seq.shape[-1] != 3:
                    target_seq = target_seq.reshape(target_seq.shape[0], 68, 3)
                print(f"target_seq: {target_seq.shape}")
                if seq == "positions":
                    lmk_0 = target_seq[0]

                template = torch.FloatTensor(source_mesh.vertices).unsqueeze(0).to(args.device)
                faces_template = torch.tensor(faces_template).unsqueeze(0).to(dtype=torch.int64, device=args.device)
                normals_template = torch.FloatTensor(source_mesh.vertex_normals).unsqueeze(0).to(args.device)

                frame, mass_src, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
                    torch.tensor(temp), faces=torch.tensor(source_mesh.faces), k_eig=args.k_eig)

                mass_template = mass_src.float().to(args.device).unsqueeze(0)
                L_template = L.float().to(args.device).unsqueeze(0)
                evals_template = evals.float().to(args.device).unsqueeze(0)
                evecs_template = evecs.float().to(args.device).unsqueeze(0)
                gradX_template = gradX.float().to(args.device).unsqueeze(0)
                gradY_template = gradY.float().to(args.device).unsqueeze(0)

                in_features_template = torch.cat((template, normals_template), dim=2)
                target_feats_seq = torch.zeros((target_seq.shape[0], 68, 3)).to(args.device)

                for i, frame in enumerate(target_seq):
                    if seq == "positions":
                        target_feats = torch.FloatTensor(frame - lmk_0).unsqueeze(0).to(args.device)
                    else:
                        target_feats = torch.FloatTensor(frame).unsqueeze(0).to(args.device)

                    target_feats_seq[i] = target_feats

                vertices_pred = model.forward_latent_njf(
                    in_features_template,
                    mass_template, L_template, evals_template, evecs_template, gradX_template, gradY_template,
                    faces_template, target_feats_seq.unsqueeze(0))

                gen_seq = vertices_pred.cpu().detach().numpy()

                os.makedirs(os.path.join(args.save_path, 'Meshes', name), exist_ok=True)
                for k in range(len(gen_seq)):
                    tri_mesh_mixture = trimesh.Trimesh(np.array(gen_seq[k]), np.asarray(source_mesh.faces), process=False)
                    tri_mesh_mixture.export(os.path.join(args.save_path, 'Meshes', name, str(k).zfill(3) + ".ply"))
                    print("Saved at " + os.path.join(args.save_path, 'Meshes', name, str(k).zfill(3) + ".ply"))



def main():
    parser = argparse.ArgumentParser(description='D2D: Dense to Dense Encoder-Decoder')

    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=float, default=1)

    # data args
    parser.add_argument('--template_file', type=str, default='./data/template.obj')

    parser.add_argument("--save_path", type=str, default="D:/EmoScanTalk/comparisons/ours/voca",
                        help='path to save the results')
    parser.add_argument("--data_path", type=str, default="D:/phd_data/VOCA_training/")

    # checkpoint args
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--models_dir", type=str, default="../Data/STM/Models")
    parser.add_argument("--model_path", type=str, default="../Data/STM/Models/STM_dn_mlp_seq_velocity_with_voca.pth.tar")

    # model hyperparameters
    parser.add_argument('--latent_channels', type=int, default=128)
    parser.add_argument('--in_channels', type=int, default=6)
    parser.add_argument('--out_channels', type=int, default=3)

    parser.add_argument('--n_points', type=int, default=3931)
    parser.add_argument('--n_faces', type=int, default=7800) #9453  7800  10000
    parser.add_argument('--k_eig', type=int, default=128)

    parser.add_argument('--batchnorm_encoder', type=str, default="GROUPNORM")
    parser.add_argument('--batchnorm_decoder', type=str, default="GROUPNORM")
    parser.add_argument('--shuffle_triangles', type=bool, default=False)

    args = parser.parse_args()

    infer_seq(args, seq="positions")

if __name__ == "__main__":
    main()
