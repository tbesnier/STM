import sys, os, glob
import trimesh
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

from dataloader_seq_diffusion import get_dataloader
from models.deformer_seq_diffusionnet import DiffusionNetAutoencoder

def train(args):
    model = DiffusionNetAutoencoder(args).to(args.device)

    criterion = nn.MSELoss()
    criterion_val = nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    starting_epoch = 0
    if args.load_model:
        checkpoint = torch.load(args.model_path, map_location=args.device)  # args.model_path
        model.load_state_dict(checkpoint['autoencoder_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        print(starting_epoch)
    dataset = get_dataloader(args)

    train_losses = []
    val_losses = []
    for epoch in range(starting_epoch, args.epochs):
        valid_loss_log = []
        if epoch%20==0 and epoch>0:
            model.eval()
            with torch.no_grad():
                t_test_loss = 0
                pbar_talk = tqdm(enumerate(dataset["valid"]), total=len(dataset["valid"]))
                for b, sample in pbar_talk:
                    vertices = sample[0][0].to(args.device)
                    template = sample[1].to(args.device)
                    name = sample[2][0]
                    faces = sample[3].to(args.device)
                    mass_template = sample[4].to(args.device)
                    L_template = sample[5][0].to(args.device).unsqueeze(0)
                    evals_template = sample[6].to(args.device)
                    evecs_template = sample[7].to(args.device)
                    gradX_template = sample[8][0].to(args.device).unsqueeze(0)
                    gradY_template = sample[9][0].to(args.device).unsqueeze(0)
                    faces_template = sample[10].to(args.device)
                    normals, normals_template = sample[11].to(args.device), sample[12].to(args.device)
                    feats = sample[13].to(args.device)
                    in_features_template = torch.cat((template, normals_template), dim=2)

                    vertices_pred = model.forward_latent_njf(
                        in_features_template,
                        mass_template, L_template, evals_template, evecs_template, gradX_template, gradY_template,
                        faces_template, feats)
                    t_test_loss += criterion_val(vertices_pred, vertices).item()

                    os.makedirs(f'{args.results_path}/Meshes_Val/{str(epoch)}/preds/{name}', exist_ok=True)
                    os.makedirs(f'{args.results_path}/Meshes_Val/targets/{name}', exist_ok=True)

                    for i in range(vertices_pred.shape[0]):
                        mesh = trimesh.Trimesh(vertices_pred[i,:,:3].cpu().detach().numpy(), faces_template[0].detach().cpu().numpy())
                        mesh.export(f'{args.results_path}/Meshes_Val/{str(epoch)}/preds/{name}/{str(i).zfill(3)}.ply')
                        mesh = trimesh.Trimesh(vertices[i,:,:3].cpu().detach().numpy(), faces[0][0].detach().cpu().numpy())
                        mesh.export(f'{args.results_path}/Meshes_Val/targets/{name}/{str(i).zfill(3)}.ply')

                    pbar_talk.set_description(
                        "(Epoch {}) VAL LOSS:{:.10f}".format((epoch + 1), (t_test_loss) / (b + 1)))
                    valid_loss_log.append(np.mean(t_test_loss))
                current_loss = np.mean(valid_loss_log)
                val_losses.append(current_loss)

        loss_log = []
        model.train()
        tloss = 0

        pbar_talk = tqdm(enumerate(dataset["train"]), total=len(dataset["train"]))
        for b, sample in pbar_talk:
            vertices = sample[0][0].to(args.device)
            template = sample[1].to(args.device)
            name = sample[2][0]
            faces = sample[3].to(args.device)
            mass_template = sample[4].to(args.device)
            L_template = sample[5][0].to(args.device).unsqueeze(0)
            evals_template = sample[6].to(args.device)
            evecs_template = sample[7].to(args.device)
            gradX_template = sample[8][0].to(args.device).unsqueeze(0)
            gradY_template = sample[9][0].to(args.device).unsqueeze(0)
            faces_template = sample[10].to(args.device)
            normals, normals_template = sample[11].to(args.device), sample[12].to(args.device)
            feats = sample[13].to(args.device)

            in_features_template = torch.cat((template, normals_template), dim=2)

            vertices_pred = model.forward_latent_njf(
                in_features_template,
                mass_template, L_template, evals_template, evecs_template, gradX_template, gradY_template,
                faces_template, feats)

            optim.zero_grad()

            loss = criterion(vertices_pred, vertices)
            loss.backward()
            optim.step()
            tloss += loss.item()
            loss_log.append(loss.item())
            pbar_talk.set_description(
                "(Epoch {}) TRAIN LOSS:{:.10f}".format((epoch + 1), tloss / (b + 1)))
        train_losses.append(np.mean(loss_log))

        torch.save({'epoch': epoch,
                    'autoencoder_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    }, args.model_path)


def test(args):
    dataset = get_dataloader(args)
    model = DiffusionNetAutoencoder(args).to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])
    metric = nn.MSELoss()

    epochs = checkpoint['epoch'] + 1
    print(epochs)

    model.eval()
    with torch.no_grad():
        t_test_loss = 0
        pbar_talk = tqdm(enumerate(dataset["test"]), total=len(dataset["test"]))
        for b, sample in pbar_talk:
            vertices = sample[0][0].to(args.device)
            template = sample[1].to(args.device)
            name = sample[2][0]
            faces = sample[3].to(args.device)
            mass_template = sample[4].to(args.device)
            L_template = sample[5][0].to(args.device).unsqueeze(0)
            evals_template = sample[6].to(args.device)
            evecs_template = sample[7].to(args.device)
            gradX_template = sample[8][0].to(args.device).unsqueeze(0)
            gradY_template = sample[9][0].to(args.device).unsqueeze(0)
            faces_template = sample[10].to(args.device)
            normals, normals_template = sample[11].to(args.device), sample[12].to(args.device)
            feats = sample[13].to(args.device)

            in_features_template = torch.cat((template, normals_template), dim=2)

            vertices_pred = model.forward_latent_njf(
                in_features_template,
                mass_template, L_template, evals_template, evecs_template, gradX_template, gradY_template,
                faces_template, feats)

            t_test_loss += metric(vertices_pred, vertices).item()
            pbar_talk.set_description(
                "TEST LOSS:{:.10f}".format((t_test_loss) / (b + 1)))

            os.makedirs(f'{args.results_path}/Meshes_test/preds/{name}', exist_ok=True)
            os.makedirs(f'{args.results_path}/Meshes_test/targets/{name}', exist_ok=True)
            for i in range(vertices_pred.shape[0]):
                mesh = trimesh.Trimesh(vertices_pred[i, :, :3].cpu().detach().numpy(),
                                       faces_template[0].detach().cpu().numpy())
                mesh.export(f'{args.results_path}/Meshes_test/preds/{name}/{str(i).zfill(3)}.ply')
                mesh = trimesh.Trimesh(vertices[i, :, :3].cpu().detach().numpy(), faces[0][0].detach().cpu().numpy())
                mesh.export(f'{args.results_path}/Meshes_test/targets/{name}/{str(i).zfill(3)}.ply')



def infer_seq(args, seq="positions"):
    import models.diffusion_net as diffusion_net
    source_mesh = trimesh.load(args.infer_test)
    temp = np.array(source_mesh.vertices)
    faces_template = np.array(source_mesh.faces)
    model = DiffusionNetAutoencoder(args).to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])
    target_seq = np.load(args.infer_seq)
    if target_seq.shape[-1] != 3:
        target_seq = target_seq.reshape(target_seq.shape[0], 68, 3)
    print(f"target_seq: {target_seq.shape}")
    if seq == "positions":
        lmk_0 = target_seq[0]
    os.makedirs(f'{args.results_path}/Meshes_infer_seq', exist_ok=True)

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
    model.eval()
    with torch.no_grad():
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

        os.makedirs(f'{args.results_path}/infer/', exist_ok=True)

        for i in range(vertices_pred.shape[0]):
            mesh = trimesh.Trimesh(vertices_pred[i, :, :3].cpu().detach().numpy(),
                                   faces_template[0].detach().cpu().numpy())
            mesh.export(f'{args.results_path}/infer/{str(i).zfill(3)}.ply')



def main():
    parser = argparse.ArgumentParser(description='D2D: Dense to Dense Encoder-Decoder')

    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epochs', type=float, default=300)
    parser.add_argument('--batch_size', type=float, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")

    # data args
    parser.add_argument('--template_file', type=str, default='./data/template.obj')

    parser.add_argument('--templates_dir_COMA', type=str, default='D:/phd_data/COMA_templates')
    parser.add_argument('--deformations_dir_COMA', type=str, default='D:/phd_data/COMA_interp')
    parser.add_argument('--templates_dir_ICT', type=str, default='D:/phd_data/ICT_templates')
    parser.add_argument('--deformations_dir_ICT', type=str, default='D:/phd_data/ICT_interp')

    parser.add_argument('--infer_test', type=str, default="D:/phd_data/demo_mesh_arnold.ply")#"D:/phd_data/ICT_interp_full/id_049_neutral/frame_0000.ply")#"../datasets/COMA_exp_sparse/FaceTalk_170725_00137_TA_neutral_no_eyes.ply")#"../datasets/test_ICT.ply")
    parser.add_argument('--infer_seq', type=str, default="./data/ravdess/ex_happy_disp.npy")#"./data/ravdess/ex_happy_disp.npy")#"D:/phd_data/ravdess/lmks_npy_aligned/Actor_01/01-02-03-02-02-01-01_aligned.npy")#"./data/ravdess/ex_happy_disp.npy")#"D:/phd_data/ravdess/lmks_npy_aligned/Actor_01/01-02-05-02-02-02-01_aligned.npy")#"../datasets/MEAD/landmarks/W009_fear_3_028.npy")#ravdess/tracking_npy_aligned/Actor_01/01-02-03-01-01-01-01_aligned.npy")#"./data/ex_vocaset_lmk.npy")

    parser.add_argument('--train_subjects', type=str, default="FaceTalk_170725_00137_TA FaceTalk_170728_03272_TA FaceTalk_170731_00024_TA"
                                                              " FaceTalk_170809_00138_TA FaceTalk_170811_03274_TA FaceTalk_170811_03275_TA"
                                                              " FaceTalk_170904_00128_TA FaceTalk_170904_03276_TA FaceTalk_170908_03277_TA"
                                                              " FaceTalk_170912_03278_TA FaceTalk_170913_03279_TA FaceTalk_170915_00223_TA"
                                                              "id_000 id_001 id_002 id_003 id_004 id_005 id_006 id_007 id_008 id_009 id_010"
                                                              "id_011 id_012 id_013 id_014 id_015 id_016 id_017 id_018 id_019 id_020 id_021"
                                                              "id_022 id_023 id_024 id_025 id_026 id_027 id_028 id_029 id_030 id_031 id_032"
                                                              "id_033 id_034 id_035 id_036 id_037 id_038 id_039 id_040 id_041 id_042"
                                                              "id_041 id_042 id_043 id_044 id_045 id_046 id_047 id_048 id_049")
    parser.add_argument('--val_subjects', type=str, default="FaceTalk_170725_00137_TA id_000")
    parser.add_argument('--test_subjects', type=str, default="FaceTalk_170725_00137_TA FaceTalk_170728_03272_TA FaceTalk_170731_00024_TA"
                                                              " FaceTalk_170809_00138_TA FaceTalk_170811_03274_TA FaceTalk_170811_03275_TA"
                                                              " FaceTalk_170904_00128_TA FaceTalk_170904_03276_TA FaceTalk_170908_03277_TA"
                                                              " FaceTalk_170912_03278_TA FaceTalk_170913_03279_TA FaceTalk_170915_00223_TA id_048 id_049")
    parser.add_argument('--results_path', type=str, default="../Data/STM/test_dn_mlp_seq")

    # checkpoint args
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--models_dir", type=str, default="../Data/STM/Models")
    parser.add_argument("--model_path", type=str, default="../Data/STM/Models/STM_dn_mlp_seq.pth.tar")

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

    #train(args)
    #test(args)
    #infer_rmsh(args)
    infer_seq(args, seq="disp")

if __name__ == "__main__":
    main()
