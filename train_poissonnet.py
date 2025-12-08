import sys, os, glob
import trimesh
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

from data_loader_poisson import get_dataloader
from models.deformer_poissonnet import PoissonNetAutoencoder


def train(args):
    model = PoissonNetAutoencoder(args).to(args.device)

    criterion = nn.MSELoss()
    criterion_val = nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    template_mesh = trimesh.load(args.template_file)

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
                    template = sample[0][1].to(args.device)
                    name = sample[0][2][0]
                    faces = sample[0][3].to(args.device)
                    mass_template = sample[0][4].to(args.device)
                    solver_template = [sample[1][0]]
                    G_template = sample[0][5].to(args.device)
                    M_template = sample[0][6].to(args.device)
                    faces_template = sample[0][7].to(args.device)
                    normals, normals_template = sample[0][8].to(args.device), sample[0][9].to(args.device)
                    feats = sample[0][10].to(args.device)
                    feats_template = sample[0][11].to(args.device)

                    in_features = torch.cat((vertices, normals), dim=2)
                    in_features_template = torch.cat((template, normals_template), dim=2)

                    vertices_pred = model.forward_latent_njf(
                        in_features_template, in_features,
                        mass_template, solver_template, G_template, M_template,
                        faces_template, feats, feats_template)

                    t_test_loss += criterion_val(vertices_pred, vertices).item()

                    os.makedirs(f'{args.results_path}/Meshes_Val/{str(epoch)}/preds', exist_ok=True)
                    os.makedirs(f'{args.results_path}/Meshes_Val/targets', exist_ok=True)
                    mesh_template = trimesh.Trimesh(template.cpu().detach().numpy()[0], faces_template[0].detach().cpu().numpy())
                    mesh_template.export(f'{args.results_path}/Meshes_Val/template_val.ply')

                    mesh = trimesh.Trimesh(vertices_pred[:,:,:3].cpu().detach().numpy()[0], faces_template[0].detach().cpu().numpy())
                    mesh.export(f'{args.results_path}/Meshes_Val/{str(epoch)}/preds/{name[:-4]}.ply')
                    mesh = trimesh.Trimesh(vertices[:,:,:3].cpu().detach().numpy()[0], faces[0].detach().cpu().numpy())
                    mesh.export(f'{args.results_path}/Meshes_Val/targets/{name[:-4]}.ply')

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
            template = sample[0][1].to(args.device)
            mass_template = sample[0][4].to(args.device)
            solver_template = [sample[1][0]]
            G_template = sample[0][5].to(args.device)
            M_template = sample[0][6].to(args.device)
            faces_template = sample[0][7].to(args.device)
            normals, normals_template = sample[0][8].to(args.device), sample[0][9].to(args.device)
            feats = sample[0][10].to(args.device)
            feats_template = sample[0][11].to(args.device)

            in_features = torch.cat((vertices, normals), dim=2)
            in_features_template = torch.cat((template, normals_template), dim=2)

            vertices_pred = model.forward_latent_njf(
                in_features_template, in_features,
                mass_template, solver_template, G_template, M_template,
                faces_template, feats, feats_template)

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
    template_mesh = trimesh.load(args.template_file)
    dataset = get_dataloader(args)
    model = PoissonNetAutoencoder(args).to(args.device)
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
            template = sample[0][1].to(args.device)
            faces = sample[0][3].to(args.device)
            mass_template = sample[0][4].to(args.device)
            solver_template = [sample[1][0]]
            G_template = sample[0][5].to(args.device)
            M_template = sample[0][6].to(args.device)
            faces_template = sample[0][7].to(args.device)
            normals, normals_template = sample[0][8].to(args.device), sample[0][9].to(args.device)
            feats = sample[0][10].to(args.device)
            feats_template = sample[0][11].to(args.device)

            in_features = torch.cat((vertices, normals), dim=2)
            in_features_template = torch.cat((template, normals_template), dim=2)

            vertices_pred = model.forward_latent_njf(
                in_features_template, in_features,
                mass_template, solver_template, G_template, M_template,
                faces_template, feats, feats_template)

            t_test_loss += metric(vertices_pred, vertices).item()
            pbar_talk.set_description(
                "TEST LOSS:{:.10f}".format((t_test_loss) / (b + 1)))

            os.makedirs(f'{args.results_path}/Meshes_test', exist_ok=True)

            for i, name in enumerate(sample[0][2]):
                mesh = trimesh.Trimesh(vertices_pred[i, :, :3].detach().cpu().numpy(), faces_template[0].detach().cpu().numpy())
                mesh.export(f'{args.results_path}/Meshes_test/' + str(name)[:-4] + '.ply')

            os.makedirs(f'{args.results_path}/Meshes_targets', exist_ok=True)
            for i, name in enumerate(sample[0][2]):
                mesh = trimesh.Trimesh(vertices[i, :, :3].detach().cpu().numpy(), faces[0].detach().cpu().numpy())
                mesh.export(f'{args.results_path}/Meshes_targets/' + str(name)[:-4] + '.ply')

def infer_rmsh(args):
    import models.poissonnet as poisson_net
    source_mesh = trimesh.load(args.infer_test)
    model = PoissonNetAutoencoder(args).to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])

    epochs = checkpoint['epoch'] + 1
    print(epochs)

    template_mesh = trimesh.load(f"/home/tbesnier/phd/projects/datasets/COMA_exp_sparse/FaceTalk_170725_00137_TA_neutral_no_eyes.ply", process=False)
    target_exp= trimesh.load(f"/home/tbesnier/phd/projects/datasets/COMA_exp_sparse/FaceTalk_170725_00137_TA_mouth_extreme.ply", process=False)
    ref_verts = np.array(template_mesh.vertices)
    lmk_idx = np.load("./data/lmk_noeyes_idx.npy")

    temp = np.array(source_mesh.vertices)
    normals_template = np.array(source_mesh.vertex_normals)
    faces_template = np.array(source_mesh.faces)
    feats_temp = np.hstack([temp, normals_template])
    vertices = np.array(target_exp.vertices)
    normals = np.array(target_exp.vertex_normals)
    faces = np.array(target_exp.faces)
    def_landmarks = vertices[lmk_idx]
    lmk_template = ref_verts[lmk_idx]
    target_feats = torch.FloatTensor(def_landmarks - lmk_template)

    model.eval()
    with torch.no_grad():
        vertices = torch.FloatTensor(vertices).unsqueeze(0).to(args.device)
        template = torch.FloatTensor(source_mesh.vertices).unsqueeze(0).to(args.device)
        faces_template = torch.tensor(faces_template).unsqueeze(0).to(dtype=torch.int64, device=args.device)
        normals_template, normals = torch.FloatTensor(source_mesh.vertex_normals).unsqueeze(0).to(args.device), torch.FloatTensor(normals).unsqueeze(0).to(args.device)

        mass_src, solver_src, G_src, M_src = poisson_net.geometry.construct_mesh_operators(
            torch.FloatTensor(temp).to(device="cuda:0"),
            faces_template[0], high_precision=True)

        mass_template = mass_src.float().to(args.device)
        solver_template = solver_src
        G_template = G_src.float().to(args.device)
        M_template = M_src.float().to(args.device)

        feats = target_feats.unsqueeze(0).to(args.device)
        feats_template = torch.FloatTensor(feats_temp).unsqueeze(0).to(args.device)

        in_features = torch.cat((vertices, normals), dim=2)
        in_features_template = torch.cat((template, normals_template), dim=2)

        vertices_pred = model.forward_latent_njf(
            in_features_template, in_features,
            mass_template, solver_template, G_template, M_template,
            faces_template, feats, feats_template)

        os.makedirs(f'{args.results_path}/Meshes_infer_rmsh', exist_ok=True)

        mesh = trimesh.Trimesh(vertices_pred[0, :, :3].detach().cpu().numpy(), faces_template[0].detach().cpu().numpy())
        mesh.export(f'{args.results_path}/Meshes_infer_rmsh/infer_pred.ply')

        os.makedirs(f'{args.results_path}/Meshes_infer_rmsh', exist_ok=True)
        mesh = trimesh.Trimesh(vertices[0, :, :3].detach().cpu().numpy(), faces)
        mesh.export(f'{args.results_path}/Meshes_infer_rmsh/infer_target.ply')

def main():
    parser = argparse.ArgumentParser(description='D2D: Dense to Dense Encoder-Decoder')

    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epochs', type=float, default=300)
    parser.add_argument('--batch_size', type=float, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")

    # data args
    parser.add_argument('--template_file', type=str,
                        default='./data/template.obj')
    parser.add_argument('--meshes_path', type=str, default='../datasets/COMA_exp_sparse')
    parser.add_argument('--infer_test', type=str, default='../datasets/COMA_exp_sparse_rmsh/arnold_aligned_ds.ply')

    parser.add_argument('--train_subjects', type=str, default="FaceTalk_170725_00137_TA FaceTalk_170728_03272_TA FaceTalk_170731_00024_TA"
                                                              " FaceTalk_170809_00138_TA FaceTalk_170811_03274_TA FaceTalk_170811_03275_TA"
                                                              " FaceTalk_170904_00128_TA FaceTalk_170904_03276_TA FaceTalk_170908_03277_TA"
                                                              " FaceTalk_170912_03278_TA FaceTalk_170913_03279_TA FaceTalk_170915_00223_TA")
    parser.add_argument('--val_subjects', type=str, default="FaceTalk_170725_00137_TA")
    parser.add_argument('--test_subjects', type=str, default="FaceTalk_170725_00137_TA FaceTalk_170728_03272_TA FaceTalk_170731_00024_TA"
                                                              " FaceTalk_170809_00138_TA FaceTalk_170811_03274_TA FaceTalk_170811_03275_TA"
                                                              " FaceTalk_170904_00128_TA FaceTalk_170904_03276_TA FaceTalk_170908_03277_TA"
                                                              " FaceTalk_170912_03278_TA FaceTalk_170913_03279_TA FaceTalk_170915_00223_TA")
    parser.add_argument('--results_path', type=str, default="../Data/STM/test")

    # checkpoint args
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--models_dir", type=str, default="../Data/STM/Models")
    parser.add_argument("--model_path", type=str, default="../Data/STM/Models/STM_test.pth.tar")

    # model hyperparameters
    parser.add_argument('--latent_channels', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=6)
    parser.add_argument('--out_channels', type=int, default=3)

    parser.add_argument('--n_points', type=int, default=3931)
    parser.add_argument('--n_faces', type=int, default=10000) #9453  7800  10000

    parser.add_argument('--batchnorm_encoder', type=str, default="GROUPNORM")
    parser.add_argument('--batchnorm_decoder', type=str, default="GROUPNORM")
    parser.add_argument('--shuffle_triangles', type=bool, default=False)

    args = parser.parse_args()

    #train(args)
    #test(args)
    infer_rmsh(args)

if __name__ == "__main__":
    main()
