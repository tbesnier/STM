import sys, os
import torch
import torch.nn as nn
import models.diffusion_net as diffusion_net
sys.path.append("./models/njf")
from models.njf.net import njf_decoder
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DiffusionNetAutoencoder(nn.Module):
    def __init__(self, args):
        super(DiffusionNetAutoencoder, self).__init__()
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.latent_channels = args.latent_channels
        self.device = args.device
        self.bs = args.batch_size
        self.n_faces = args.n_faces

        # encoder
        self.encoder = diffusion_net.layers.DiffusionNet(C_in=6,
                                                         C_out=self.latent_channels,
                                                         C_width=128,
                                                         N_block=4,
                                                         outputs_at='faces',
                                                         dropout=False,
                                                         normalization="None")
        # decoder
        self.decoder = njf_decoder(latent_features_shape=(self.bs, self.n_faces, self.latent_channels + 204), args=args)

        # self.last_layer = nn.Linear(128, 3)
        # self.layers = [nn.Linear(self.latent_channels + 204, 128),
        #                    nn.ReLU(),
        #                    nn.Linear(128, 128),
        #                    nn.ReLU(),
        #                    nn.Linear(128, 128),
        #                    nn.ReLU(),
        #                    self.last_layer]
        # self.mlp_dec = nn.Sequential(*self.layers)

        #self.encoder_lmk = PNEncoder(in_features=3, hidden_dim=128, out_dim=self.latent_channels)
        #self.encoder_lmk = SimpleBatchedGNN(hidden_dim1=128, hidden_dim2=32)

        #self.ca = LandmarkToMeshCrossAttention(dim_vertex=self.latent_channels,dim_node=32,out_dim=32,num_heads=4,dropout=0.0)

        #nn.init.constant_(self.last_layer.weight, 0)
        #nn.init.constant_(self.last_layer.bias, 0)

    def forward_latent_njf(self, template,
                mass_template, L_template, evals_template, evecs_template, gradX_template, gradY_template, faces_template, feats):

        z_template = self.encoder(template, mass=mass_template, L=L_template, evals=evals_template,
                                  evecs=evecs_template,
                                  gradX=gradX_template, gradY=gradY_template, faces=faces_template)

        # PC encoding
        #z_lmk = feats.squeeze(0)
        #z_lmk = self.encoder_lmk(z_lmk, self.edges)
        #feat_field = self.ca(z_template, z_lmk)
        #z = z_lmk.expand((z_template.shape[0], z_template.shape[1], z_lmk.shape[-1]))
        #feat_field = torch.cat((z_template, z), dim=-1)

        # Brute concatenation
        z_lmk = feats.reshape((-1)).unsqueeze(0)
        z = z_lmk.unsqueeze(1).expand((z_template.shape[0], z_template.shape[1], z_lmk.shape[-1]))
        feat_field = torch.cat((z_template, z), dim=-1)

        # MLP decoder
        #delta = self.mlp_dec(feat_field)

        # NJF decoder
        delta, pred_jac = self.decoder.predict_map(feat_field, source_verts=template, source_faces=faces_template,
                                        batch=False, target_vertices=None)
        delta, pred_jac = delta.to(self.device), pred_jac.to(self.device)

        pred = delta + template[:, :, :3]
        return pred