import sys, os
import torch
import torch.nn as nn
import models.diffusion_net as diffusion_net
sys.path.append("./models/njf")
from models.njf.net import njf_decoder
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CrossMeshAttention(nn.Module):
    def __init__(self, f: int, heads: int = 4, d_k: int = None, d_v: int = None):
        """
        Cross-mesh attention mapping target features onto source vertices.

        Args:
            f     : Input feature dimensionality.
            heads : Number of attention heads.
            d_k   : Dimensionality of queries/keys per head (default: f // heads).
            d_v   : Dimensionality of values per head (default: f // heads).
        """
        super().__init__()
        self.f = f
        self.heads = heads
        self.d_k = d_k or (f // heads)
        self.d_v = d_v or (f // heads)

        # Linear maps for Q, K, V
        self.to_q = nn.Linear(f, heads * self.d_k, bias=False)
        self.to_k = nn.Linear(f, heads * self.d_k, bias=False)
        self.to_v = nn.Linear(f, heads * self.d_v, bias=False)

        # Final projection to 2f
        self.to_out = nn.Linear(heads * self.d_v, 2 * f, bias=False)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, F_src: torch.Tensor, F_tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F_src : Tensor of shape (N, f) — source vertex features.
            F_tgt : Tensor of shape (M, f) — target vertex features.

        Returns:
            Tensor of shape (N, 2f) — mapped source features.
        """
        N = F_src.shape[0]
        M = F_tgt.shape[0]

        # Compute Q, K, V
        Q = self.to_q(F_src).view(N, self.heads, self.d_k)  # (N, H, d_k)
        K = self.to_k(F_tgt).view(M, self.heads, self.d_k)  # (M, H, d_k)
        V = self.to_v(F_tgt).view(M, self.heads, self.d_v)  # (M, H, d_v)

        # Compute attention scores
        # (N, H, d_k) @ (H, d_k, M) -> (N, H, M)
        scores = torch.einsum("nhk,mhk->nhm", Q, K) * self.scale
        attn = scores.softmax(dim=-1)  # (N, H, M)

        # Aggregate values
        out = torch.einsum("nhm,mhv->nhv", attn, V)  # (N, H, d_v)
        out = out.contiguous().view(N, -1)  # (N, H*d_v)

        # Final linear projection
        return self.to_out(out)


class DiffusionNetAutoencoder(nn.Module):
    def __init__(self, args):
        super(DiffusionNetAutoencoder, self).__init__()
        if args.use_hks:
            self.in_channels = 16
        else:
            self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.latent_channels = args.latent_channels
        self.device = args.device
        self.bs = args.batch_size
        self.n_faces = args.n_faces
        self.dataset = args.dataset

        # encoder
        self.encoder = diffusion_net.layers.DiffusionNet(C_in=6,
                                                         C_out=self.latent_channels // 2,
                                                         C_width=128,  # self.latent_channels*2,
                                                         N_block=4,
                                                         outputs_at='faces',
                                                         dropout=False,
                                                         normalization="None")
        self.encoder_def = diffusion_net.layers.DiffusionNet(C_in=self.in_channels,
                                                             C_out=self.latent_channels // 2,
                                                             C_width=128,  # self.latent_channels*2,
                                                             N_block=4,
                                                             outputs_at='global_mean',
                                                             dropout=False,
                                                             normalization="None")

        # decoder
        self.decoder = njf_decoder(latent_features_shape=(self.bs, self.n_faces, self.latent_channels), args=args)

        print("encoder parameters: ", count_parameters(self.encoder))
        print("decoder parameters: ", count_parameters(self.decoder))

    def forward_latent_njf(self, template, vertices, mass, L, evals, evecs, gradX, gradY, faces,
                mass_template, L_template, evals_template, evecs_template, gradX_template, gradY_template, faces_template,
                           feats, feats_temp):

        z_template = self.encoder(template, mass=mass_template, L=L_template, evals=evals_template, evecs=evecs_template,
                                  gradX=gradX_template, gradY=gradY_template, faces=faces_template)
        z = self.encoder_def(feats, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        z_ref = self.encoder_def(feats_temp, mass=mass_template, L=L_template, evals=evals_template,
                                 evecs=evecs_template, gradX=gradX_template, gradY=gradY_template, faces=faces_template)
        z = z - z_ref
        z = z.unsqueeze(1).expand((z_template.shape[0], z_template.shape[1], z.shape[-1]))

        cat_latent = torch.cat((z_template, z), dim=-1)

        delta, pred_jac, pred_jac_restricted = self.decoder.predict_map(cat_latent, source_verts=template, source_faces=faces_template,
                                        batch=False, target_vertices=vertices)
        delta, pred_jac, pred_jac_restricted = delta.to(self.device), pred_jac.to(self.device), pred_jac_restricted.to(self.device)
        pred = delta + template[:, :, :3]
        return pred, cat_latent, pred_jac, pred_jac_restricted
