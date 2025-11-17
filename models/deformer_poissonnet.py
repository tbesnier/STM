import sys, os
import torch
import torch.nn as nn
import models.poissonnet as poisson_net
import math
sys.path.append("./models/njf")
from models.njf.net import njf_decoder


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
        self.to_out = nn.Linear(heads * self.d_v, 2 * 32, bias=False)
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


class CrossAttention7800x68(nn.Module):
    def __init__(self, d_vertices=32, d_landmarks=3, d_out=64, n_heads=8):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        # Project to common attention dimension d_out
        self.q_proj = nn.Linear(d_vertices, d_out)
        self.k_proj = nn.Linear(d_landmarks, d_out)
        self.v_proj = nn.Linear(d_landmarks, d_out)

        # Multi-head attention: queries are vertices, keys/values are landmarks
        self.attn = nn.MultiheadAttention(
            embed_dim=d_out,
            num_heads=n_heads,
            batch_first=True,  # so we use [B, L, C] tensors
        )

    def forward(self, vertices, landmarks):
        """
        vertices:  [B, 7800, 32]
        landmarks: [B,   68,  3]
        returns:
            out:       [B, 7800, 64]
            attn_w:    [B, 7800, 68]  (attention weights, optional)
        """

        # Project to common dimension
        Q = self.q_proj(vertices)   # [B, 7800, 64]
        K = self.k_proj(landmarks)  # [B,   68, 64]
        V = self.v_proj(landmarks)  # [B,   68, 64]

        # Cross-attention: each vertex attends over all landmarks
        out, attn_w = self.attn(Q, K, V)  # out: [B, 7800, 64]

        return out, attn_w


class LandmarkStatsFusion(nn.Module):
    def __init__(self, d_vert=32, d_lm=3, d_out=64):
        super().__init__()
        # project landmark pos to a small feature space first
        self.lm_proj = nn.Linear(d_lm, 16)

        self.fuse = nn.Sequential(
            nn.Linear(d_vert + 2 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, d_out),
        )

    def forward(self, vertices, landmarks):
        # vertices:  [B, 7800, 32]
        # landmarks: [B,   68,  3]
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)
        if landmarks.dim() == 2:
            landmarks = landmarks.unsqueeze(0)

        B, V, _ = vertices.shape

        lm_feat = self.lm_proj(landmarks)          # [B, 68, 16]
        lm_mean = lm_feat.mean(dim=1)             # [B, 16]
        lm_max  = lm_feat.max(dim=1).values       # [B, 16]

        stats = torch.cat([lm_mean, lm_max], dim=-1)  # [B, 32]
        stats = stats.unsqueeze(1).expand(-1, V, -1)  # [B, 7800, 32]

        x = torch.cat([vertices, stats], dim=-1)      # [B, 7800, 64]
        out = self.fuse(x)                            # [B, 7800, 64]
        return out


class PoissonNetAutoencoder(nn.Module):
    def __init__(self, args):
        super(PoissonNetAutoencoder, self).__init__()
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.latent_channels = args.latent_channels
        self.device = args.device
        self.bs = args.batch_size
        self.n_faces = args.n_faces

        # encoder
        self.encoder = poisson_net.PoissonNet(C_in=6,
                                            C_out=self.latent_channels // 2,
                                            C_width=128,
                                            n_blocks=4,
                                            head='linear',
                                            extra_features=0,
                                            outputs_at="vertices")

        self.cma = LandmarkStatsFusion(d_vert=32, d_lm=3, d_out=64)
        # decoder
        self.decoder = njf_decoder(latent_features_shape=(self.bs, self.n_faces, self.latent_channels), args=args)

        self.layers = [nn.Linear(self.latent_channels, 128),
                           nn.ReLU(),
                           nn.Linear(128, 128),
                           nn.ReLU(),
                           nn.Linear(128, 128),
                           nn.ReLU(),
                           nn.Linear(128, 3)]
        self.mlp_dec = nn.Sequential(*self.layers)

        print("encoder parameters: ", count_parameters(self.encoder))
        print("decoder parameters: ", count_parameters(self.decoder))

    def forward_latent_njf(self, template, vertices,
                mass_template, solver_template, G_template, M_template, faces_template, feats, feats_temp):

        z_template = self.encoder(template, M=M_template, G=G_template, solver=solver_template, vertex_mass=mass_template,
                                  faces=faces_template)
        z_lmk = feats#self.mlp(feats.squeeze(0))

        z_attn = self.cma(z_template.squeeze(0), z_lmk.squeeze(0))

        feat_field = z_attn

        # delta, pred_jac = self.decoder.predict_map(feat_field, source_verts=template, source_faces=faces_template,
        #                                 batch=False, target_vertices=None)
        # delta, pred_jac = delta.to(self.device), pred_jac.to(self.device)

        delta = self.mlp_dec(feat_field)

        pred = delta + template[:, :, :3]
        return pred

