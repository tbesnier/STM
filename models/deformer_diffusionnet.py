from __future__ import annotations
import sys, os
import torch
import torch.nn as nn
import models.diffusion_net as diffusion_net
sys.path.append("./models/njf")
from models.njf.net import njf_decoder


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    edge_index: (2, E)
    returns: (2, E + num_nodes) with i->i self loops added (no dedup).
    """
    device = edge_index.device
    loops = torch.arange(num_nodes, device=device, dtype=edge_index.dtype)
    loops = torch.stack([loops, loops], dim=0)  # (2, N)
    return torch.cat([edge_index, loops], dim=1)


class GraphConv(nn.Module):
    """
    Simple GCN-style layer (mean aggregation) without external deps (no PyG).
    Message passing: src -> dst along edge_index.

    x: (B, N, Cin)
    edge_index: (2, E)
    out: (B, N, Cout)
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, N, Cin = x.shape
        assert edge_index.dim() == 2 and edge_index.shape[0] == 2, "edge_index must be (2, E)."

        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)
        E = src.numel()

        # Aggregate neighbor features into dst nodes
        msgs = x[:, src, :]  # (B, E, Cin)

        agg = x.new_zeros(B, N, Cin)
        idx = dst.view(1, E, 1).expand(B, E, Cin)  # (B, E, Cin)
        agg.scatter_add_(dim=1, index=idx, src=msgs)  # sum over incoming edges

        # Degree for mean aggregation
        deg = x.new_zeros(N)
        deg.scatter_add_(0, dst, torch.ones(E, device=x.device, dtype=x.dtype))
        deg = deg.clamp_min(1.0).view(1, N, 1)  # avoid div0

        neigh_mean = agg / deg  # (B, N, Cin)

        out = self.lin_self(x) + self.lin_neigh(neigh_mean)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        return out


class LandmarkGCNFeatureExtractor(nn.Module):
    """
    Extract per-landmark features from per-node displacement vectors (or any node signal).
      input:  d  (B, N, in_dim)   e.g. in_dim=3 for displacement vectors
      output: h  (B, N, out_dim)
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 64,
        out_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        add_loops: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1
        self.add_loops = add_loops

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers = []
        for i in range(num_layers):
            layers.append(GraphConv(dims[i], dims[i + 1], dropout=dropout if i < num_layers - 1 else 0.0))
        self.layers = nn.ModuleList(layers)

    def forward(self, d: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, N, _ = d.shape
        if self.add_loops:
            edge_index = add_self_loops(edge_index, num_nodes=N)

        h = d
        for layer in self.layers:
            h = layer(h, edge_index)
        return h  # (B, N, out_dim)

class LandmarksToMeshCrossAttention(nn.Module):
    """
    Cross-attend mesh vertex tokens (queries) to landmark tokens (keys/values),
    where landmark tokens are built from:
      - landmark features (B, N, lm_feat_dim)
      - learnable positional encoding via landmark index (0..N-1)

    Forward:
      mesh_feat: (B, V, f)
      lm_feat:   (B, N, lm_feat_dim)
    Output:
      (B, V, f + f')
    """

    def __init__(
        self,
        mesh_feat_dim: int,      # f
        lm_feat_dim: int,        # landmark feature dim (from GCN)
        out_landmark_dim: int,   # f'
        *,
        d_model: int = 128,
        n_heads: int = 8,
        pe_dim: int = 32,
        max_landmarks: int = 256,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."

        self.max_landmarks = max_landmarks
        self.use_residual = use_residual

        # Learnable landmark index embedding (positional encoding by landmark index)
        self.lm_index_emb = nn.Embedding(max_landmarks, pe_dim)

        # Build landmark tokens from (lm_feat + index-embed) -> d_model
        self.lm_token = nn.Sequential(
            nn.Linear(lm_feat_dim + pe_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Mesh -> queries
        self.mesh_in_norm = nn.LayerNorm(mesh_feat_dim)
        self.q_proj = nn.Linear(mesh_feat_dim, d_model)

        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Attention output -> f'
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_landmark_dim),
        )

    def forward(self, mesh_feat: torch.Tensor, lm_feat: torch.Tensor) -> torch.Tensor:
        """
        mesh_feat: (B, V, f)
        lm_feat:   (B, N, lm_feat_dim)
        returns:   (B, V, f + f')
        """
        B, V, f = mesh_feat.shape
        B2, N, _ = lm_feat.shape
        assert B == B2, "Batch size mismatch between mesh_feat and lm_feat."
        if N > self.max_landmarks:
            raise ValueError(f"N={N} exceeds max_landmarks={self.max_landmarks}.")

        # Landmark index positional encoding
        idx = torch.arange(N, device=lm_feat.device, dtype=torch.long).unsqueeze(0).expand(B, N)  # (B, N)
        idx_emb = self.lm_index_emb(idx)  # (B, N, pe_dim)

        lm_tok = torch.cat([lm_feat, idx_emb], dim=-1)  # (B, N, lm_feat_dim + pe_dim)
        kv = self.lm_token(lm_tok)  # (B, N, d_model)
        kv = self.kv_norm(kv)

        mesh_feat_normed = self.mesh_in_norm(mesh_feat)
        q = self.q_proj(mesh_feat_normed)  # (B, V, d_model)
        q = self.q_norm(q)

        attn_out, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)  # (B, V, d_model)

        if self.use_residual:
            attn_out = attn_out + q

        lm_to_mesh = self.out_proj(attn_out)  # (B, V, f')

        return torch.cat([mesh_feat, lm_to_mesh], dim=-1)  # (B, V, f + f')

class PNEncoder(nn.Module):
    """
    Encodes a single mesh frame (a point cloud) into a fixed-size descriptor.
    This is inspired by PointNet: each vertex is processed with a shared MLP,
    and a symmetric pooling function (max pooling) is used to obtain a global feature.
    """

    def __init__(self, in_features=3, hidden_dim=128, out_dim=64):
        """
        Args:
            in_features (int): Number of features per vertex (typically 3 for (x,y,z)).
            hidden_dim (int): Hidden dimension for the MLP.
            out_dim (int): Output dimension of the frame descriptor.
        """
        super(PNEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, vertices):
        """
        Args:
            vertices (torch.Tensor): Tensor of shape (B, N, 3) or (N, 3) where B is batch size and N is number of vertices.
        Returns:
            torch.Tensor: Encoded frame feature of shape (B, out_dim) (or (out_dim,) for a single frame).
        """
        x = self.mlp(vertices)

        x, _ = torch.max(x, dim=1)  # shape: (B, out_dim)
        return self.linear(x)

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
                                                         outputs_at='vertices',
                                                         dropout=False,
                                                         normalization="None")
        # decoder
        #self.decoder = njf_decoder(latent_features_shape=(self.bs, self.n_faces, 2*self.latent_channels + 204), args=args)

        self.last_layer = nn.Linear(128, 3)
        self.layers = [nn.Linear(self.latent_channels*2 + 204, 128),
                           nn.ReLU(),
                           nn.Linear(128, 128),
                           nn.ReLU(),
                           nn.Linear(128, 128),
                           nn.ReLU(),
                           self.last_layer]
        self.mlp_dec = nn.Sequential(*self.layers)

        self.edges = torch.tensor([
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
            [12, 13],
            [13, 14], [14, 15], [15, 16],
            [17, 18], [18, 19], [19, 20], [20, 21], [22, 23], [23, 24], [24, 25], [25, 26],
            [27, 28], [28, 29], [29, 30], [31, 32], [32, 33], [33, 34], [34, 35],
            [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],
            [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],
            [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58],
            [58, 59],
            [59, 60], [60, 48],
            [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60]
        ], dtype=torch.long, device="cuda:0").swapaxes(0, 1)

        self.ca = LandmarksToMeshCrossAttention(mesh_feat_dim=self.latent_channels,
                                                lm_feat_dim=self.latent_channels // 2,
                                                out_landmark_dim=self.latent_channels,
                                                d_model=128,
                                                n_heads=8,
                                                dropout=0.0)
        self.gcn = LandmarkGCNFeatureExtractor(in_dim=3, hidden_dim=128, out_dim=self.latent_channels // 2,
                                               num_layers=3, dropout=0.0)

        nn.init.constant_(self.last_layer.weight, 0)
        nn.init.constant_(self.last_layer.bias, 0)

    def forward_latent_njf(self, template,
                mass_template, L_template, evals_template, evecs_template, gradX_template, gradY_template, faces_template, feats):

        z_template = self.encoder(template, mass=mass_template, L=L_template, evals=evals_template,
                                  evecs=evecs_template,
                                  gradX=gradX_template, gradY=gradY_template, faces=faces_template)

        # CA of features
        z_lmk = feats
        lm_feat = self.gcn(z_lmk, self.edges)
        feat_field_ca = self.ca(z_template, lm_feat)

        # Brute concatenation
        z_lmk = feats.reshape((-1)).unsqueeze(0)
        z = z_lmk.unsqueeze(1).expand((z_template.shape[0], z_template.shape[1], z_lmk.shape[-1]))
        feat_field = torch.cat((feat_field_ca, z), dim=-1)

        # MLP decoder
        delta = self.mlp_dec(feat_field)

        # NJF decoder
        #delta, pred_jac = self.decoder.predict_map(feat_field, source_verts=template, source_faces=faces_template,
        #                                batch=False, target_vertices=None)
        #delta, pred_jac = delta.to(self.device), pred_jac.to(self.device)

        pred = delta + template[:, :, :3]
        return pred