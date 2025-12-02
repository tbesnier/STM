import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.poissonnet as poisson_net
import math
sys.path.append("./models/njf")
from models.njf.net import njf_decoder


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class BatchedGNNLayer(nn.Module):
    """
    One message-passing layer for batched graphs with shared topology.

    x_bi' = W_self * x_bi + W_neigh * sum_{j in N(i)} x_bj

    Inputs:
        x: (B, N, in_dim)
        edge_index: (E, 2) long, edges [src, dst] (shared across batch)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        """
        x: (B, N, in_dim)
        edge_index: (E, 2), long
        """
        if x.dim() == 2:
            # fallback to single graph: (N, F) -> (1, N, F)
            x = x.unsqueeze(0)

        B, N, in_dim = x.shape
        edge_index = edge_index.long()
        src = edge_index[:, 0]   # (E,)
        dst = edge_index[:, 1]   # (E,)

        # ---- Aggregate neighbor messages (sum) ----
        # Flatten batch and node dims: (B, N, F) -> (B*N, F)
        x_flat = x.reshape(B * N, in_dim)
        agg_flat = torch.zeros_like(x_flat)

        # For each batch, offset node indices by b * N
        batch_offsets = torch.arange(B, device=x.device).unsqueeze(1) * N  # (B, 1)
        src_flat = (src.unsqueeze(0) + batch_offsets).reshape(-1)         # (B*E,)
        dst_flat = (dst.unsqueeze(0) + batch_offsets).reshape(-1)         # (B*E,)

        # agg_flat[dst_flat] += x_flat[src_flat]
        agg_flat.index_add_(0, dst_flat, x_flat[src_flat])

        # Back to (B, N, in_dim)
        agg = agg_flat.view(B, N, in_dim)

        # ---- Linear transforms ----
        h_self = self.lin_self(x)     # (B, N, out_dim)
        h_neigh = self.lin_neigh(agg) # (B, N, out_dim)

        out = F.relu(h_self + h_neigh)  # (B, N, out_dim)
        return out


class SimpleBatchedGNN(nn.Module):
    """
    Tiny 2-layer GNN for batched graphs with shared topology.

    Input:  x (B, N, 3)
    Output: x_out (B, N, hidden_dim2)
    """
    def __init__(self, hidden_dim1=128, hidden_dim2=8):
        super().__init__()
        self.gnn1 = BatchedGNNLayer(in_dim=3, out_dim=hidden_dim1)
        self.gnn2 = BatchedGNNLayer(in_dim=hidden_dim1, out_dim=hidden_dim2)

    def forward(self, x, edge_index):
        x = self.gnn1(x, edge_index)
        x = self.gnn2(x, edge_index)
        x, _ = torch.max(x, dim=1)
        return x  # (B, N, hidden_dim2)

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
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, vertices):
        """
        Args:
            vertices (torch.Tensor): Tensor of shape (B, N, 3) or (N, 3) where B is batch size and N is number of vertices.
        Returns:
            torch.Tensor: Encoded frame feature of shape (B, out_dim) (or (out_dim,) for a single frame).
        """
        x = self.mlp(vertices)

        x, _ = torch.max(x, dim=0)  # shape: (B, out_dim)
        return x.unsqueeze(0)

class AtoMWithAttention(nn.Module):
    def __init__(self, f, num_heads=4):
        super().__init__()
        # Map A's 3-dim features to the same dim as B (f)
        self.a_proj = nn.Linear(3, f)
        # Cross-attention: queries from B, keys/values from A
        self.attn = nn.MultiheadAttention(
            embed_dim=f,
            num_heads=num_heads,
            batch_first=True,  # requires PyTorch >= 1.10
        )

    def forward(self, A, B):
        """
        A: [n, 3]
        B: [m, f]
        returns:
            C:   [m, 2f]  (concat of attended A and B)
            A_m: [m, f]   (A mapped to m positions)
        """
        # 1) Project A to f-dim
        A_proj = self.a_proj(A)        # [n, f]

        # 2) Add batch dimension for MultiheadAttention
        A_proj = A_proj.unsqueeze(0)   # [1, n, f]  -> keys/values
        B_in   = B.unsqueeze(0)        # [1, m, f]  -> queries

        # 3) Cross-attention: each B_j attends over all A_i
        #    output: [1, m, f]
        A_m, attn_weights = self.attn(
            query=B_in,
            key=A_proj,
            value=A_proj
        )

        # 4) Remove batch dimension
        A_m = A_m.squeeze(0)           # [m, f]

        # 5) Concatenate along feature dimension
        C = torch.cat([A_m, B], dim=-1)  # [m, 2f]

        return C, A_m, attn_weights.squeeze(0)


class PoissonNetAutoencoder(nn.Module):
    def __init__(self, args):
        super(PoissonNetAutoencoder, self).__init__()
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.latent_channels = args.latent_channels
        self.device = args.device
        self.bs = args.batch_size
        self.n_faces = args.n_faces
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
        ], dtype=torch.long, device="cuda:0")

        # encoder
        self.encoder = poisson_net.PoissonNet(C_in=6,
                                            C_out=self.latent_channels,
                                            C_width=128,
                                            n_blocks=4,
                                            head='linear',
                                            extra_features=0,
                                            outputs_at="faces")
        # decoder
        self.decoder = njf_decoder(latent_features_shape=(self.bs, self.n_faces, self.latent_channels + 32), args=args)

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
        self.encoder_lmk = SimpleBatchedGNN(hidden_dim1=128, hidden_dim2=32)

        #print("encoder parameters: ", count_parameters(self.encoder))
        #print("decoder parameters: ", count_parameters(self.decoder))

        #nn.init.constant_(self.last_layer.weight, 0)
        #nn.init.constant_(self.last_layer.bias, 0)

    def forward_latent_njf(self, template, vertices,
                mass_template, solver_template, G_template, M_template, faces_template, feats, feats_temp):

        z_template = self.encoder(template, M=M_template, G=G_template, solver=solver_template, vertex_mass=mass_template,
                                  faces=faces_template)

        # PC encoding
        z_lmk = feats.squeeze(0)
        z_lmk = self.encoder_lmk(z_lmk, self.edges)
        z = z_lmk.expand((z_template.shape[0], z_template.shape[1], z_lmk.shape[-1]))
        feat_field = torch.cat((z_template, z), dim=-1)

        # Brute concatenation
        #z_lmk = feats.reshape((-1)).unsqueeze(0)
        #z = z_lmk.unsqueeze(1).expand((z_template.shape[0], z_template.shape[1], z_lmk.shape[-1]))
        #feat_field = torch.cat((z_template, z), dim=-1)

        # MLP decoder
        #delta = self.mlp_dec(feat_field)

        # NJF decoder
        delta, pred_jac = self.decoder.predict_map(feat_field, source_verts=template, source_faces=faces_template,
                                        batch=False, target_vertices=None)
        delta, pred_jac = delta.to(self.device), pred_jac.to(self.device)

        pred = delta + template[:, :, :3]
        return pred

