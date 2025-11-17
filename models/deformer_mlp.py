import torch
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


class PointNet_MLP_AutoEncoder(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.device = args.device
        self.num_points = args.n_points
        self.n_faces = args.n_faces
        self.bs = args.batch_size
        self.device = args.device
        self.latent_channels = args.latent_channels
        self.in_channels = args.in_channels

        self.layers_enc = [nn.Linear(6, 128),
                           nn.ReLU(),
                           nn.Linear(128, 128),
                           nn.ReLU(),
                           nn.Linear(128, 128),
                           nn.ReLU(),
                           nn.Linear(128, self.latent_channels // 2)]
        self.encoder = nn.Sequential(*self.layers_enc)

        self.encoder_def = PNEncoder(in_features=self.in_channels, hidden_dim=256, out_dim=self.latent_channels // 2)

        self.layers_dec = [nn.Linear(self.latent_channels, 128),
                       nn.ReLU(),
                       nn.Linear(128, 128),
                       nn.ReLU(),
                       nn.Linear(128, 128),
                       nn.ReLU(),
                       nn.Linear(128, 3)]

        self.decoder = nn.Sequential(*self.layers_dec)

    def forward(self, template, vertices, feats, feats_temp):
        z_template = self.encoder(template)
        z = self.encoder_def(feats)
        z_ref = self.encoder_def(feats_temp)
        z = z - z_ref
        z = z.unsqueeze(1).expand((z_template.shape[0], z_template.shape[1], z.shape[-1]))
        cat_latent = torch.cat((z_template, z), dim=-1)
        delta = self.decoder(cat_latent)
        pred = delta + template[:, :, :3]
        return pred, cat_latent