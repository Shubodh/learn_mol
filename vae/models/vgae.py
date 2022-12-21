import torch # type: ignore
from torch_geometric.nn import GCNConv, GATConv, VGAE
from models.dimenet import DimeNet

class VariationalDimenetEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=1, heads=1,num_layers=1):
        super(VariationalDimenetEncoder, self).__init__()
        self.conv1 = []
        self.conv1.append(DimeNet(
            hidden_channels=64,
            out_channels=hidden_channels,
            num_blocks=4,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3, 
        ))
        self.conv1 = torch.nn.ModuleList(self.conv1)
        # self.conv_mu = GCNConv(hidden_channels, out_channels)
        # self.conv_logstd = GCNConv(hidden_channels, out_channels)
        self.conv_mu = GATConv(hidden_channels, out_channels)
        self.conv_logstd = GATConv(hidden_channels, out_channels)

    def forward(self, x, pos, edge_index, edge_weights, batch):
        # print('Input:', x.shape)
        for conv in self.conv1:
            x = conv(x, pos, batch).relu()
        # print('Post_dimenet:', x.shape)
        mu = self.conv_mu(x, edge_index=edge_index)
        # print('Post_mu:', x.shape)
        std = self.conv_logstd(x, edge_index=edge_index)
        return mu, std

def VGAE_Dimenet(in_channels, out_channels, hidden_channels, edge_dim=1, heads=1,num_layers=1):
    return VGAE(VariationalDimenetEncoder(in_channels, hidden_channels, out_channels, num_layers=num_layers, edge_dim=edge_dim, heads=heads))