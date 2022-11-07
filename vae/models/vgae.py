import torch
from torch_geometric.nn import GCNConv, GATConv, VGAE

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=1, heads=1,num_layers=1):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = []
        self.conv1.append(GATConv(in_channels, 2 * out_channels, edge_dim=edge_dim, heads=heads))
        if num_layers > 1:
            self.conv1.append(GATConv(2 * out_channels, 2 * out_channels, edge_dim=edge_dim, heads=heads))
        self.conv1 = torch.nn.ModuleList(self.conv1)
        
        self.conv_mu = GATConv(2 * out_channels, out_channels, edge_dim=edge_dim, heads=heads)
        self.conv_logstd = GATConv(2 * out_channels, out_channels, edge_dim=edge_dim, heads=heads)

    def forward(self, x, edge_index, edge_weights):
        for conv in self.conv1:
            x = conv(x, edge_index, edge_attr=edge_weights).relu()
        return self.conv_mu(x, edge_index, edge_attr=edge_weights), self.conv_logstd(x, edge_index)

def VGAutoEncoder(in_channels, out_channels, edge_dim=1, heads=1,num_layers=1):
    return VGAE(VariationalGCNEncoder(in_channels, out_channels, num_layers=num_layers, edge_dim=edge_dim, heads=heads))
