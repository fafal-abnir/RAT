from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F


class RAT(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)
        self.conv_role = GCNConv(input_channels, hidden_channels)

    def forward(self, x, edge_index_normal, edge_index_role):
        x = self.conv_role(x, edge_index_role)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index_normal)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index_normal)
        return x
