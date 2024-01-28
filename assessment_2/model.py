import torch
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv


class Model(torch.nn.Module):
    """ Node classification model using GCN"""
    def __init__(self, hidden_channels, graph_in, graph_out, hidden_dim, dropout=0.0, num_roles=120, max_layout_size=1025):
        super().__init__()
        role_embedding_dim = 16
        layout_embedding_dim = 16
        self.role_embedding = torch.nn.Embedding(num_roles, role_embedding_dim, padding_idx=-100)
        self.layout_embedding = torch.nn.Embedding(max_layout_size, layout_embedding_dim, padding_idx=-100)
        assert len(hidden_channels) > 0

        self.linear = nn.Linear(role_embedding_dim + layout_embedding_dim * 4, graph_in)
        in_channels = graph_in
        self.convs = torch.nn.ModuleList()
        last_dim = hidden_channels[0]
        conv = SAGEConv
        self.convs.append(conv(in_channels, hidden_channels[0]))
        for i in range(len(hidden_channels) - 1):
            self.convs.append(conv(hidden_channels[i], hidden_channels[i + 1]))
            last_dim = hidden_channels[i + 1]
        self.convs.append(conv(last_dim, graph_out))

        self.dense = torch.nn.Sequential(
            nn.Linear(graph_out, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_layout: Tensor, x_role: Tensor, edge_index: Tensor) -> Tensor:
        x = torch.concat([
            self.layout_embedding(x_layout[:, 0]),
            self.layout_embedding(x_layout[:, 1]),
            self.layout_embedding(x_layout[:, 2]),
            self.layout_embedding(x_layout[:, 3]),
            self.role_embedding(x_role)
        ], dim=1)
        x = self.linear(x)
        
        # TODO: prune x to remove -100 padding, find -100 index from x_role
        mask = (x_role != -100).unsqueeze(1).expand(-1, x.shape[1])
        x = torch.masked_select(x, mask)
        
        # TODO: use to_dense_adj to convert edge_index to adjacency matrix
        
        # TODO: use to_dense_batch to convert x to batched tensor
        
        # pass though conv layers
        for conv in self.convs:
            x = conv(x, edge_index).relu()

        x = self.dense(x)

        return x

