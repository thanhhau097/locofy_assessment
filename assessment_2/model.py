import torch
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_dense_adj, to_dense_batch


class Model(torch.nn.Module):
    """Node classification model using GCN"""

    def __init__(
        self,
        hidden_channels,
        graph_in,
        graph_out,
        hidden_dim,
        dropout=0.0,
        num_roles=120,
        max_layout_size=1281,
        role_pad_idx=0,
    ):
        super().__init__()
        self.role_pad_idx = role_pad_idx
        role_embedding_dim = 16
        layout_embedding_dim = 16
        self.role_embedding = torch.nn.Embedding(num_roles, role_embedding_dim, padding_idx=0)
        self.layout_embedding = torch.nn.Embedding(max_layout_size, layout_embedding_dim)
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
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_layout: Tensor, x_role: Tensor, edge_index: Tensor) -> Tensor:
        x = torch.concat(
            [
                self.layout_embedding(x_layout[:, :, 0]),
                self.layout_embedding(x_layout[:, :, 1]),
                self.layout_embedding(x_layout[:, :, 2]),
                self.layout_embedding(x_layout[:, :, 3]),
                self.role_embedding(x_role),
            ],
            dim=-1,
        )
        x = self.linear(
            x
        )  # B, N, D where B is batch size, N is number of nodes, D is embedding dimension

        # prune x to remove self.role_pad_idx padding, find self.role_pad_idx index from x_role
        # list of sequence lengths of each batch element
        lengths = (x_role != self.role_pad_idx).sum(dim=1).cpu()
        x = torch.cat([x[i, : lengths[i]] for i in range(len(lengths))], dim=0)

        # get batch index with this format: [0, 0, 1, 2, 2, 2] to concat x
        batch = (x_role != self.role_pad_idx).nonzero(as_tuple=False)[:, 0]
        # use to_dense_batch to convert x to batched tensor
        x, _ = to_dense_batch(x, batch=batch)

        # if x.shape[1] != x_role.shape[1]:
        #     import pdb; pdb.set_trace()
        #     print(batch)
        #     print(x.shape)
        #     print(x_layout.shape, x_role.shape)

        # pass though conv layers
        for conv in self.convs:
            x = conv(x, edge_index).relu()

        x = self.dense(x)

        return x
