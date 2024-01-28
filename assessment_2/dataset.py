import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class AbsoluteDataset(Dataset):
    def __init__(self, paths, roles, split="train", **kwargs):
        self.paths = paths
        self.roles = roles
        self.split = split

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with open(path, "r") as f:
            data = json.load(f)
        
        image_path = data["image_path"]
        nodes = data["nodes"]
        
        node_id_to_idx = {node["id"]: i for i, node in enumerate(nodes)}
        edges = data["edges"]
        edge_index = np.array([[node_id_to_idx[e[0]], node_id_to_idx[e[1]]] for e in edges])
        
        h = max([node["box"][3] for node in nodes])
        w = max([node["box"][2] for node in nodes])
        
        boxes = np.array([node["box"] for node in nodes]).astype(np.int64)        
        layout_feat = torch.tensor(boxes, dtype=torch.long)
        
        roles = [self.roles.index[node["role"]] for node in nodes]
        role_feat = torch.tensor(roles, dtype=torch.long)
        
        edge_index = torch.tensor(np.swapaxes(edge_index, 0, 1).astype(np.int64))
        
        target = [node["is_absolute"] for node in nodes]
        target = torch.tensor(target, dtype=torch.long)
        return layout_feat, role_feat, edge_index, target


def collate_fn(batch):
    layout_feat, role_feat, edge_index, target = zip(*batch)
    max_size = max([len(x) for x in layout_feat])

    # each layout_feat is a n x 4 tensor, where n is the number of nodes in the graph, we need to pad -100 to make them all the same size
    layout_feat = [torch.cat([x, torch.ones(max_size - len(x), 4) * -100]) for x in layout_feat]
    layout_feat = torch.stack(layout_feat)
    
    # each role_feat is a n tensor, where n is the number of nodes in the graph, we need to pad -100 to make them all the same size
    role_feat = [torch.cat([x, torch.ones(max_size - len(x)) * -100]) for x in role_feat]
    role_feat = torch.stack(role_feat)

    # edge_index = torch.stack(edge_index)
    
    target = [torch.cat([x, torch.zeros(max_size - len(x))]) for x in target]
    target = torch.stack(target)

    return {
        "layout_feat": layout_feat,
        "role_feat": role_feat,
        "edge_index": edge_index,
        "target": target,
    }
