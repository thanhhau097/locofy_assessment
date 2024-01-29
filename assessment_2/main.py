"""FastAPI serving"""

import numpy as np
import ray
import requests
from fastapi import FastAPI
from ray import serve
import safetensors
import torch

from model import Model
from pydantic import BaseModel


class Data(BaseModel):
    nodes: list
    edges: list
    image_path: str


ROLES = [
    "<PAD_TOKEN>",
    "Abbr",
    "Audio",
    "Canvas",
    "DescriptionList",
    "DescriptionListDetail",
    "DescriptionListTerm",
    "Details",
    "DisclosureTriangle",
    "EmbeddedObject",
    "Figcaption",
    "FooterAsNonLandmark",
    "HeaderAsNonLandmark",
    "Iframe",
    "IframePresentational",
    "LabelText",
    "LayoutTable",
    "LayoutTableCell",
    "LayoutTableRow",
    "Legend",
    "LineBreak",
    "ListMarker",
    "PluginObject",
    "Pre",
    "Section",
    "StaticText",
    "SvgRoot",
    "Video",
    "alert",
    "alertdialog",
    "application",
    "article",
    "banner",
    "blockquote",
    "button",
    "caption",
    "checkbox",
    "code",
    "columnheader",
    "combobox",
    "complementary",
    "contentinfo",
    "deletion",
    "dialog",
    "document",
    "emphasis",
    "figure",
    "form",
    "generic",
    "gridcell",
    "group",
    "heading",
    "img",
    "insertion",
    "link",
    "list",
    "listbox",
    "listitem",
    "main",
    "menu",
    "menubar",
    "menuitem",
    "navigation",
    "none",
    "option",
    "paragraph",
    "progressbar",
    "radio",
    "radiogroup",
    "region",
    "row",
    "rowgroup",
    "rowheader",
    "search",
    "searchbox",
    "separator",
    "slider",
    "spinbutton",
    "status",
    "strong",
    "subscript",
    "superscript",
    "tab",
    "table",
    "tablist",
    "tabpanel",
    "textbox",
    "time",
    "timer",
    "toolbar",
    "tooltip",
    "tree",
    "treeitem",
]


def load_model():
    model = Model(
        hidden_channels=[64, 64, 64, 64],
        graph_in=256,
        graph_out=256,
        hidden_dim=256,
        dropout=0,
        num_roles=len(ROLES),
        role_pad_idx=0,
    )

    weights_path = "./outputs/model.safetensors"
    model.load_state_dict(safetensors.torch.load_file(weights_path))
    model.eval()

    return model


app = FastAPI()
model = load_model()


@serve.deployment
@serve.ingress(app)
class Deployment:
    @app.post("/predict")
    def predict(self, data: Data):
        nodes = data.nodes

        node_id_to_idx = {node["id"]: i for i, node in enumerate(nodes)}
        edges = data.edges
        edge_index = np.array(
            [
                [node_id_to_idx[e[0]], node_id_to_idx[e[1]]]
                for e in edges
                if e[0] in node_id_to_idx and e[1] in node_id_to_idx
            ]
        )

        boxes = np.array([node["box"] for node in nodes]).astype(np.int64)
        boxes[boxes < 0] = 0
        # remove larger value than max image size
        boxes[boxes[:, 0] > 1280, 0] = 1280
        boxes[boxes[:, 1] > 720, 1] = 720
        boxes[boxes[:, 2] > 1280, 2] = 1280
        boxes[boxes[:, 3] > 720, 3] = 720

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # height

        layout_feat = torch.tensor(boxes, dtype=torch.long)

        roles = [ROLES.index(node["role"]) for node in nodes]
        role_feat = torch.tensor(roles, dtype=torch.long)

        edge_index = torch.tensor(np.swapaxes(edge_index, 0, 1).astype(np.int64))

        output = model(
            x_layout=torch.stack([layout_feat]),
            x_role=torch.stack([role_feat]),
            edge_index=edge_index,
        )[0]
        predictions = torch.softmax(output, dim=-1)
        predictions = predictions.cpu().detach().numpy()

        predictions = np.argmax(predictions, axis=1)

        # return node that has prediction = 1
        return [node for node, pred in zip(nodes, predictions) if pred == 1]


if __name__ == "__main__":
    data = {
        "image_path": "",
        "nodes": [
            {
                "id": "1",
                "box": [0, 0, 1280, 10],
                "is_absolute": False,
                "role": "none",
            },
            {
                "id": "2",
                "box": [0, 0, 1280, 0],
                "is_absolute": False,
                "role": "generic",
            },
        ],
        "edges": [
            ["1", "2"],
        ],
    }
    serve.run(Deployment.bind(), route_prefix="/")

    resp = requests.post(
        "http://localhost:8000/predict",
        json=data,
        headers={"Content-Type": "application/json"},
    )
    print(resp.json())
