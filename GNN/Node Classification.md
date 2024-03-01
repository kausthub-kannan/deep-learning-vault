# Node Classification

We will look through a quick implementation of node classification using the Cora dataset and understand how GNN works.

### Setup and Imports
```python
!pip install torch_geometric --quiet

import torch
from torch_geometric.data import Data
import torch_geometric as pyg
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GCNConv

import os
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def visualise(dataset, label_dict, colorlist=['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']):

  G = pyg.utils.to_networkx(dataset[0], to_undirected=True)
  node_color = []
  nodelist = [[], [], [], [], [], [], []]
  labels = dataset[0].y
  for n, i in enumerate(labels):
      node_color.append(colorlist[i])
      nodelist[i].append(n)
  pos = nx.spring_layout(G, seed = 42)
  plt.figure(figsize = (10, 10))
  labellist = list(label_dict.values())
  for num, i in enumerate(zip(nodelist, labellist)):
      n, l = i[0], i[1]
      nx.draw_networkx_nodes(G, pos, nodelist=n, node_size = 5, node_color = colorlist[num], label=l)
  nx.draw_networkx_edges(G, pos, width = 0.25)
  plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

def gnn_dataset_stats(dataset, label_dict):

  # Dataset overview
  print(f'Dataset: {dataset}:')
  print('======================')
  print(f'Number of graphs: {len(dataset)}')
  print(f'Number of features: {dataset.num_features}')
  print(f'Number of classes: {dataset.num_classes}')
  print('======================')

  # Data instance overview
  data = dataset[0]
  print(f'Number of nodes: {data.num_nodes}')
  print(f'Number of edges: {data.num_edges}')
  print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
  print(f'Number of training nodes: {data.train_mask.sum()}')
  print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
  print(f'Has isolated nodes: {data.has_isolated_nodes()}')
  print(f'Has self-loops: {data.has_self_loops()}')
  print(f'Is undirected: {data.is_undirected()}')

  visualise(dataset, label_dict)
```

### Load the Data
```python
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

cora_dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
label_dict = {
    0: "Theory",
    1: "Reinforcement_Learning",
    2: "Genetic_Algorithms",
    3: "Neural_Networks",
    4: "Probabilistic_Methods",
    5: "Case_Based",
    6: "Rule_Learning"}

gnn_dataset_stats(cora_dataset, label_dict)
```

### Understand Data split
```python
print(f'Number of training nodes: {cora_dataset.train_mask.sum()}')
print(f'Number of validation nodes: {cora_dataset.val_mask.sum()}')
print(f'Number of test nodes: {cora_dataset.test_mask.sum()}')
```

### GNN Model
```python
from torch_geometric.nn import GCNConv

hidden_channels = [16]

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234567)
        self.gc1 = GCNConv(cora_dataset.num_features, hidden_channels[0])
        self.gc2 = GCNConv(hidden_channels[0], cora_dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gc2(x, edge_index)
        return x
```

### Train
```python
model = GCN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

dataset = cora_dataset[0]

history = {"train_loss":[], "val_loss":[]}

for epoch in range(1, 201):
      model.train()
      optimizer.zero_grad()
      out = model(dataset.x, dataset.edge_index)
      train_loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])
      train_loss.backward()
      optimizer.step()
      history["train_loss"].append(train_loss)

      model.eval()
      out = model(dataset.x, dataset.edge_index)
      val_loss = criterion(out[dataset.val_mask], dataset.y[dataset.val_mask])
      history["val_loss"].append(val_loss)

      print(f"Epoch {epoch}: | Train Loss: {train_loss} | Val Loss: {val_loss}")
```

### Evaluate
```python
model.eval()
out = model(dataset.x, dataset.edge_index)
pred = out.argmax(dim=1)  
test_correct = pred[dataset.test_mask] == dataset.y[dataset.test_mask]
test_acc = int(test_correct.sum()) / int(dataset.test_mask.sum())  
print(f"Accuracy: {test_acc}")
```