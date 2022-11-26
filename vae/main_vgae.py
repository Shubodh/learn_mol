from xyz2graph import MolGraph, to_networkx_graph, to_plotly_figure
from plotly.offline import init_notebook_mode, iplot
import networkx as nx
import numpy as np
import torch
import re
from itertools import combinations
from math import sqrt
from torch_geometric.nn import DimeNet
from rdkit import Chem
import random
import wandb
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch import Tensor
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from datamodules.featurizer import featurize
from models.vgae import VGAutoEncoder
from utils.loops import train_vgae, test_vgae, isomap
import pickle
wandb.login()
device = 'cuda'

data_file = './data/tmQM_X.xyz'
charges_file = './data/tmQM_X.q'
bo_file = './data/tmQM_X.BO'
# data_list = featurize(data_file, charges_file, bo_file)
# with open('./data/data_list_bo.pkl', 'wb') as handle:
#     pickle.dump(data_list, handle)
with open('./data/data_list_bo.pkl', 'rb') as handle:
    data_list = pickle.load(handle)
N = len(data_list)
split = [0.8, 0.2]
N_train = int(N * split[0])
random.seed(42)
random.shuffle(data_list)
batch_size = 32
lr = 0.001
num_layers = 3
out_channels = 2
num_features = data_list[0].x.shape[1]
epochs = 100
edge_dim = data_list[0].edge_attr.shape[1]
heads = 1
train_data = data_list[:N_train]
test_data = data_list[N_train:]
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
beta=0
model = VGAutoEncoder(num_features, out_channels, num_layers=num_layers, edge_dim=edge_dim, heads=heads)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
torch.save(model.state_dict(), 'temp.m')
model.load_state_dict(torch.load('temp.m'))
config = {
    "model": "vgae",
    "beta": beta,
    "num_layers": num_layers,
    "heads": heads,
    "latent_channels": out_channels,
    "learning_rate": lr,
    "batch_size": batch_size,
    "node_features": num_features,
    "edge_dim": edge_dim
}
wandb.init(project="vgae", entity="mll-metal", config={
    "beta": beta,
    "num_layers": num_layers,
    "heads": heads,
    "latent_channels": out_channels,
    "learning_rate": lr,
    "batch_size": batch_size
})
metrics = [
            "loss_kl/train",
            "loss_kl/val",
            "loss/test",
            "loss_kl/test",
            "loss/val",
            "loss_kl/val",
            "auc/val",
            "ap/val",
            ]
for i in metrics:
    wandb.define_metric(name=i, step_metric='epoch')


for epoch in range(1, epochs + 1):
    loss = train_vgae(epoch, model, train_loader, optimizer, beta)
    test_loss = test_vgae(epoch, model, test_loader)
isomap(model, data_list, f'./maps/vgae_only_recon.png')
