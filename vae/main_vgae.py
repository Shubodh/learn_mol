from xyz2graph import MolGraph, to_networkx_graph, to_plotly_figure
from plotly.offline import init_notebook_mode, iplot
import networkx as nx
import numpy as np
import torch
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
from models.vgae import VGAutoEncoder, VGAE_Dimenet
from utils.loops import train_vgae, test_vgae, map_latent_space
import pickle
import sys
import os
import yaml
from torch_geometric.datasets import QM9

config_file = sys.argv[1]
if not os.path.exists(config_file):
    print("Config file does not exist")
    exit(1)
with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)
    device = config.get("device", "cuda")
    data_file = config.get("data_file", "./data/tmQM_X.xyz")
    charges_file = config.get("charges_file", "./data/tmQM_X.q")
    bo_file = config.get("bo_file", "./data/tmQM_X.BO")
    batch_size = config.get("batch_size", 32)
    lr = config.get("lr", 0.001)
    num_layers = config.get("num_layers", 3)
    hidden_channels = config.get("hidden_channels", 8)
    out_channels = config.get("out_channels", 2)
    epochs = config.get("epochs", 50)
    beta = config.get("beta", 0.3)
    dataset = config.get("dataset", "qm9")
    pre_processed_file = config.get("pre_processed_file", "./data/data_list.pkl")
    save_features = config.get("features_file", "./data/data_list.pkl")
    save_model = config.get("save_model", False)
    load_model = config.get("load_model", "False")
    latent_space_file = config.get("latent_space_file", "False")
    save_wandb = config.get("save_wandb", True)

data_list = []
if dataset == 'tmqm':
    if not pre_processed_file  == "False": 
        with open(pre_processed_file, 'rb') as handle:
            data_list = pickle.load(handle)
    else:
        data_list = featurize(data_file, charges_file, bo_file)
        if not save_features == "False":
            with open(save_features, 'wb') as handle:
                pickle.dump(data_list, handle)

if dataset == 'qm9':
    data_list_raw = QM9('./data/qm9')
    N = len(data_list_raw)
    N = 20000
    data_list = [g for g in data_list_raw[:N] if g.x.shape[0] < 20] 

N = len(data_list)
split = [0.8, 0.2]
N_train = int(N * split[0])

random.seed(42)
random.shuffle(data_list)
num_features = data_list[0].x.shape[1]
edge_dim = data_list[0].edge_attr.shape[1]
heads = 1
train_data = data_list[:N_train]
test_data = data_list[N_train:]
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
model_config = {
    "in_channels": num_features,
    "hidden_channels": hidden_channels,
    "out_channels": out_channels,
    "num_layers": num_layers,
    "edge_dim": edge_dim,
    "heads": heads
    }

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
if save_wandb: 
    wandb.login()
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

if not load_model == "False": 
    checkpoint = torch.load(load_model)
    model_config = checkpoint['model_config']
    model = VGAE_Dimenet(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("Loaded model from checkpoint")
else:
    model = VGAE_Dimenet(**model_config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        loss = train_vgae(epoch, model, train_loader, optimizer, beta, save_wandb=save_wandb)
        test_loss = test_vgae(epoch, model, test_loader, save_wandb=save_wandb)

if save_model:
    torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config
            }, 'checkpoints/gatconv' + wandb.run.name + '.chkpt')

if not latent_space_file == "False":
    map_latent_space(model, data_list, latent_space_file, qm9=dataset=='qm9')
