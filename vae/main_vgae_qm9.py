import torch # type: ignore
import random
import wandb
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from models.vgae import VGAutoEncoder, VGAE_Dimenet
from utils.loops import train_vgae, test_vgae, map_latent_space
import pickle
from torch_geometric.datasets import QM9
import sys
import os
import yaml

wandb.login()  # type: ignore
config_file = sys.argv[1]
if not os.path.exists(config_file):
    print("Config file does not exist")
    exit(1)

config = yaml.safe_load(config_file)
device = config.get("device", "cuda")
batch_size = config.get("batch_size", 32)
lr = config.get("lr", 0.001)
num_layers = config.get("num_layers", 1)
out_channels = config.get("out_channels", 2)
epochs = config.get("epochs", 50)
beta = config.get("beta", 0.3)
hidden_channels = config.get("hidden_channels", 8)
save_model = config.get("save_model", False)
load_model = config.get("load_model", "False")
latent_space_file = config.get("latent_space_file", "False")
device = 'cuda'
# parameters
data_list_raw = QM9('./data/qm9')
N = len(data_list_raw)
N = 20000
data_list = [g for g in data_list_raw[:N] if g.x.shape[0] < 20]  # type: ignore
random.seed(42)
random.shuffle(data_list)
N = len(data_list)
print(N)
split = [0.8, 0.2]
N_train = int(N * split[0])
num_features = data_list[0].x.shape[1]
edge_dim = data_list[0].edge_attr.shape[1]
heads = 1
print(f'Node features: {num_features}')
print(f'Edge features: {edge_dim}')
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
# inizialize the optimizer
config = {
    "model": "vgae_dimenet",
    "beta": beta,
    "num_layers": num_layers,
    "heads": heads,
    "latent_channels": out_channels,
    "learning_rate": lr,
    "batch_size": batch_size,
    "node_features": num_features,
    "edge_dim": edge_dim
}
wandb.init(project="vgae", entity="mll-metal", config={  # type: ignore
    "beta": beta,
    "num_layers": num_layers,
    "heads": heads,
    "latent_channels": out_channels,
    "learning_rate": lr,
    "batch_size": batch_size
})
run_name = wandb.run.name # type: ignore
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
    wandb.define_metric(name=i, step_metric='epoch')  # type: ignore

if not load_model == "False": 
    checkpoint = torch.load(load_model)
    model_config = checkpoint['model_config']
    model = VGAutoEncoder(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("Loaded model from checkpoint")
else:
    model = VGAutoEncoder(**model_config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        loss = train_vgae(epoch, model, train_loader, optimizer, beta, encoder='dimenet')
        test_loss = test_vgae(epoch, model, test_loader, encoder='dimenet')

if save_model:
    torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config
            }, 'checkpoints/dimenet' + wandb.run.name + '.chkpt')

if not latent_space_file == "False":
    map_latent_space(model, data_list, latent_space_file)
