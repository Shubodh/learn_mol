import torch # type: ignore
import random
import wandb
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from models.vgae import VGAutoEncoder, VGAE_Dimenet
from utils.loops import train_vgae, test_vgae, map_latent_space
import pickle
from torch_geometric.datasets import QM9
wandb.login()  # type: ignore
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
# random.seed(42)
# random.shuffle(data_list)
batch_size = 32
lr = 0.001
num_layers = 1
out_channels = 2
num_features = data_list[0].x.shape[1]
epochs = 100
edge_dim = data_list[0].edge_attr.shape[1]
heads = 1
hidden_channels = 8
print(f'Node features: {num_features}')
print(f'Edge features: {edge_dim}')
train_data = data_list[:N_train]
test_data = data_list[N_train:]
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
beta=0.1
model = VGAE_Dimenet(num_features, hidden_channels, out_channels, num_layers=num_layers, edge_dim=edge_dim, heads=heads))

model = model.to(device)
# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=epochs)
torch.save(model.state_dict(), 'temp.m')
model.load_state_dict(torch.load('temp.m'))
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
run_name = wandb.run.name
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


for epoch in range(1, epochs + 1):
    loss = train_vgae(epoch, model, train_loader, optimizer, beta, encoder='dimenet')
    test_loss = test_vgae(epoch, model, test_loader, encoder='dimenet')
map_latent_space(model, data_list, f'./maps/vgae_only_recon.png', qm9=True)
