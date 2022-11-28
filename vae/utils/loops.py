import wandb
import torch
from torch_geometric.utils import negative_sampling
from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt

def train_vgae(epoch, model, loader, optimizer, beta=0.2, train=True):
    model.train()
    running_loss_kl = 0
    running_loss = 0
    n = 0
    if train:
        for data in loader:
            data = data.to('cuda')
            n += 1
            optimizer.zero_grad()
            z = model.encode(data.x, data.edge_index, data.edge_attr)
            loss = model.recon_loss(z, data.edge_index)
            #if args.variational:
            kl = model.kl_loss()
            running_loss += loss.item()
            running_loss_kl += kl.item()
            loss = loss + beta * kl
            loss.backward()
            optimizer.step()
        wandb.log({"epoch": epoch, 'loss_kl/train': running_loss_kl/n, 'loss_recon/train': running_loss/n})

    return float((running_loss+running_loss_kl)/n)

def test_vgae(epoch, model, loader):
    model.eval()
    running_loss = 0
    running_loss_kl = 0
    running_auc = 0
    running_ap = 0
    n = 0
    for data in loader: 
        data = data.to('cuda')
        n += 1
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index, data.edge_attr)
            loss = model.recon_loss(z, data.edge_index)
            kl = model.kl_loss()
            running_loss += loss.item()
            running_loss_kl += kl.item()
            neg_edges = negative_sampling(data.edge_index)
            auc, ap = model.test(z, data.edge_index, neg_edges)
            running_auc += auc.item()
            running_ap += ap.item()
    wandb.log({"epoch": epoch, 'loss_kl/val': running_loss_kl/n, 'loss_recon/val': running_loss/n, 'auc/val': running_auc/n, 'ap/val': running_ap/n})
    return float((running_loss+running_loss_kl)/n)

def isomap(model, data, file_name):
    model.eval()
    z = [model.encode(i.x, i.edge_index, i.edge_attr).detach().cpu().numpy() for i in data]
    z = np.vstack(z)
    # for i in data:
    #     print(i.x.shape)
    #     k = model.encode(i.x, i.edge_index, i.edge_attr).detach().cpu().numpy()
    #     print(k.shape)
    # z = np.array(z)
    print('latent space shape:', z.shape)
    # tsne = manifold.TSNE(n_components=2, random_state=42)
    # z_tsne = tsne.fit_transform(z)
    plt.scatter(z[:, 0], z[:, 1])
    plt.savefig(file_name)
