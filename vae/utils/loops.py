import wandb
import torch
from torch_geometric.utils import negative_sampling
from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from config.gvae import *
from sklearn.manifold import TSNE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def slice_atom_type_from_node_feats(node_features, as_index=False):
    """
    This function only works for the MolGraphConvFeaturizer used in the dataset.
    It slices the one-hot encoded atom type from the node feature matrix.
    Unknown atom types are not considered and not expected in the datset.
    """
    supported_atoms = SUPPORTED_ATOMS
    atomic_numbers =  ATOMIC_NUMBERS

    # Slice first X entries from the node feature matrix
    atom_types_one_hot = node_features[:, :len(supported_atoms)]
    if not as_index:
        # Map the index to the atomic number
        atom_numbers_dummy = torch.Tensor(atomic_numbers).repeat(atom_types_one_hot.shape[0], 1)
        atom_types = torch.masked_select(atom_numbers_dummy, atom_types_one_hot.bool())
    else:
        atom_types = torch.argmax(atom_types_one_hot, dim=1)
    return atom_types

def train_vgae(epoch, model, loader, optimizer, beta=0.2, train=True, save_wandb=True):
    model.train()
    running_loss_kl = 0
    running_loss = 0
    running_auc = 0
    running_ap = 0
    n = 0
    if train:
        for data in loader:
            data = data.to(device)
            n += 1
            optimizer.zero_grad()
            z = model.encode(data.z, data.pos, data.edge_index, data.edge_attr, data.batch)
            loss = model.recon_loss(z, data.edge_index)
            #if args.variational:
            kl = model.kl_loss()
            running_loss += loss.item()
            running_loss_kl += kl.item()
            loss = loss + beta * kl
            neg_edges = negative_sampling(data.edge_index) # samples equal number of negative edges to match number of positive edges
            auc, ap = model.test(z, data.edge_index, neg_edges)
            running_auc += auc.item()
            running_ap += ap.item()
            loss.backward()
            optimizer.step()
        if save_wandb:
            wandb.log({"epoch": epoch, 'loss_kl/train': running_loss_kl/n, 'loss_recon/train': running_loss/n,
                   'auc/train': running_auc/n, 'ap/train': running_ap/n})

    return float((running_loss+running_loss_kl)/n)

def test_vgae(epoch, model, loader, save_wandb=True):
    model.eval()
    running_loss = 0
    running_loss_kl = 0
    running_auc = 0
    running_ap = 0
    n = 0
    for data in loader: 
        n += 1
        data = data.to(device)
        with torch.no_grad():
            z = model.encode(data.z, data.pos, data.batch)
            loss = model.recon_loss(z, data.edge_index)
            kl = model.kl_loss()
            running_loss += loss.item()
            running_loss_kl += kl.item()
            neg_edges = negative_sampling(data.edge_index)
            auc, ap = model.test(z, data.edge_index, neg_edges)
            running_auc += auc.item()
            running_ap += ap.item()
    if save_wandb:
        wandb.log({"epoch": epoch, 'loss_kl/val': running_loss_kl/n, 'loss_recon/val': running_loss/n, 'auc/val': running_auc/n, 'ap/val': running_ap/n})
    return float((running_loss+running_loss_kl)/n) 

def map_latent_space(model, data, file_name, qm9=False, ind=False, mol_ind=False):
    PT = Chem.GetPeriodicTable()
    model.eval()
    atom_types = []
    types = {0:'H' , 1:'C' , 2:'N', 3:'O', 4:'F'}
    n_types = len(types)
    print(n_types)
    mol_idx = []
    for idx, g in enumerate(data):
        if qm9:
            type_one_hot = g.x[:, :n_types]
            t = list(torch.argmax(type_one_hot, dim=1).cpu().numpy())
            # print(idx)
            atom_types.extend(t)
            mi = [idx]*len(t)
            mol_idx.extend(mi)
        else:
            atom_types.extend(slice_atom_type_from_node_feats(g.x).cpu().numpy())
    # print('starting encoding')
    atom_types = np.array(atom_types)
    mol_idx = np.array(mol_idx)
    data = [i.to(device) for i in data]
    z = []
    for idx, i in enumerate(data):
        z.append( model.encode(i.x, i.edge_index, i.edge_attr).detach().cpu().numpy() )
    z = np.vstack(z)
    if z.shape[1] != 2:
        z = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(z)
    for mi in np.unique(mol_idx):
        for i in np.unique(atom_types):
            print(f"atom type: {i}")
            if qm9: label = types[i]
            else: label = PT.GetElementSymbol(i)
            if mol_ind:
                label+= f' {mi}'
            z_filt = z[np.where(np.logical_and(mol_idx == mi, atom_types==i))]
            if z_filt.shape[0] == 0: continue
            plt.scatter(z_filt, z_filt, label=label)
            plt.legend()

        if ind: plt.show()
    if not ind:
        plt.legend()
        # plt.show()
        plt.savefig(file_name)