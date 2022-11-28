import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils.gvae_utils import (count_parameters, gvae_loss, 
        slice_edge_type_from_edge_feats, slice_atom_type_from_node_feats)
from models.gvae import GVAE
from config.gvae import DEVICE as device
from datamodules.featurizer import featurize
import random
import wandb
import pickle

wandb.login()
# Load data
data_file = './data/tmQM_X.xyz'
charges_file = './data/tmQM_X.q'
bo_file = './data/tmQM_X.BO'
# data_list = featurize(data_file, charges_file, bo_file)
# with open('data_list.pkl', 'wb') as handle:
#     pickle.dump(data_list, handle)
data_list = []
with open('data_list.pkl', 'rb') as handle:
    data_list = pickle.load(handle)
N = len(data_list)
split = [0.8, 0.2]
N_train = int(N * split[0])
batch_size=32
random.seed(42)
random.shuffle(data_list)
train_data = data_list[:N_train]
test_data = data_list[N_train:]
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Load model

# Define loss and optimizer
loss_fn = gvae_loss
lr = 0.01
kl_beta = 0.8
encoder_embedding_size=64
edge_dim=3
latent_embedding_size=64
decoder_hidden_neurons=128
epochs = 200
print(train_data[0].x.shape, train_data[0].edge_attr.shape, train_data[0].edge_index.shape)
model = GVAE(
            feature_size=train_data[0].x.shape[1], 
            encoder_embedding_size=encoder_embedding_size, 
            edge_dim=edge_dim, 
            latent_embedding_size=latent_embedding_size, 
            decoder_hidden_neurons=decoder_hidden_neurons
            )
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print("Model parameters: ", count_parameters(model))
# Train function
def run_one_epoch(data_loader, type, epoch, kl_beta):
    # Store per batch loss and accuracy 
    all_losses = []
    all_kldivs = []

    # Iterate over data loader
    for _, batch in enumerate(tqdm(data_loader)):
        # Some of the data points have invalid adjacency matrices 
        try:
            # Use GPU
            batch.to(device)  
            # Reset gradients
            optimizer.zero_grad() 
            # Call model
            triu_logits, node_logits, mu, logvar = model(batch.x.float(), 
                                                        batch.edge_attr.float(),
                                                        batch.edge_index, 
                                                        batch.batch) 
            # Calculate loss and backpropagate
            edge_targets = slice_edge_type_from_edge_feats(batch.edge_attr.float())
            node_targets = slice_atom_type_from_node_feats(batch.x.float(), as_index=True)
            loss, kl_div = loss_fn(triu_logits, node_logits,
                                   batch.edge_index, edge_targets, 
                                   node_targets, mu, logvar, 
                                   batch.batch, kl_beta)
            if type == "Train":
                loss.backward()  
                optimizer.step() 
            # Store loss and metrics
            all_losses.append(loss.detach().cpu().numpy())
            #all_accs.append(acc)
            all_kldivs.append(kl_div.detach().cpu().numpy())
            if type == "Train":
                wandb.log({"epoch": epoch, 'loss_kl/train': sum(all_kldivs)/len(all_kldivs), 'loss_recon/train': sum(all_losses)/len(all_losses)})
            if type == "Test":
                wandb.log({"epoch": epoch, 'loss_kl/test': all_kldivs[-1], 'loss_recon/test': all_losses[-1]})
        except IndexError as error:
            # For a few graphs the edge information is not correct
            # Simply skip the batch containing those
            print("Error: ", error)
    
    # Perform sampling
    # if type == "Test":
    #     generated_mols = model.sample_mols(num=250)
    #     print(f"Generated {generated_mols} molecules.")
    #     wandb.log({"Sampled molecules": float(generated_mols), "epoch": epoch})

wandb.init(project="graphVAE", entity="shivanshseth", config={
    "beta": kl_beta,
    "num_layers": 4,
    "edge_dim": edge_dim,
    "decoder_hidden_neurons": decoder_hidden_neurons,
    "encoder_embedding_size": encoder_embedding_size,
    "latent_embedding_size": latent_embedding_size,
    "learning_rate": lr,
    "epochs": epochs,
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

# Run training
for epoch in range(epochs): 
    model.train()
    run_one_epoch(train_loader, type="Train", epoch=epoch, kl_beta=kl_beta)
    if epoch % 5 == 0:
        print("Start test epoch...")
        model.eval()
        run_one_epoch(test_loader, type="Test", epoch=epoch, kl_beta=kl_beta)
torch.save(model, './gvae.m')

generated_mols = model.sample_mols(num=250)
print(f"Generated {generated_mols} molecules.")
wandb.log({"Sampled molecules": float(generated_mols), "epoch": epoch})