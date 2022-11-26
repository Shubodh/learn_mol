import networkx as nx
from datamodules.molgraph import MolGraph
from rdkit import Chem
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch import Tensor
from config.gvae import MAX_MOLECULE_SIZE, ATOMIC_NUMBERS, SUPPORTED_ATOMS, SUPPORTED_EDGES
from sklearn.preprocessing import OneHotEncoder
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_charges(charge_file):
    charges = open(charge_file,"r").read().split('CSD_code = ')
    charges_raw = [i.splitlines()[:-1] for i in charges[1:]]
    charges = {}
    csd_codes = []
    for mol in charges_raw:
        res = []
        csd_codes.append(mol[0])
        for k in mol[1:]:
            c = k.split()[1]
            if c == 'charge': continue # for 'Total charge = x'
            res.append(float(c))
        charges[csd_codes[-1]] = res
    return charges, set(csd_codes)

def get_bond_orders(bo_file):
    BO = open(bo_file,"r").read().split('CSD_code = ')
    BO = [i.splitlines()[:-1] for i in BO[1:]]
    bond_orders = {}
    csd_codes = []
    for mol in BO:
        res = {}
        # if 'Fe' in mol[1]:
        if True:
            csd_codes.append(mol[0])
            for k in mol[1:]:
                k = k.split()
                p_idx = int(k[0])-1
                p_atom = k[1]
                i = 3
                while i < len(k)-1:
                    c_atom, c_idx, bo = k[i], int(k[i+1])-1, float(k[i+2])
                    # print(f'{c_atom}, {c_idx}, {bo}')
                    res[frozenset([c_idx, p_idx])] = bo
                    i += 3
            bond_orders[csd_codes[-1]] = res
    return bond_orders, set(csd_codes)

PT = Chem.GetPeriodicTable()
def to_networkx_graph(graph: MolGraph) -> nx.Graph:
    """Creates a NetworkX graph.
    Atomic elements and coordinates are added to the graph as node attributes 'element' and 'xyz" respectively.
    Bond lengths are added to the graph as edge attribute 'length''"""
    G = nx.Graph(graph.adj_list)
    node_attrs = {}
    for num, (element, xyz) in enumerate(graph):
       anum = PT.GetAtomicNumber(element)
       enc = OneHotEncoder(categories=[ATOMIC_NUMBERS])
       anum_one_hot = enc.fit_transform([[anum]]).toarray()[0]
       x = list(anum_one_hot)
       x.append(graph.charges[num])
       node_attrs[num] = {}
       node_attrs[num]['x'] = x
    nx.set_node_attributes(G, node_attrs)
    nx.set_node_attributes(G, node_attrs)
    edge_attrs = {edge: {'x': [graph.bond_orders[edge], length]} for edge, length in graph.bond_lengths.items()}
    nx.set_edge_attributes(G, edge_attrs)
    return G

def to_networkx_graph_et(graph: MolGraph) -> nx.Graph:
    bondTypeDict = {
        1: "SINGLE",
        2: "DOUBLE",
        3: "TRIPLE"
    }
    G = nx.Graph(graph.adj_list)
    node_attrs = {}
    for num, (element, xyz) in enumerate(graph):
       anum = PT.GetAtomicNumber(element)
       enc = OneHotEncoder(categories=[ATOMIC_NUMBERS])
       anum_one_hot = enc.fit_transform([[anum]]).toarray()[0]
       x = list(anum_one_hot)
    #    xyz = list(xyz)
    #    x.extend(xyz)
       node_attrs[num] = {}
       node_attrs[num]['x'] = x
    nx.set_node_attributes(G, node_attrs)
    edge_attrs = {}
    for edge, length in graph.bond_lengths.items():
        bo = int(round(graph.bond_orders[edge]))
        bt = bondTypeDict.get(bo, "SINGLE")
        enc = OneHotEncoder(categories=[SUPPORTED_EDGES])
        bt_one_hot = enc.fit_transform([[bt]]).toarray()[0]
        edge_attrs[edge] = {}
        edge_attrs[edge]['x'] = list(bt_one_hot)
    nx.set_edge_attributes(G, edge_attrs)
    return G

def featurize(data_file, charges_file, bo_file):
    bond_orders, csd_codes_bo = get_bond_orders(bo_file)
    charges, csd_codes_c = get_charges(charges_file)
    valid_csd_codes = csd_codes_c & csd_codes_bo 

    data = open(data_file).read().splitlines()
    graphs = []
    nodes = []
    data_list = []
    for ndx, line in enumerate(data):
        if ndx < len(data)-1:
            if line == '':
                total_atoms_in_mol = int(data[ndx+1])
                #print(total_atoms_in_mol,ndx+1+total_atoms_in_mol)
                csd_code = data[ndx+2].split()[2]
                # print(csd_code)
                mol_xyz = data[ndx+1:ndx+3+total_atoms_in_mol]
                #finds complexes containing Fe (Iron)
                # if csd_code in valid_csd_codes and total_atoms_in_mol < MAX_MOLECULE_SIZE:
                if 'Fe' in np.array(mol_xyz)[1] and csd_code in valid_csd_codes:
                    mol = MolGraph()
                    # Read the data from the xyz coordinate block
                    mol.read_xyz(mol_xyz, bond_orders[csd_code], charges[csd_code])
                    elements = set(mol.elements)
                    nodes.append(mol.elements)
                    G = to_networkx_graph_et(mol)
                    # if 0 not in G: continue
                    # bfs = nx.bfs_tree(G, source=0)
                    # p = from_networkx(bfs)
                    p = from_networkx(G)
                    # recreating node and edge attr lists in bfs node ordering
                    G = G.to_directed()
                    graphs.append(G)
                    p.x = Tensor([G.nodes[i]['x'] for i in G.nodes])
                    p.x = p.x.to(device)
                    p.edge_attr = Tensor([G.edges[i]['x'] for i in G.edges])
                    p.edge_attr = p.edge_attr.to(device)
                    p.edge_index = p.edge_index.to(device)
                    data_list.append(p)    
    return data_list
