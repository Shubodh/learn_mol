import networkx as nx
from datamodules.molgraph import MolGraph
from rdkit import Chem
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch import Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    return bond_orders, csd_codes

PT = Chem.GetPeriodicTable()
def to_networkx_graph(graph: MolGraph) -> nx.Graph:
    """Creates a NetworkX graph.
    Atomic elements and coordinates are added to the graph as node attributes 'element' and 'xyz" respectively.
    Bond lengths are added to the graph as edge attribute 'length''"""
    G = nx.Graph(graph.adj_list)
    node_attrs = {num: {'x': [PT.GetAtomicNumber(element), xyz[0], xyz[1], xyz[2]], 'xyz': xyz} for num, (element, xyz) in enumerate(graph)}
    nx.set_node_attributes(G, node_attrs)
    edge_attrs = {edge: {'x': [graph.bond_orders[edge], length]} for edge, length in graph.bond_lengths.items()}
    nx.set_edge_attributes(G, edge_attrs)
    return G

def featurize(data_file, charges_file=None, bo_file=None):
    csd_codes = []
    if bo_file is not None:
        bond_orders, csd_codes = get_bond_orders(bo_file)
    data = open(data_file).read().splitlines()
    if charges_file is not None:
        charges = open(charges_file).read().splitlines()
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
                mol_xyz = data[ndx+1:ndx+1+total_atoms_in_mol]
                #finds complexes containing Fe (Iron)
                if csd_code in csd_codes and total_atoms_in_mol < 30:
                # if 'Fe' in np.array(mol_xyz)[1]:
                    mol = MolGraph()
                    # Read the data from the xyz coordinate block
                    mol.read_xyz(mol_xyz, bond_orders[csd_code])
                    elements = set(mol.elements)
                    nodes.append(mol.elements)
                    G = to_networkx_graph(mol)
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
