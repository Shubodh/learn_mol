import re
from itertools import combinations
from numpy import sqrt
import networkx as nx
from rdkit import Chem

from sklearn.preprocessing import OneHotEncoder
import numpy as np
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_MOLECULE_SIZE=20
SUPPORTED_EDGES = ["SINGLE", "DOUBLE", "TRIPLE"]
SUPPORTED_ATOMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 26, 33, 34, 35, 53]
ATOMIC_NUMBERS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 26, 33, 34, 35, 53]
DEVICE = "cuda"

atomic_radii = dict(Ac=1.88, Ag=1.59, Al=1.35, Am=1.51, As=1.21, Au=1.50, B=0.83, Ba=1.34, Be=0.35, Bi=1.54, Br=1.21,
                    C=0.68, Ca=0.99, Cd=1.69, Ce=1.83, Cl=0.99, Co=1.33, Cr=1.35, Cs=1.67, Cu=1.52, D=0.23, Dy=1.75,
                    Er=1.73, Eu=1.99, F=0.64, Fe=1.34, Ga=1.22, Gd=1.79, Ge=1.17, H=0.23, Hf=1.57, Hg=1.70, Ho=1.74,
                    I=1.40, In=1.63, Ir=1.32, K=1.33, La=1.87, Li=0.68, Lu=1.72, Mg=1.10, Mn=1.35, Mo=1.47, N=0.68,
                    Na=0.97, Nb=1.48, Nd=1.81, Ni=1.50, Np=1.55, O=0.68, Os=1.37, P=1.05, Pa=1.61, Pb=1.54, Pd=1.50,
                    Pm=1.80, Po=1.68, Pr=1.82, Pt=1.50, Pu=1.53, Ra=1.90, Rb=1.47, Re=1.35, Rh=1.45, Ru=1.40, S=1.02,
                    Sb=1.46, Sc=1.44, Se=1.22, Si=1.20, Sm=1.80, Sn=1.46, Sr=1.12, Ta=1.43, Tb=1.76, Tc=1.35, Te=1.47,
                    Th=1.79, Ti=1.47, Tl=1.55, Tm=1.72, U=1.58, V=1.33, W=1.37, Y=1.78, Yb=1.94, Zn=1.45, Zr=1.56)


class MolGraph:
    """Represents a molecular graph."""
    __slots__ = ['elements', 'x', 'y', 'z', 'adj_list',
                 'atomic_radii', 'bond_lengths', 'bond_orders', 'charges']

    def __init__(self):
        self.elements = []
        self.x = []
        self.y = []
        self.z = []
        self.adj_list = {}
        self.atomic_radii = []
        self.charges = None
        self.bond_lengths = {}
        self.bond_orders = None

    def read_xyz(self, molxyz, bo=None, charges=None):
        """Reads an XYZ file, searches for elements and their cartesian coordinates
        and adds them to corresponding arrays."""
        pattern = re.compile(
            r'([A-Za-z]{1,3})\s*(-?\d+(?:\.\d+)?)\s*(-?\d+(?:\.\d+)?)\s*(-?\d+(?:\.\d+)?)')
        for element, x, y, z in pattern.findall(str(molxyz)):
            self.elements.append(element)
            self.x.append(float(x))
            self.y.append(float(y))
            self.z.append(float(z))
        self.atomic_radii = [atomic_radii[element]
                             for element in self.elements]
        if bo is not None:
            self.bond_orders = bo
        if charges is not None:
            self.charges = charges
        self._generate_adjacency_list()

    def _generate_adjacency_list(self):
        """Generates an adjacency list from atomic cartesian coordinates."""
        node_ids = range(len(self.elements))
        for i, j in combinations(node_ids, 2):
            x_i, y_i, z_i = self.__getitem__(i)[1]
            x_j, y_j, z_j = self.__getitem__(j)[1]
            distance = sqrt((x_i - x_j) ** 2 + (y_i - y_j)
                            ** 2 + (z_i - z_j) ** 2)
            if self.bond_orders is None:
                if 0.1 < distance < (self.atomic_radii[i] + self.atomic_radii[j]) * 1.3:
                    dist_limit = (
                        self.atomic_radii[i] + self.atomic_radii[j]) * 1.3
                    self.adj_list.setdefault(i, set()).add(j)
                    self.adj_list.setdefault(j, set()).add(i)
                    self.bond_lengths[frozenset([i, j])] = round(distance, 5)
            else:
                if frozenset([i, j]) in self.bond_orders:
                    dist_limit = (
                        self.atomic_radii[i] + self.atomic_radii[j]) * 1.3
                    self.bond_lengths[frozenset([i, j])] = round(distance, 5)
                    self.adj_list.setdefault(i, set()).add(j)
                    self.adj_list.setdefault(j, set()).add(i)
                assert len(self.bond_orders) > 0, f'{len(self.bond_orders)}'

    def edges(self):
        """Creates an iterator with all graph edges."""
        edges = set()
        for node, neighbours in self.adj_list.items():
            for neighbour in neighbours:
                edge = frozenset([node, neighbour])
                if edge in edges:
                    continue
                edges.add(edge)
                yield node, neighbour

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, position):
        return self.elements[position], (
            self.x[position], self.y[position], self.z[position])


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
    NOTE: This function was changed multiple times. Sticking with this version for BFS task, i.e. the task of ordering atoms in Fe molecule in BFS, see `learn_mol/text_learning/bfs_mol_graph.py`.
    Atomic elements and coordinates are added to the graph as node attributes 'element' and 'xyz" respectively.
    Bond lengths are added to the graph as edge attribute 'length''
    """
    G = nx.Graph(graph.adj_list)
    node_attrs = {num: {'x': [PT.GetAtomicNumber(element), xyz[0], xyz[1], xyz[2]], 'xyz': xyz} for num, (element, xyz) in enumerate(graph)}
    nx.set_node_attributes(G, node_attrs)
    edge_attrs = {edge: {'x': [graph.bond_orders[edge], length]} for edge, length in graph.bond_lengths.items()}
    nx.set_edge_attributes(G, edge_attrs)
    return G


def featurize(data_file, charges_file, bo_file):
    bond_orders, csd_codes_bo = get_bond_orders(bo_file)
    charges, csd_codes_c = get_charges(charges_file)
    valid_csd_codes = csd_codes_c & csd_codes_bo

    data = open(data_file).read().splitlines()
    graphs = []
    csd_codes = []
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
                if 'Fe' in np.array(mol_xyz)[1] and csd_code in valid_csd_codes and total_atoms_in_mol < MAX_MOLECULE_SIZE:
                # if 'Fe' in np.array(mol_xyz)[1] and csd_code in valid_csd_codes:
                    mol = MolGraph()
                    # Read the data from the xyz coordinate block
                    mol.read_xyz(mol_xyz, bond_orders[csd_code], charges[csd_code])
                    elements = set(mol.elements)
                    nodes.append(mol.elements)
                    G = to_networkx_graph(mol)
                    # if 0 not in G: continue
                    # bfs = nx.bfs_tree(G, source=0)


                    # # p = from_networkx(bfs)
                    # p = from_networkx(G)
                    # # recreating node and edge attr lists in bfs node ordering
                    # G = G.to_directed()
                    graphs.append(G)
                    csd_codes.append(csd_code)
                    # p.x = Tensor([G.nodes[i]['x'] for i in G.nodes])
                    # p.x = p.x.to(device)
                    # p.edge_attr = Tensor([G.edges[i]['x'] for i in G.edges])
                    # p.edge_attr = p.edge_attr.to(device)
                    # p.edge_index = p.edge_index.to(device)
                    # data_list.append(p)
                    # if len(graphs) == 3: break
    return graphs, csd_codes
