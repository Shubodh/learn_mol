"""
GIVEN BO Bond order file as input:
Module for generating rdkit molobj/smiles/molecular graph from free atoms

"""

import copy
import itertools

from collections import defaultdict

import numpy as np
import networkx as nx

import sys

import helper_graph as helg #import BOadj_to_BOmat

from rdkit.Chem import rdmolops
from rdkit.Chem import rdchem
try:
    from rdkit.Chem import rdEHTTools #requires RDKit 2019.9.1 or later
except ImportError:
    rdEHTTools = None

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

global __ATOM_LIST__
__ATOM_LIST__ = \
    ['h',  'he',
     'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne',
     'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar',
     'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu',
     'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',
     'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag',
     'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe',
     'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy',
     'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt',
     'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',
     'fr', 'ra', 'ac', 'th', 'pa', 'u',  'np', 'pu']


global atomic_valence
global atomic_valence_electrons

atomic_valence = defaultdict(list)
atomic_valence[1] = [1]
atomic_valence[5] = [3,4]
atomic_valence[6] = [4]
atomic_valence[7] = [3,4]
atomic_valence[8] = [2,1,3]
atomic_valence[9] = [1]
atomic_valence[14] = [4]
atomic_valence[15] = [5,3] #[5,4,3]
atomic_valence[16] = [6,3,2] #[6,4,2]
atomic_valence[17] = [1]
atomic_valence[32] = [4]
atomic_valence[35] = [1]
atomic_valence[53] = [1]

atomic_valence_electrons = {}
atomic_valence_electrons[1] = 1
atomic_valence_electrons[5] = 3
atomic_valence_electrons[6] = 4
atomic_valence_electrons[7] = 5
atomic_valence_electrons[8] = 6
atomic_valence_electrons[9] = 7
atomic_valence_electrons[14] = 4
atomic_valence_electrons[15] = 5
atomic_valence_electrons[16] = 6
atomic_valence_electrons[17] = 7
atomic_valence_electrons[32] = 4
atomic_valence_electrons[35] = 7
atomic_valence_electrons[53] = 7

def int_atom(atom):
    """
    convert str atom to integer atom
    """
    global __ATOM_LIST__
    #print(atom)
    atom = atom.lower()
    return __ATOM_LIST__.index(atom) + 1

def get_proto_mol(atoms):
    """
    """
    mol = Chem.MolFromSmarts("[#" + str(atoms[0]) + "]")
    rwMol = Chem.RWMol(mol)
    for i in range(1, len(atoms)):
        a = Chem.Atom(atoms[i])
        rwMol.AddAtom(a)

    mol = rwMol.GetMol()

    return mol

def get_AC(mol, covalent_factor=1.3):
    """

    Generate adjacent matrix from atoms and coordinates.

    AC is a (num_atoms, num_atoms) matrix with 1 being covalent bond and 0 is not


    covalent_factor - 1.3 is an arbitrary factor

    args:
        mol - rdkit molobj with 3D conformer

    optional
        covalent_factor - increase covalent bond length threshold with facto

    returns:
        AC - adjacent matrix

    """

    # Calculate distance matrix
    dMat = Chem.Get3DDistanceMatrix(mol)

    pt = Chem.GetPeriodicTable()
    num_atoms = mol.GetNumAtoms()
    AC = np.zeros((num_atoms, num_atoms), dtype=int)

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        Rcov_i = pt.GetRcovalent(a_i.GetAtomicNum()) * covalent_factor
        for j in range(i + 1, num_atoms):
            a_j = mol.GetAtomWithIdx(j)
            Rcov_j = pt.GetRcovalent(a_j.GetAtomicNum()) * covalent_factor
            if dMat[i, j] <= Rcov_i + Rcov_j:
                AC[i, j] = 1
                AC[j, i] = 1

    return AC


def read_xyz_file(filename, look_for_charge=True):
    """
    """

    atomic_symbols = []
    xyz_coordinates = []
    charge = 0
    title = ""

    with open(filename, "r") as file:
        for line_number, line in enumerate(file):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                title = line
                if "charge=" in line:
                    charge = int(line.split("=")[1])
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y), float(z)])

    atoms = [int_atom(atom) for atom in atomic_symbols]

    return atoms, charge, xyz_coordinates, num_atoms

def xyz2AC(atoms, xyz, charge, use_huckel=False):
    """

    atoms and coordinates to atom connectivity (AC)

    args:
        atoms - int atom types
        xyz - coordinates
        charge - molecule charge

    optional:
        use_huckel - Use Huckel method for atom connecitivty

    returns
        ac - atom connectivity matrix
        mol - rdkit molecule

    """

    if use_huckel:
        return xyz2AC_huckel(atoms, xyz, charge)
    else:
        return xyz2AC_vdW(atoms, xyz)


def xyz2AC_vdW(atoms, xyz):

    # Get mol template
    mol = get_proto_mol(atoms)

    # Set coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)

    AC = get_AC(mol)

    return AC, mol

def xyz2AC_huckel(atomicNumList, xyz, charge):
    """

    args
        atomicNumList - atom type list
        xyz - coordinates
        charge - molecule charge

    returns
        ac - atom connectivity
        mol - rdkit molecule

    """
    mol = get_proto_mol(atomicNumList)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i,(xyz[i][0],xyz[i][1],xyz[i][2]))
    mol.AddConformer(conf)

    num_atoms = len(atomicNumList)
    AC = np.zeros((num_atoms,num_atoms)).astype(int)

    mol_huckel = Chem.Mol(mol)
    mol_huckel.GetAtomWithIdx(0).SetFormalCharge(charge) #mol charge arbitrarily added to 1st atom    

    passed,result = rdEHTTools.RunMol(mol_huckel)
    opop = result.GetReducedOverlapPopulationMatrix()
    tri = np.zeros((num_atoms, num_atoms))
    tri[np.tril(np.ones((num_atoms, num_atoms), dtype=bool))] = opop #lower triangular to square matrix
    for i in range(num_atoms):
        for j in range(i+1,num_atoms):
            pair_pop = abs(tri[j,i])   
            if pair_pop >= 0.15: #arbitry cutoff for bond. May need adjustment
                AC[i,j] = 1
                AC[j,i] = 1

    return AC, mol

# def AC2mol(mol, AC, atoms, charge, allow_charged_fragments=True, 
def AC2mol(mol, ACmat, BOmat, atoms, charge, allow_charged_fragments=True, 
           use_graph=True, use_atom_maps=False):
    """
    """

    # convert AC matrix to bond order (BO) matrix
    # BO, atomic_valence_electrons = AC2BO(
    #     AC,
    #     atoms,
    #     charge,
    #     allow_charged_fragments=allow_charged_fragments,
    #     use_graph=use_graph)

    # print("dbef here", BO, AC)

    # add BO connectivity and charge info to mol object
    mol = BO2mol(
        mol,
        BOmat,
        atoms,
        atomic_valence_electrons,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_atom_maps=use_atom_maps)

    # If charge is not correct don't return mol
    if Chem.GetFormalCharge(mol) != charge:
        return []

    # BO2mol returns an arbitrary resonance form. Let's make the rest
    mols = rdchem.ResonanceMolSupplier(mol, Chem.UNCONSTRAINED_CATIONS, Chem.UNCONSTRAINED_ANIONS)
    mols = [mol for mol in mols]

    return mols



def BO2mol(mol, BO_matrix, atoms, atomic_valence_electrons,
           mol_charge, allow_charged_fragments=True,  use_atom_maps=False):
    """
    based on code written by Paolo Toscani

    From bond order, atoms, valence structure and total charge, generate an
    rdkit molecule.

    args:
        mol - rdkit molecule
        BO_matrix - bond order matrix of molecule
        atoms - list of integer atomic symbols
        atomic_valence_electrons -
        mol_charge - total charge of molecule

    optional:
        allow_charged_fragments - bool - allow charged fragments

    returns
        mol - updated rdkit molecule with bond connectivity

    """

    l = len(BO_matrix)
    l2 = len(atoms)
    BO_valences = list(BO_matrix.sum(axis=1))

    if (l != l2):
        raise RuntimeError('sizes of adjMat ({0:d}) and Atoms {1:d} differ'.format(l, l2))

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE#,
        # 4: Chem.BondType.AROMATIC
    }

    # print("TODO: AROMATIC bond type")
    # The problem is: Aromatic rings is getting BO of almost 1.5, instead of
    # 1/2. Will have to see how to account this
    #     58  C   3.983        C   60 1.459    C   56 1.406    H   59 0.967

    for i in range(l):
        for j in range(i + 1, l):
            bo = int(round(BO_matrix[i, j]))
            if (bo == 0):
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.AddBond(i, j, bt)

    mol = rwMol.GetMol()

    # if allow_charged_fragments:
    #     mol = set_atomic_charges(
    #         mol,
    #         atoms,
    #         atomic_valence_electrons,
    #         BO_valences,
    #         BO_matrix,
    #         mol_charge,
    #         use_atom_maps)
    # else:
    #     mol = set_atomic_radicals(mol, atoms, atomic_valence_electrons, BO_valences,
    #                                                         use_atom_maps)

    return mol



def BOfile_to_BOadj(BOfile): #
    BO = open(BOfile,"r").read().split('CSD_code = ')
    # print
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
    
    # print(list(bond_orders.items())[:1][0][1])
    # TODO: Check if it is loading all atoms or ignoring the last ones.
    # print(bond_orders)
    return bond_orders


def xyz_bo_2mol(BOmat, ACmat, atoms, coordinates, charge=0, allow_charged_fragments=True,
            use_graph=True, use_huckel=False, embed_chiral=True,
            use_atom_maps=False):
    """
    Generate a rdkit molobj from atoms, coordinates and a total_charge.

    args:
        atoms - list of atom types (int)
        coordinates - 3xN Cartesian coordinates
        charge - total charge of the system (default: 0)

    optional:
        allow_charged_fragments - alternatively radicals are made
        use_graph - use graph (networkx)
        use_huckel - Use Huckel method for atom connectivity prediction
        embed_chiral - embed chiral information to the molecule

    returns:
        mols - list of rdkit molobjects

    """

    # Get atom connectivity (AC) matrix, list of atomic numbers, molecular charge,
    # and mol object with no connectivity information
    AC_ignore, mol = xyz2AC(atoms, coordinates, charge, use_huckel=use_huckel)

    # Convert AC to bond order matrix and add connectivity and charge info to
    # mol object
    new_mols = AC2mol(mol, ACmat, BOmat, atoms, charge,
                     allow_charged_fragments=allow_charged_fragments,
                     use_graph=use_graph,
                     use_atom_maps=use_atom_maps)

    # Check for stereocenters and chiral centers
    # if embed_chiral:
    #     for new_mol in new_mols:
    #         chiral_stereo_check(new_mol)

    return new_mols


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(usage='%(prog)s [options] molecule.xyz')
    # parser.add_argument('structure', metavar='structure', type=str)
    parser.add_argument('-s', '--sdf',
        action="store_true",
        help="Dump sdf file")
    parser.add_argument('--ignore-chiral',
        action="store_true",
        help="Ignore chiral centers")
    parser.add_argument('--no-charged-fragments',
        action="store_true",
        help="Allow radicals to be made")
    parser.add_argument('--no-graph',
        action="store_true",
        help="Run xyz2mol without networkx dependencies")

    # huckel uses extended Huckel bond orders to locate bonds (requires RDKit 2019.9.1 or later)
    # otherwise van der Waals radii are used
    parser.add_argument('--use-huckel',
        action="store_true",
        help="Use Huckel method for atom connectivity")
    parser.add_argument('-o', '--output-format',
        action="store",
        type=str,
        help="Output format [smiles,sdf] (default=sdf)")
    parser.add_argument('-c', '--charge',
        action="store",
        metavar="int",
        type=int,
        help="Total charge of the system")

    args = parser.parse_args()

    # read xyz file
    # filename = args.structure

    # allow for charged fragments, alternatively radicals are made
    charged_fragments = not args.no_charged_fragments

    # quick is faster for large systems but requires networkx
    # if you don't want to install networkx set quick=False and
    # uncomment 'import networkx as nx' at the top of the file
    quick = not args.no_graph

    # chiral comment
    embed_chiral = not args.ignore_chiral

    # huckel uses extended Huckel bond orders to locate bonds (requires RDKit 2019.9.1 or later)
    # otherwise van der Waals radii are used
    # TODO: use van der waals as Biswajit sir was suggesting.
    use_huckel = args.use_huckel

    # if explicit charge from args, set it
    if args.charge is not None:
        charge = int(args.charge)








    # read atoms and coordinates. Try to find the charge
    data_path = '/home/shubodh/OneDrive/mll_projects/2022/xyz2mol/examples/tmQM/'
    full_bo_file = 'tmQM_X.BO'

    # all_ids = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15] #range(1,16) #[1, 2, 3] #
    all_ids = [2]

    # IMPORTANT NOTE (s) about tmQM dataset: (Modifications made etc)
    # 1. For tmQM_X1_Fe_mol_7.BO, removed last line since no of atoms was > 84 (i.e. was 85)
    # 2. Even after above, still not working, so skipping 7 in all_ids currently.
    # 

    for indi_id in all_ids:

        # indi_id = 4
        indi_name = 'tmQM_X1_Fe_mol_' + str(indi_id)
        fe_bo_file = indi_name + '.BO'
        fe_xyz_file = indi_name + '.xyz'
        # print("\n For file ", fe_xyz_file)

        BOfile = data_path + fe_bo_file
        xyz_filename = data_path + fe_xyz_file

        # # read atoms and coordinates. Try to find the charge
        # atoms, charge, xyz_coordinates = read_xyz_file(filename)

        atoms, charge, xyz_coordinates, num_atoms = read_xyz_file(xyz_filename)
        BOadj = BOfile_to_BOadj(BOfile)
        BOmat, ACmat = helg.BOadj_to_BOmat(BOadj, num_vertices=num_atoms)

        # Get the molobjs
        mols = xyz_bo_2mol(BOmat, ACmat, atoms, xyz_coordinates,
            charge=charge,
            use_graph=quick,
            allow_charged_fragments=charged_fragments,
            embed_chiral=embed_chiral,
            use_huckel=use_huckel)

        # Print output
        for mol in mols:
            # Canonical hack
            isomeric_smiles = not args.ignore_chiral
            smiles_original = Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles)

            #TODO: Sanitize molecule (make sure it is valid), see gvae_utils.py
            # print("TODO now: sanitize mol")
            # try:
                # Chem.SanitizeMol(mol)
            # except:
                # smiles = None




            m = Chem.MolFromSmiles(smiles_original)
            smiles_FromTo = Chem.MolToSmiles(m, isomericSmiles=isomeric_smiles)
            # print(f"isometric smiles {isomeric_smiles}")
            # print("Is smiles_FromTo==smiles_original? : " , smiles_FromTo==smiles_original)

            smiles_canon = Chem.CanonSmiles(smiles_original)
            m = Chem.MolFromSmiles(smiles_canon, sanitize=False)
            m2 = helg.set_dative_bonds(m)
            smiles_canon = Chem.CanonSmiles(Chem.MolToSmiles(m2))
            # print("Is smiles_canon == smiles_FromTo? : ", smiles_canon==smiles_FromTo)
            print('mol{}a=Chem.MolFromSmiles("{}")'.format(str(indi_id),smiles_canon))
            # print(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))

            # TODO: Check using networkx topology: see helg.topology_from_rdkit and helg.is_isomeric.