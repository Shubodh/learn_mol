import pybel
from rdkit import Chem
import helper_graph as helg #import BOadj_to_BOmat

def xyz_to_smiles(fname: str) -> str:
    mol = next(pybel.readfile("xyz", fname))
    smi = mol.write(format="smi")
    return smi.split()[0].strip()


# def smiles_to_xyz(smi_str):
#     mymol_smi = pybel.readstring("smi", smi_str)
#     xyz_out = mymol_smi.write(format="xyz")
#     print(xyz_out)
    # return xyz_out

if __name__ == "__main__":
    data_path = '/home/shubodh/OneDrive/mll_projects/2022/xyz2mol/examples/tmQM/'
    full_bo_file = 'tmQM_X.BO'

    all_ids = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15] #range(1,16) #[1, 2, 3] #
    # 8, 9 issue with obabel
    # all_ids = [3]

    for indi_id in all_ids:
        # indi_id = 3
        indi_name = 'tmQM_X1_Fe_mol_' + str(indi_id)
        fe_bo_file = indi_name + '.BO'
        fe_xyz_file = indi_name + '.xyz'

        BOfile = data_path + fe_bo_file
        xyz_filename = data_path + fe_xyz_file

        smi = xyz_to_smiles(xyz_filename)
        # smiles_FromTo = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        # print(indi_id)
        smiles_canon = Chem.CanonSmiles(smi)
        m = Chem.MolFromSmiles(smiles_canon, sanitize=False)
        m2 = helg.set_dative_bonds(m)
        smiles_canon = Chem.CanonSmiles(Chem.MolToSmiles(m2))
        # print(smi)
        # print(smiles_canon)
        # print("\n")
        print('mol{}b=Chem.MolFromSmiles("{}")'.format(str(indi_id),smiles_canon))
        # print(type(smi))
        # print(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))
        # print(Chem.CanonSmiles("OC[Fe](P)(CO)(CO)CO"))
        # print(smi==)