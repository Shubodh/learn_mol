# Installation

Main packages needed are rdkit, OpenBabel. Other standard packages are NumPy, pandas, jupyter etc.

Be careful with OpenBabel installation. Use `conda install openbabel -c openbabel -y` for python package installation. (instead of `conda install openbabel -c conda-forge`, this is not working)

The environment file `xyz2smiles_environment.yml` has been provided for conda, for example, you can create new env with this using `conda env create -f environment.yml`. **However, it is recommended you install packages one by one** instead of using environment file directly, since it is not clean in current state. I will clean up the file eventually when you can use this method.


# Usage

Just run `python xyz2smiles_using_openbabel.py` for xyz to smiles using OpenBABEL or `python xyz2mol_from_BOfile_metal_chelate.py` for xyz to smiles using rdkit. It will output SMILES strings for 15 molecules in `learn_mol/xyz2smiles/examples/tmQM/` folder.

Note that the code is not yet currently clean. I have not removed unnecessary code as there are some bugs (=O instead of OH, small c's, disconnected structures etc), i.e. "unnecessary code" might be necessary when I am fixing the above bugs.



# References

1. [xyz2mol](https://github.com/jensengroup/xyz2mol): xyz to rdkit mol objects for organic molecules.  