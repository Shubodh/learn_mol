import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jax.numpy as jnp
import jax
from jax.example_libraries import optimizers
import sklearn.manifold, sklearn.cluster
import rdkit, rdkit.Chem, rdkit.Chem.Draw

soldata = pd.read_csv('/home/saishubodh/2022/learn_mol/data/tmQM/tmQM_y.csv.gz')
print(soldata.head())
