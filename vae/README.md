This folder `learn_mol/vae` trains various architectures (currently: graphVAE & DimeNet-modified) on QM9 and tmQM datasets.
 
Find instructions for conda environment setup/packages installation as well as how to run code below.


# 1. Installation
## Version currently using
* python 3.8
* pytorch 1.13.1
* cuda 11.6
* pyg 2.2.0 (pytorch geometric)

## Setup 
First create a conda environment. Then install pytorch & pytorch geometric as follows:

* ```conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia```   
(skipped torchaudio above as it is not necessary. Also, while on ADA, be careful with 25 GB home limit, these packages alone takes up like 5-10 GB.)
* ```conda install pyg -c pyg ```   (God alone knows how much more GB this will take)
Now small packages xyz2graph, rdkit etc:
* ```python -m pip install git+https://github.com/zotko/xyz2graph.git```
* ```pip install rdkit```
* ```pip install matplotlib pyyaml sympy```


# 2. Code Usage 
