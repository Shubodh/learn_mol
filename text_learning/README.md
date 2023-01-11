
# 1. xyz2smiles
Text/NLP based learning for molecules, i.e. input will be SMILES / SELFIES.
See `./xyz2smiles` folder.
Currently simple RNN, not using any GNN. But will do so in future.

Put this on pause currently for many issues xyz2smiles was giving (see notion).

# 2. MORE:
Also using this folder `text_learning` for other misc tasks for Shubodh. Such as
* `bfs_mol_graph.py`: Take a simple molecular graph and order the atoms based on BFS (breadth first search) (AND more like Dijkstra, Cartesian ordering etc) and then replot the graph with atom labels and index. DO NOTE: It is not just BFS but also has all kinds of ordering done in that .py file.
