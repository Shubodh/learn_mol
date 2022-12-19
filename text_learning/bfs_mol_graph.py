import networkx as nx
import matplotlib.pyplot as plt

import sys
import os

from module_temp import featurize

from rdkit import Chem

# Write a function which saves matplotlib figure to a file
def save_fig(path, fig_id, tight_layout=True, fig_extension="png", resolution=300):
	path = os.path.join(path, fig_id + "." + fig_extension)
	print("Saving figure", fig_id)
	if tight_layout:
		plt.tight_layout()
	plt.savefig(path, format=fig_extension, dpi=resolution)

def example_code_1():
	"""
	This example 1 is a generic graph example for BFS using NetworkX library.
	We create a simple graph with two branches of tree and then use BFS with source "d" to order the nodes.
	"""
	# Construct simple graph with two branches of tree
	H = nx.Graph()
	nx.add_path(H, ["a", "b", "c", "d", "e", "f", "g"])
	nx.add_path(H, ["c", "h", "i", "j", "k"])
	# Better to use strings as node names instead of integers. Because integers can be confused with index/labels.
	# nx.add_path(H, [0, 1, 2, 3, 4, 5, 6])
	# nx.add_path(H, [2, 7, 8, 9, 10])

	original_order = list(H.nodes())
	original_labels = {}

	# Here we are going to set the original order of the nodes as labels.
	for i, node_name in enumerate(original_order):
		original_labels[node_name] = i 
	nx.set_node_attributes(H, original_labels, 'label')

	plt.figure()
	nx.draw(H, with_labels=True, font_weight='bold', node_color="lightpink",
			labels={node: f"{node} ({data['label']}) " for node, data in H.nodes(data=True)})
	plt.title("Original Graph")


	# Here, we are going to use BFS to order the nodes and set new order as labels.
	order = list(nx.bfs_tree(H, source="d").nodes())
	print("Put pen on paper and work it out. Our source/root is d")
	print(f"original order {original_order}")
	bfs_order_a_source = list(nx.bfs_tree(H, source="a").nodes())
	print(f"BFS Order with \"a\" source: {bfs_order_a_source}") #NodeView((0, 1, 2, 3, 7, 4, 8, 5, 9, 6, 10))
	print(f"BFS Order with \"d\" source: {order}") #NodeView((3, 2, 4, 1, 7, 5, 0, 8, 6, 9, 10))

	H_bfs_source_d = nx.Graph()
	# create the BFS graph, first nodes then edges.
	for i, node_name in enumerate(order):
		H_bfs_source_d.add_node(node_name, label=i)
	for u, v, data in H.edges(data=True):
		H_bfs_source_d.add_edge(u, v)


	plt.figure()
	nx.draw(H_bfs_source_d, with_labels=True, font_weight='bold', node_color="yellow",
			labels={node: f"{node} ({data['label']}) " for node, data in H_bfs_source_d.nodes(data=True)})
	plt.title("BFS Graph")
	plt.show()

def make_graph_simpler_for_bfs_viz(graph_nx):
	"""
	`graph_nx` is a original graph in networkx format.
	Output is a simpler graph with lesser information, but it's the same graph structure.
	"""
	simpler_graph = nx.Graph()

	order = list(graph_nx.nodes)
	for i, node_name in enumerate(order):
		atomic_symbol = PT.GetElementSymbol(graph_nx.nodes[node_name]['x'][0]) #graph_nx.nodes[id_node]['x'][0] is atomic number.
		label_g = str(atomic_symbol) + ":o" + str(i)
		simpler_graph.add_node(node_name, label=label_g)
	
	for u, v, data in graph_nx.edges(data=True):
		simpler_graph.add_edge(u, v)
	return simpler_graph


def change_graph_order_nx(graph_nx, root_node_id, order_type="bfs"):
	"""
	This function changes the order of nodes in a graph (BFS or DFS) and returns the new graph.
	`graph_nx` is a original graph in networkx format.
	`order_type` is either "bfs" or "dfs" .
	"""
	if order_type == "bfs":
		order = list(nx.bfs_tree(graph_nx, root_node_id).nodes)
	elif order_type == "dfs":
		order = list(nx.dfs_tree(graph_nx, root_node_id).nodes)
	else:
		raise ValueError("order_type must be either 'bfs' or 'dfs'")

	new_graph = nx.Graph()


	# # Add the atoms to the new graph in the order determined by the BFS
	for i, node_name in enumerate(order):
		# atomic_symbol = PT.GetElementSymbol(graph_nx.nodes[i]['x'][0]) #graph_nx.nodes[id_node]['x'][0] is atomic number.
		atomic_symbol = PT.GetElementSymbol(graph_nx.nodes[node_name]['x'][0]) #graph_nx.nodes[id_node]['x'][0] is atomic number.
		label_g = str(atomic_symbol) + ":o" + str(i)
		new_graph.add_node(node_name, label=label_g)
	
	for u, v, data in graph_nx.edges(data=True):
		new_graph.add_edge(u, v)
	return new_graph


if __name__ == '__main__':
	PT = Chem.GetPeriodicTable()

	testing =False #True  # 

	if testing:
		example_code_1()
		sys.exit()

	data_xyz = '../data/tmQM_uiocompcat/tmQM_X.xyz'
	charges = '../data/tmQM_uiocompcat/tmQM_X.q'
	BO = '../data/tmQM_uiocompcat/tmQM_X.BO'
	graphs, csd_codes = featurize(data_xyz, charges, BO)

	# for G in graphs:
	# 	print(len(G.nodes))

	graphs_few = graphs[0:10]

	# iterate two lists at the same time and also keep track of the index
	for i, (G, csd_code) in enumerate(zip(graphs_few, csd_codes)):
		for id_node in G.nodes:
			atomic_no = (G.nodes[id_node]['x'][0])
			atomic_symbol = PT.GetElementSymbol(atomic_no)
			if atomic_symbol == "Fe":
				root_node_id = id_node 
				root_node_info = {"atomic_no": atomic_no, "atomic_symbol": atomic_symbol}

		simpler_G = make_graph_simpler_for_bfs_viz(G)
		graph_BFS = change_graph_order_nx(G, root_node_id, order_type="bfs")
		graph_DFS = change_graph_order_nx(G, root_node_id, order_type="dfs")

		# Plot the original graph
		# plt.figure()
		# nx.draw(G, with_labels=True, font_weight='bold', node_color='yellow')#,
		# plt.title("Original Graph")

		save_path = "./bfs_viz/"
		prefix = str(i) + "_" + csd_code + "_"
		# Plot the original simpler graph
		plt.figure()
		nx.draw(simpler_G, with_labels=True, font_weight='bold', node_color='yellow',
				labels={node: f"{node} ({data['label']}) " for node, data in simpler_G.nodes(data=True)})
		plt.title("Original simpler Graph")
		save_fig(save_path, prefix + "ariginal_simpler_graph")

		# Plot the BFS-ordered graph
		plt.figure()
		nx.draw(graph_BFS, with_labels=True, font_weight='bold', node_color='lightgreen', 
				labels={node: f"{node} ({data['label']}) " for node, data in graph_BFS.nodes(data=True)})
		plt.title("BFS-Ordered Graph")
		save_fig(save_path, prefix + "bfs_ordered_graph")

		# Plot the DFS-ordered graph
		plt.figure()
		nx.draw(graph_DFS, with_labels=True, font_weight='bold', node_color='lightpink',
				labels={node: f"{node} ({data['label']}) " for node, data in graph_DFS.nodes(data=True)})
		plt.title("DFS-Ordered Graph")
		save_fig(save_path, prefix + "dfs_ordered_graph")

		# plt.show()
