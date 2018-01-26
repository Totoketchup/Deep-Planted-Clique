import networkx as nx
import itertools
import numpy as np
from tqdm import tqdm
import h5py
import argparse

def get_args():
	parser = argparse.ArgumentParser(description='Create Planted Clique Dataset')
	# Add arguments
	parser.add_argument(
		'-v', '--nodes', type=int, help='Number of nodes', required=True)
	parser.add_argument(
		'-n', '--elements', type=int, help='Number of graphs generated', required=True)
	parser.add_argument(
		'-k', '--clique', type=int, help='Clique size', required=True)
	args = parser.parse_args()
	return args.nodes, args.elements, args.clique
	
def add_random_k_clique(G, k):
	L = np.random.choice(len(G.nodes), k, replace=False)
	edges = itertools.combinations(L, 2)
	G.add_edges_from(edges)

V, N, K = get_args()

seed = 3
np.random.seed(seed)

name = "adjacency-N{}-K{}".format(V, K)

with h5py.File('data/'+name+'.h5', 'w') as h5_file:

	feature_shape = (2*N, V, V)
	
	labels_shape = (2*N,)
	features_df = h5_file.create_dataset('features', shape=feature_shape)
	labels_df = h5_file.create_dataset('labels', shape=labels_shape)

	data = []
	labels = []
	for n in tqdm(range(2*N)):
		g = nx.erdos_renyi_graph(V, 0.5, seed=seed*n)
		print list(g.edges)

		if n < N:
			labels_df[n] = 0
		else:
			add_random_k_clique(g, K) # Add K clique in the graph
			labels_df[n] = 1
		
		features_df[n] = nx.adjacency_matrix(g).todense()

