import networkx as nx
import itertools
import numpy as np
from tqdm import tqdm
import h5py
import time
import scipy
import argparse
from networkx.algorithms.centrality import degree_centrality, betweenness_centrality, closeness_centrality
from networkx.algorithms.cluster import clustering
from networkx.algorithms.distance_measures import diameter, radius
from networkx.algorithms.shortest_paths.generic import shortest_path_length

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

def get_features(a):
	return np.array([np.max(a), np.min(a), np.mean(a), np.std(a)])

def get_topological_features(G, nodes=None):
	N_ = len(G.nodes)
	if nodes is None:
		nodes = G.nodes
	# Degree centrality
	d_c = get_features(degree_centrality(G).values())
	print 'a'
	# Betweeness centrality
	b_c = get_features(betweenness_centrality(G).values())
	print 'b'

	# Close ness centrality
	c_c = get_features(closeness_centrality(G).values())
	print 'c'
	# Clustering
	c = get_features(clustering(G).values())
	print 'd'

	d = diameter(G)
	r = radius(G)

	s_p_average = []
	for s in shortest_path_length(G):
		dic = s[1]
		lengths = dic.values()
		s_p_average += [sum(lengths)/float(N_)]

	s_p_average = get_features(s_p_average)

	features = np.concatenate((d_c, b_c, c_c, c, s_p_average, [d], [r]), axis=0)

	return features

V, N, K = get_args()

seed = 3
np.random.seed(seed)

name = "clique-N{}-K{}".format(V, K)
name += 'test'
with h5py.File('data/'+name+'.h5', 'w') as h5_file:

	feature_shape = (2*N, 22)
	
	labels_shape = (2*N,)
	features_df = h5_file.create_dataset('features', shape=feature_shape)
	labels_df = h5_file.create_dataset('labels', shape=labels_shape)

	data = []
	labels = []
	for n in tqdm(range(2*N)):
		g = nx.erdos_renyi_graph(V, 0.5, seed=seed*n)
		if n < N:
			labels_df[n] = 0
		else:
			add_random_k_clique(g, K) # Add K clique in the graph
			labels_df[n] = 1
		
		features_df[n] = get_topological_features(g)

