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
from joblib import Parallel, delayed
from multiprocessing import Pool
import multiprocessing
    

def get_args():
	parser = argparse.ArgumentParser(description='Create Planted Clique Dataset')
	# Add arguments
	parser.add_argument(
		'-v', '--nodes', type=int, help='Number of nodes', required=True)
	parser.add_argument(
		'-n', '--elements', type=int, help='Number of graphs generated', required=True)
	parser.add_argument(
		'-k', '--clique', type=int, help='Clique size', required=True)
	parser.add_argument(
		'-e', '--extension', type=int, help='Number of extended top nodes', required=False, default=0)
	parser.add_argument(
		'-ex', '--exclude', type=int, help='Exclude top k nodes for extension', required=False, default=1)
	parser.add_argument(
		'-m', '--multiple', type=int, help='Analyze the top m*k nodes', required=False, default=1)
	parser.add_argument(
		'-l', '--laplacian', type=int, help='Use Laplacian eigenvectors for top K top E', required=False, default=0)
	parser.add_argument(
		'-a', '--aligned', type=int, help='1D data', required=False, default=0)
	parser.add_argument(
		'-topo', '--topological', type=int, help='Use topological features', required=False, default=0)

	args = parser.parse_args()
	return args.nodes, args.elements, args.clique,args.extension, args.exclude, args.multiple, args.laplacian, args.aligned, args.topological

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
	# Betweeness centrality
	b_c = get_features(betweenness_centrality(G).values())
	# Close ness centrality
	c_c = get_features(closeness_centrality(G).values())
	# Clustering
	c = get_features(clustering(G).values())

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

def get_topk(G, k, E, ex, L, fl):

	degrees = G.degree(G.nodes)

	# Deletion or merging of N - k nodes
	if L:
		laplacian = nx.laplacian_matrix(G).toarray()
		eigenvalues, eigenvectors = np.linalg.eig(laplacian)
		# Sort according to eigenvalues
		second = eigenvalues.argsort()[-2] # Second eigenvalue
		connectivity = np.abs(eigenvectors[:, second]) # Second eigenvector

		top_connectivity = connectivity.argsort()[::-1] # decreasing order nodes
		top_k_nodes = top_connectivity[:k]

		_, deg = zip(*degrees)
		deg = np.array(deg)
		top_k_degree = deg[top_k_nodes][:, np.newaxis]
	else:
		top = sorted(degrees, key = lambda x : x[1], reverse=True)
		top_k = top[:k] # Get the top k nodes
		top_k_nodes, top_k_degree = zip(*top_k)
		top_k_degree = np.array(top_k_degree)[:, np.newaxis]

	# Add top E nodes for each top k nodes
	if E > 0:
		extension = []
		for top_node in top_k_nodes:
			neighbors = [n for n in G.neighbors(top_node)] # get neighbors of the top k nodes
			# Exclude the top k nodes
			if ex:
				neighbors_set = set(neighbors) # create set of neighbors -> easy to remove elements
				neighbors_set = neighbors_set - set(top_k_nodes)
				neighbors = list(neighbors_set)	
			deg = G.degree(neighbors)
			top_e = sorted(deg, key = lambda x : x[1], reverse=True)[:E]
			top_e = [t[1] for t in top_e]
			extension += [top_e]
		extension = np.array(extension)
		output = np.concatenate((top_k_degree, extension), 1)
	else:
		output = top_k_degree

	output = np.array(output)

	if fl:
		output = np.reshape(np.transpose(output), -1)

	return output

V, N, K, E, exclude, M, L, flatten, topo = get_args()
exclude = bool(exclude)
L = bool(L)
flatten = bool(flatten)
topo = bool(topo)

seed = 3
np.random.seed(seed)

inputs = range(2*N) 
def processInput(i):
	g = nx.erdos_renyi_graph(V, 0.5, seed=seed*i)
	if i < N:
		label = 0
	else:
		add_random_k_clique(g, K) # Add K clique in the graph
		label = 1
	feature = get_topk(g, M*K, E, exclude, L, flatten)
	return feature, label

num_cores = multiprocessing.cpu_count()
pool = Pool(processes=6)
results = pool.map(processInput, range(2*N))
pool.close()
pool.join()
# results = Parallel(n_jobs=5)(processInput(i) for i in inputs)
print 'HA'
print results
features, labels = zip(*results)

print 'OH'
name = "clique-N{}-K{}-E{}-M{}-ex{}-L:{}-F:{}".format(V, K, E, M, exclude, L, flatten)
with h5py.File('data/'+name+'.h5', 'w') as h5_file:

	dim = E+1
	

	if not flatten:
		feature_shape = (2*N, M*K, dim)
	else:
		feature_shape = (2*N, M*K*dim)

	labels_shape = (2*N,)
	features_df = h5_file.create_dataset('features', shape=feature_shape)
	labels_df = h5_file.create_dataset('labels', shape=labels_shape)
	features_df[...] = features
	labels_df[...] = labels

 	# data = []
	# labels = []
	# for n in tqdm(range(2*N)):
	# 	g = nx.erdos_renyi_graph(V, 0.5, seed=seed*n)
	# 	if n < N:
	# 		labels_df[n] = 0
	# 	else:
	# 		add_random_k_clique(g, K) # Add K clique in the graph
	# 		labels_df[n] = 1
	# 	features_df[n] = get_topk(g, M*K, E, exclude, L, flatten)
	# 	print feature_shape
