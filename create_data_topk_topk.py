import networkx as nx
import itertools
import numpy as np
from tqdm import tqdm
import h5py
import time
import scipy
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
    parser.add_argument(
        '-e', '--extension', type=int, help='Number of extended top nodes', required=False, default=0)
    parser.add_argument(
        '-ex', '--exclude', type=bool, help='Exclude top k nodes for extension', required=False, default=True)
    parser.add_argument(
        '-m', '--multiple', type=int, help='Analyze the top m*k nodes', required=False, default=1)
    parser.add_argument(
        '-l', '--laplacian', type=bool, help='Analyze Laplacian', required=False, default=True)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    server = args.server
    port = args.port[0].split(",")
    keyword = args.keyword
    # Return all variable values
    return server, port, keyword

def add_random_k_clique(G, k):
	L = np.random.choice(len(G.nodes), k, replace=False)
	edges = itertools.combinations(L, 2)
	G.add_edges_from(edges)

def get_topk(G, k, E, P, ex, L):

	degrees = G.degree(G.nodes)

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

	G_ = G
	# Deletion or merging of N - k nodes
	if L == 'Deletion':
		laplacian = nx.laplacian_matrix(G_, list(top_k_nodes)).toarray()
		eigenvalues, eigenvectors = np.linalg.eig(laplacian)
		# Sort according to eigenvalues
		second = eigenvalues.argsort()[-2] # Second eigenvalue
		connectivity = eigenvectors[:, second]
		output = np.concatenate((output, np.abs(connectivity[:, np.newaxis])), 1)
	# elif L == 'Merge':
	# 	for i in range(len(G_.nodes)-k-40):
	# 		deg = G_.degree(G_.nodes)
	# 		mins = sorted(deg, key = lambda x : x[1], reverse=False)
	# 		mins_nodes = zip(*mins)[0]
	# 		checking_nodes = mins_nodes[:10]
	# 		m = 5
	# 		for u,v in itertools.combinations(checking_nodes,2):
	# 				u_d = G_.degree(u)
	# 				v_d = G_.degree(v)
	# 				s = u_d + v_d
	# 				if G_.has_edge(u,v):
	# 					if s < m:
	# 						min_tuple = (u,v) if u_d >= v_d else (v,u)
	# 						m = s
	# 		print G_.degree(G_.nodes)
	# 		G_ = nx.contracted_edge(G_, min_tuple, self_loops=False)
			
	# 	initial_degree = G.degree(G_.nodes)
	# 	laplacian = nx.laplacian_matrix(G_).toarray()
	# 	eigenvalues, eigenvectors = np.linalg.eig(laplacian)

	# 	idx = eigenvalues.argsort()   
	# 	eigenvalues = eigenvalues[idx]
	# 	eigenvectors = eigenvectors[:,idx]
	# 	print eigenvectors
	# 	print eigenvalues
	# 	print len(G_.nodes)
	# 	print G_.degree(G_.nodes)
	# 	# print laplacian

	# 	output = np.concatenate((output, laplacian), 1)

	return output

# Number of elements
N = 5000
# Size of clique
K = 10
# Number of nodes
nodes = 100
# Number of additional top E
E = 0
# Top M*K
M = 2
#
L = 'Deletion'#'Merge'
# Exclude top k nodes when searching the top E neighbors
exclude = True

seed = 3
np.random.seed(seed)

if L is None:
	name = "clique-N{}-K{}-E{}-M{}-P{}-ex{}".format(nodes, K, E, M, P, exclude)
else:
	name = "clique-N{}-K{}-E{}-M{}-P{}-ex{}-Lapl{}".format(nodes, K, E, M, P, exclude, L)
# name += 'test'
with h5py.File('data/'+name+'.h5', 'w') as h5_file:
	if L is None:
		dim = E+1
	else: 
		dim = E+1+1
	feature_shape = (2*N, M*K, dim)
	labels_shape = (2*N,)
	features_df = h5_file.create_dataset('features', shape=feature_shape, compression="lzf")
	labels_df = h5_file.create_dataset('labels', shape=labels_shape, compression="lzf")

	data = []
	labels = []
	for n in tqdm(range(2*N)):
		g = nx.erdos_renyi_graph(nodes, 0.5, seed=seed*n)
		if n < N:
			labels_df[n] = 0
		else:
			add_random_k_clique(g, K) # Add K clique in the graph
			labels_df[n] = 1

		features_df[n] = get_topk(g, M*K, E, P, exclude, L)

