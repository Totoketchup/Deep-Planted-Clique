import networkx as nx
import itertools
import numpy as np
from tqdm import tqdm

def add_random_k_clique(G, k):
	L = np.random.choice(len(G.nodes), k)
	edges = itertools.combinations(L, 2)
	G.add_edges_from(edges)

def get_topk(G, k):
	degrees = G.degree(G.nodes)
	top_k = sorted(degrees, key = lambda x : x[1], reverse=True)[:k]
	top_k_nodes = [t[0] for t in top_k]
	extension = []
	for top_node in top_k_nodes:
		neighbors = [n for n in G.neighbors(top_node)]
		deg = G.degree(neighbors)
		top_extend = sorted(deg, key = lambda x : x[1], reverse=True)[:extend]
		top_extend = [t[1] for t in top_extend]
		extension += [top_extend]
	extension = np.array(extension)
	top_k = [t[1] for t in top_k]
	top_k = np.array(top_k)[:, np.newaxis]
	a = np.concatenate((top_k, extension),1)
	return a


N = 10000
K = 31
nodes = 1000
extend = 50

name = "clique-N{}-K{}-E{}".format(nodes, K, extend)

data = []
labels = []
for n in tqdm(range(2*N)):

	g = nx.erdos_renyi_graph(nodes, 0.5)
	if n < N:
		g.graph['label'] = 0
		labels += [0]
	else:
		add_random_k_clique(g, K)
		g.graph['label'] = 1
		labels += [1]

	data += [get_topk(g, K)]

data = np.array(data)
labels = np.array(labels)
print labels.shape
print data.shape
np.save(name+'-features',data)
np.save(name+'-labels',labels)
