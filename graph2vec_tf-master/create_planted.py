import networkx as nx
import itertools
import numpy as np
from tqdm import tqdm
import os

def add_random_k_clique(G, k):
	L = np.random.choice(len(G.nodes), k)
	edges = itertools.combinations(L, 2)
	G.add_edges_from(edges)

N = 5000
K = 10
nodes = 100
name = "clique-N{}-K{}".format(nodes,K)

file = open(name+".Labels", "w") 

if not os.path.exists(name):
    os.makedirs(name)

for n in tqdm(range(2*N)):

	g = nx.erdos_renyi_graph(nodes, 0.5)
	if n < N:
		g.graph['label'] = 0
	else:
		add_random_k_clique(g, K)
		g.graph['label'] = 1
	nx.write_gexf(g, name+"/{}.gexf".format(n))
	file.write("{}.gexf {}\n".format(n, g.graph['label']))

file.close() 

