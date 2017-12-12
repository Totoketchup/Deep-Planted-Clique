import networkx as nx
import itertools
import numpy as np
from tqdm import tqdm
import os

def add_random_k_clique(G, k):
	L = np.random.choice(len(G.nodes), k)
	edges = itertools.combinations(L, 2)
	G.add_edges_from(edges)

def get_topk(G, k):
	degrees = G.degree(G.nodes)
	top_k = sorted(degrees, key = lambda x : x[1], reverse=True)
	top_k = [t[1] for t in top_k]
	return top_k

k = 10

N = 100000
K = 10
nodes = 100

name = "clique-N{}-K{}-k{}".format(nodes, K, k)

file_label = open(name+"labels.txt", "w") 
file_feat = open(name+"features.txt", "w") 


for n in tqdm(range(2*N)):

	g = nx.erdos_renyi_graph(nodes, 0.5)
	if n < N:
		g.graph['label'] = 0
	else:
		add_random_k_clique(g, K)
		g.graph['label'] = 1

	item = ",".join(map(str, get_topk(g, k)))
	file_feat.write(item)
	file_label.write(str(g.graph['label'])+' ')



file.close() 