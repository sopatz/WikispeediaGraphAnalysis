import networkx as nx
import numpy
import scipy

#reading and outputting number of nodes and edges for checking
G = nx.read_graphml("Graphs/linkGraph.graphml")

numNodes = nx.number_of_nodes(G)
print(f"Number of Nodes: {numNodes}")

numEdges = nx.number_of_edges(G)
print(f"Number of Edges: {numEdges}")

#getting pagerank
#using alpha as 0.85 (that is the value used by google)
pr_scores = nx.pagerank(G, alpha=0.85)

print(f"PageRank results: {pr_scores}")