import networkx as nx
import os
from urllib.parse import unquote  # for decoding Unicode characters such as 'É'


G = nx.DiGraph()

with open("links.tsv", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip() # remove whitespaces
        if not line or line.startswith("#"): # skip over comment lines or empty lines
            continue
        parts = line.split("\t") # links.tsv uses tab character as seperator
        if len(parts) == 2: # ensure each edge has only two parts (source --> destination)
            src, dst = [unquote(p).replace("_", " ") for p in parts]  # decode Unicode characters and replace '_' with ' '
            G.add_edge(src, dst)

# output num of nodes and edges for tracking
numNodes = nx.number_of_nodes(G)
print(f"Number of Nodes: {numNodes}")

numEdges = nx.number_of_edges(G)
print(f"Number of Nodes: {numEdges}")

graphFolder = "Graphs"
graphOutputFile = "linkGraph.graphml"
pathToOutput = os.path.join(graphFolder, graphOutputFile)

nx.write_graphml(G, pathToOutput)
