import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy
import heapq
import time

#reading and outputting number of nodes and edges for checking
G = nx.read_graphml("Graphs/linkGraph.graphml")

# Getting PageRank
# Using alpha value of 0.85 (that is the value used by Google)
pagerank_scores = nx.pagerank(G, alpha=0.85)

# normalize pagerank to [0,1]
min_pr = min(pagerank_scores.values())
max_pr = max(pagerank_scores.values())
if max_pr - min_pr > 0:
    pr_norm = {n: (pagerank_scores[n] - min_pr) / (max_pr - min_pr) for n in G.nodes()}
else:
    pr_norm = {n: 0.0 for n in G.nodes()}

# === Utility: reconstruct path from prev dict ===
def _reconstruct(prev, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path

def astar_with_pagerank(G, start, goal, beta=5.0):
    """
    A* variant using a PageRank-derived heuristic:
      h(n) = beta * (1 - pr_norm[n])
    This biases the search toward high-PageRank nodes (low h).
    beta controls how strongly the heuristic influences search.
    If beta == 0, reduces to Dijkstra.
    Note: This heuristic is not provably admissible; use with that caveat.
    Returns (path, nodes_expanded, time_seconds).
    """
    t0 = time.perf_counter()
    def h(node):
        # lower h for higher PageRank (we want to prefer high-PR nodes)
        return beta * (1.0 - pr_norm.get(node, 0.0))

    frontier = [(h(start), 0, start)]  # (f, g, node)
    gscore = {start: 0}
    prev = {start: None}
    nodes_expanded = 0

    while frontier:
        f, gcur, node = heapq.heappop(frontier)
        # follow typical A* behavior: skip if g is stale
        if gcur != gscore.get(node, float('inf')):
            continue
        if node == goal:
            t = time.perf_counter() - t0
            return _reconstruct(prev, goal), nodes_expanded, t
        nodes_expanded += 1
        for nbr in G.neighbors(node):
            tentative_g = gcur + 1
            if tentative_g < gscore.get(nbr, float('inf')):
                gscore[nbr] = tentative_g
                prev[nbr] = node
                heapq.heappush(frontier, (tentative_g + h(nbr), tentative_g, nbr))
    t = time.perf_counter() - t0
    return None, nodes_expanded, t

# === Example usage snippet ===
if __name__ == "__main__":
    # replace these with two real node IDs from your graph
    start_node = "Smallpox"
    goal_node = "Socialism"

    print("\nRunning PR-informed A* (beta=5.0)...")
    path_pr, expanded_pr, t_pr = astar_with_pagerank(G, start_node, goal_node, beta=5.0)
    print("Path length:", len(path_pr) if path_pr else None, "Nodes expanded:", expanded_pr, "Time:", t_pr)
    print(path_pr)