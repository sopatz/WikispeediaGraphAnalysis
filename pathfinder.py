import networkx as nx
import heapq
import time

#reading and outputting number of nodes and edges for checking
graph = nx.read_graphml("Graphs/linkGraph.graphml")

# Getting PageRank
# Using alpha value of 0.85 (that is the value used by Google)
pagerank_scores = nx.pagerank(graph, alpha=0.85)

# Normalize pagerank to [0,1]
min_pr = min(pagerank_scores.values())
max_pr = max(pagerank_scores.values())
# 0 = lowest original PageRank, 1 = highest original PageRank
normalized_pagerank = {n: 0.0 for n in graph.nodes()}

# Helper function to reconstruct path from predecessor dictionary
def reconstruct(prev_node, goal_node):
    path = []
    current_node = goal_node
    # Start at goal node and repeatedly follow previous node back to the start of the path
    # prev[start] should have None as its value, making this our stopping case
    while current_node is not None:
        path.append(current_node)
        current_node = prev_node.get(current_node)
    # Since path is reconstructed from goal node to start node, it needs to be reversed to get the real path
    path.reverse()
    return path

def astar_with_pagerank(graph, start, goal, beta=5.0):
    """
    A* variant using a PageRank-derived heuristic:
      h(n) = beta * (1 - pr_norm[n])
    This biases the search toward high-PageRank nodes (low h).
    beta controls how strongly the heuristic influences search.
    If beta == 0, reduces to Dijkstra.
    Note: This heuristic is not provably admissible; use with that caveat.
    Returns (path, nodes_expanded, time_seconds).
    """
    initial_time = time.perf_counter()
    def heuristic(node):
        # Lower value returned for higher PageRank (we want to prefer high-PR nodes)
        # beta scales the magnitude of the heuristic, beta = 0 turns off the heuristic
        # If node is not found in pagerank, its pagerank value defaults to 0.0
        return beta * (1.0 - normalized_pagerank.get(node, 0.0))

    # Initialize heap with single tuple in it
    # Each tuple contains: heuristic value of node, path cost from start node to current node (each edge has cost of 1), node itself
    frontier = [(heuristic(start), 0, start)]

    # "bestcost" holds the best known cost (distance) from start node to each node discovered so far
    bestcost = {start: 0}

    # "prev" will hold predecessor map for each discovered node (all nodes travelled to to get to the current node and their parent)
    # start node has no predecessor, so it is mapped to "None" for its parent node
    prev = {start: None}

    # To count the number of nodes expanded from the frontier (for performance benchmark)
    nodes_expanded = 0

    # Run until frontier heap of nodes is empty
    while frontier:
        # Pop node with smallest priority value
        priority, bestcost_current, node = heapq.heappop(frontier)

        # Ignore entries whose bestcost_current does not match the best cost for that node (because heap can contain older/stale entries for the same node)
        if bestcost_current != bestcost.get(node, float('inf')):
            continue

        # If the goal node has been popped, record the elapsed time and reconstruct the path taken to get there
        if node == goal:
            elapsed_time = time.perf_counter() - initial_time
            return reconstruct(prev, goal), nodes_expanded, elapsed_time
        
        # Increment expansion count because we are about to expand the current node's neighbors
        nodes_expanded += 1

        # Iterate over all neighbors of the current node
        for neighbor in graph.neighbors(node):
            # All edges travelled have cost of 1, so add one to the tentative best score
            tentative_bestcost = bestcost_current + 1

            # Check if the tentative best cost is better than the previously-known best cost for that node
            # If node does not have a best cost value yet, it defaults to infinity so any distance is an improvement
            if tentative_bestcost < bestcost.get(neighbor, float('inf')):
                # Update best cost of current neighbor to the improved value
                bestcost[neighbor] = tentative_bestcost

                # Set neighbor's previous node for when it is added to heap
                prev[neighbor] = node

                # Add neighbor node to heap 
                # Small heuristic value (large PageRank) = lower priority value, meaning it will be popped first
                heapq.heappush(frontier, (tentative_bestcost + heuristic(neighbor), tentative_bestcost, neighbor))

    # Frontier empties without reaching the goal --> return no path     
    elapsed_time = time.perf_counter() - initial_time
    return None, nodes_expanded, elapsed_time

# Example usage of A* search with PageRank heuristic
if __name__ == "__main__":
    # Specify start and goal webpage from graph
    start_page = "Beer"
    goal_page = "Typewriter"

    print(f"Path from {start_page} to {goal_page}: ")
    path, nodes_expanded, elapsed_time = astar_with_pagerank(graph, start_page, goal_page, beta=5.0)
    print(f"Path: {path}")
    print(f"Path length: {len(path)}")
    print(f"Nodes expanded: {nodes_expanded}")
    print(f"Time elapsed: {elapsed_time:.6f} seconds")