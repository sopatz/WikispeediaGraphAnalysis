import networkx as nx
import heapq
import time

# Create graph using networkx
graph = nx.read_graphml("Graphs/linkGraph.graphml")

# Get the PageRank scores of all nodes in the graph
# Using alpha value of 0.85 (that is the value used by Google)
pagerank_scores = nx.pagerank(graph, alpha=0.85)

# Normalize pagerank values to [0,1]
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


# Function that implements the A* shortest path algorithm with a PageRank-informed heuristic
def astar_with_pagerank(graph, start, goal, beta=5.0):
    """
    A* variant using a PageRank-derived heuristic:
      heuristic(node) = beta * (1 - normalized_pagerank[node])
    This biases the search toward high-PageRank nodes (low heuristic value).
    "beta" controls how strongly the heuristic influences search.
    If beta == 0, algorithm just becomes Dijkstra's algorithm
    Note: This heuristic is not provably admissible, meaning it does not always find the most optimal path
    Returns (path, nodes_expanded, elapsed_time (in seconds))
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


# Function that uses the A* PageRank algorithm to find the k best paths
def k_best_paths(graph, start, goal, k, beta=5.0):
    # List to store the results for each path
    results = []

    # Holds edges removed from the graph, forcing A* to find a different path
    # These edges will be restored to the graph after the k paths are found
    removed_edges = []

    # Repeat following steps k times in an attempt to find k different paths
    for i in range(k):
        # Find shortest path using A* w/ PageRank algorithm
        path, nodes_expanded, time_elapsed = astar_with_pagerank(graph, start, goal, beta)

        # Stop looking for new shortest paths if none are found by exiting the loop
        if path is None:
            break

        results.append((path, nodes_expanded, time_elapsed))

        # Remove one unique edge of this path to force a new route next time
        if len(path) > 1:
            # FInd first edge in the path
            edge_to_remove = (path[0], path[1])  # Only removing 1st edge causes minimal graph disturbance
            # Remove the edge, but keep it in "removed_edges" for restoration later
            if graph.has_edge(*edge_to_remove):
                graph.remove_edge(*edge_to_remove)
                removed_edges.append(edge_to_remove)

    # Restore removed edges (so original graph remains unchanged)
    # Important if algorithm is run multiple times
    for source, target in removed_edges:
        graph.add_edge(source, target)

    return results


# Example usage of A* search with PageRank heuristic to find k shortest paths
if __name__ == "__main__":
    # Specify start and goal webpage from graph
    start_page = "Attack on Pearl Harbor"
    goal_page = "Double bass"
    k = 5

    print(f"\n{k} best paths found from the page \"{start_page}\" to \"{goal_page}\" using A* search with a PageRank-informed heuristic: ")
    results = k_best_paths(graph, start_page, goal_page, k, beta=5.0)
    total_time = 0
    for i, (path, nodes_expanded, time_elapsed) in enumerate(results, 1):
        print(f"\n=== Path #{i} ===")
        print(f"Path: {path}")
        print(f"Length: {len(path)}")
        print(f"Nodes Expanded: {nodes_expanded}")
        print(f"Time: {time_elapsed:.6f}s")
        total_time += time_elapsed
    print(f"\nTotal Time Elapsed: {total_time:.6f}s\n")
