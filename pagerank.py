import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy

#reading and outputting number of nodes and edges for checking
G = nx.read_graphml("Graphs/linkGraph.graphml")

numNodes = nx.number_of_nodes(G)
print(f"Number of Nodes: {numNodes}")

numEdges = nx.number_of_edges(G)
print(f"Number of Edges: {numEdges}")

#getting pagerank
#using alpha as 0.85 (that is the value used by google)
pagerank_scores = nx.pagerank(G, alpha=0.85)

#setting up values for visualizing the pageranks
#get the max and min page rank for scale
print("getting the min and max of page ranks")
max_pr = max(pagerank_scores.values())
min_pr = min(pagerank_scores.values())

print("giving nodes values based on scale")
if max_pr == min_pr: # Handle case where all scores are the same
    node_sizes = [4000] * len(G.nodes)
else:
    node_sizes = [(v - min_pr) / (max_pr - min_pr) * 10000 + 10
                  for v in pagerank_scores.values()]
    
#setting values for repulsion on graph
pr_values = np.array(list(pagerank_scores.values()))

min_pr = pr_values.min()
max_pr = pr_values.max()
if max_pr == min_pr:
    repulsion_multipliers = np.ones(len(G.nodes))
else:
    repulsion_multipliers = (pr_values - min_pr) / (max_pr - min_pr)
#color the nodes based on pagerank
print("coloring the nodes")
node_colors = list(pagerank_scores.values())

#map the values to the nodes
node_multiplier_map = dict(zip(G.nodes(), repulsion_multipliers))

plt.figure(figsize=(10, 7))

# Define the layout for the nodes (e.g., spring_layout for a more organic look)
pos = nx.spring_layout(G, seed=42) # Start with a random but stable layout
nodes = list(G.nodes())
num_nodes = len(nodes)
damping_factor = 0.5  # Controls how quickly the positions adjust
iterations = 10      # Number of times to apply the custom force

# Convert positions to a NumPy array for easier math
pos_array = np.array([pos[node] for node in nodes])
iterationNum = 0
for i in range(iterations):
    # Calculate total force (delta) on each node
    total_delta = np.zeros_like(pos_array)
    
    iterationNum = iterationNum + 1
    print(f"Iteration {iterationNum}")

    for j in range(num_nodes):
        node_j = nodes[j]
        for k in range(num_nodes):
            if j == k:
                continue
                
            node_k = nodes[k]

            # Vector representing the direction and distance from k to j
            difference_vector = pos_array[j] - pos_array[k]
            distance = np.linalg.norm(difference_vector)
            
            # Prevent division by zero if nodes are exactly on top of each other
            if distance == 0:
                continue
            
            # --- CORE LOGIC: PageRank-Weighted Repulsion ---
            
            # The force is inverse-square (F ~ 1/d^2), mimicking gravity/electrostatics.
            # We multiply the force magnitude by the source node's PageRank multiplier.
            
            # PageRank-weighted repulsion magnitude
            repulsion_strength = 0.01 * node_multiplier_map[node_k]
            
            # Force magnitude (F = strength / distance^2)
            force_magnitude = repulsion_strength / (distance ** 2)
            
            # Apply force in the direction of the difference vector
            # The unit vector is difference_vector / distance
            force_vector = (difference_vector / distance) * force_magnitude
            
            # Add this force to node j's total delta
            total_delta[j] += force_vector
    
    # Update positions based on the calculated forces
    # Damping factor ensures smooth, non-chaotic movement
    pos_array += total_delta * damping_factor

# Convert the NumPy array back to the NetworkX position dictionary
pos_final = {nodes[i]: pos_array[i] for i in range(num_nodes)}

# Draw the nodes, using the scaled sizes and color map
nodes = nx.draw_networkx_nodes(
    G, pos, 
    node_size=node_sizes, 
    node_color=node_colors, 
    cmap=plt.cm.RdYlBu, # Colormap: Red-Yellow-Blue, often used for spectrum visualization
    alpha=0.8
)

# Draw the edges
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=5, edge_color='gray')

# Draw the labels
#nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# Add a color bar to explain the colors
if nodes is not None:
    plt.colorbar(nodes, orientation='vertical', label='PageRank Score')

plt.title("Graph Visualization Emphasizing PageRank")
plt.axis('off') # Turn off the axis lines and labels
print("Displaying graph")
plt.show()
print("graph displayed")