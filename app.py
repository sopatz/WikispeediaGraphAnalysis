from flask import Flask, render_template, request
import networkx as nx
from pathfinder import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        #reading and outputting number of nodes and edges for checking
        graph = nx.read_graphml("Graphs/linkGraph.graphml")

        # gets the two inputs from the form
        input1 = request.form.get('startInput')
        input2 = request.form.get('endInput')
        input3 = request.form.get('numInput', type=int)

        # get path, just need to display it onto the website
        results = k_best_paths(graph, input1, input2, input3, beta=5.0)
        # for i, (path, nodes_expanded) in enumerate(results, 1):
        #     # terminal info
        #     print(f"Path from {input1} to {input2}: ")
        #     print(f"Path: {path}")
        #     print(f"Path length: {len(path)}")
        #     print(f"Nodes expanded: {nodes_expanded}")

    return render_template('home.html', results=results)

if __name__ == '__main__':
    app.run(port=5001, debug=True)