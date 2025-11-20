from flask import Flask, render_template, request, jsonify
import networkx as nx
from pathfinder import *
import pandas as pd
from urllib.parse import unquote  # for decoding Unicode characters such as 'É'

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

# When browser sends request to the URL /suggest, call the function "suggest()" to handle this request
@app.route("/suggest")
def suggest():
    """Return article suggestions as the user types."""

    # Create data frame of all the articles with one column called "title"
    df = pd.read_csv("articles.tsv", sep="\t", header=None, names=["title"], comment="#")
    
    # Function to clean the article titles, making them human-readable
    def clean_title(raw):
        decoded = unquote(raw)  # Decode Unicode characters
        true_titles = decoded.replace("_", " ")  # Convert underscore separators to spaces
        return true_titles
    # Clean all article titles in our data frame
    article_titles = sorted(clean_title(t) for t in df["title"])
    
    # Grab the text the user types in to the input
    typed_text = request.args.get("q", "").lower()
    # Return no results if typed text is empty
    if not typed_text:
        return jsonify([])  # jsonify() turns Python list into JSON that JavaScript can read

    # Find every article containing the "typed_text" substring
    matches = [title for title in article_titles if typed_text in title.lower()]
    return jsonify(matches[:10])  # limit to top 10 results

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

    if (results):
        return render_template('home.html', results=results)
    else:
        message = "Could not find path between " + input1 + " and " + input2 + "."
        return render_template('home.html', message=message)

if __name__ == '__main__':
    app.run(port=5001, debug=True)