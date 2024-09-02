import networkx as nx

# Load the graph from the gpickle file
G_loaded = nx.read_gpickle("knowledge_graph.gpickle")

# Retrieve all nodes with their attributes
nodes_data = [{'id': node, 'attributes': G_loaded.nodes[node]} for node in G_loaded.nodes()]

# Print nodes information
for node_info in nodes_data:
    print(f"Node ID: {node_info['id']}, Attributes: {node_info['attributes']}")

# Retrieve all edges with their attributes
edges_data = [{'from': u, 'to': v, 'attributes': G_loaded.edges[u, v]} for u, v in G_loaded.edges()]

# Print edges information
for edge_info in edges_data:
    print(f"From: {edge_info['from']}, To: {edge_info['to']}, Attributes: {edge_info['attributes']}")
