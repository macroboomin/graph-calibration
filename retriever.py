import networkx as nx
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]= "7"

# Load the knowledge graph
G_loaded = nx.read_gpickle("./bio/knowledge_graph.gpickle")

# Initialize the model and tokenizer for Phi-3
model = AutoModelForCausalLM.from_pretrained(
    "EmergentMethods/Phi-3-mini-4k-instruct-graph",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
)
tokenizer = AutoTokenizer.from_pretrained("EmergentMethods/Phi-3-mini-4k-instruct-graph")

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
)

generation_args = { 
    "max_new_tokens": 2000, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
}

def extract_entities_and_relationships(query_text):
    messages = [
        {"role": "system", "content": """
        A chat between a curious user and an artificial intelligence Assistant. The Assistant is an expert at identifying entities and relationships in text. The Assistant responds in JSON output only.

        The User provides text in the format:

        -------Text begin-------
        <User provided text>
        -------Text end-------

        The Assistant follows the following steps before replying to the User:

        1. **identify the most important entities** The Assistant identifies the most important entities in the text. These entities are listed in the JSON output under the key "nodes", they follow the structure of a list of dictionaries where each dict is:

        "nodes":[{"id": <entity N>, "type": <type>, "detailed_type": <detailed type>}, ...]

        where "type": <type> is a broad categorization of the entity. "detailed type": <detailed_type>  is a very descriptive categorization of the entity.

        2. **determine relationships** The Assistant uses the text between -------Text begin------- and -------Text end------- to determine the relationships between the entities identified in the "nodes" list defined above. These relationships are called "edges" and they follow the structure of:

        "edges":[{"from": <entity 1>, "to": <entity 2>, "label": <relationship>}, ...]

        The <entity N> must correspond to the "id" of an entity in the "nodes" list.

        The Assistant never repeats the same node twice. The Assistant never repeats the same edge twice.
        The Assistant responds to the User in JSON only, according to the following JSON schema:

        {"type":"object","properties":{"nodes":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"type":{"type":"string"},"detailed_type":{"type":"string"}},"required":["id","type","detailed_type"],"additionalProperties":false}},"edges":{"type":"array","items":{"type":"object","properties":{"from":{"type":"string"},"to":{"type":"string"},"label":{"type":"string"}},"required":["from","to","label"],"additionalProperties":false}}},"required":["nodes","edges"],"additionalProperties":false}
        """}, 
        {"role": "user", "content": f"""
        -------Text begin-------
        {query_text}
        -------Text end-------
        """}
    ]
    
    output = pipe(messages, **generation_args)
    generated_text = output[0]['generated_text']
    
    try:
        result = eval(generated_text)
        return result.get('nodes', []), result.get('edges', [])
    except Exception as e:
        print(f"Failed to parse output: {e}")
        return [], []

# Example query
query_text = "Which of the following would result in Angelman syndrome?"
query_nodes, query_edges = extract_entities_and_relationships(query_text)

def find_similar_and_adjacent_nodes(G, query_nodes):
    similar_nodes = set()  # Use a set to avoid duplicates
    for query_node in query_nodes:
        for node, attrs in G.nodes(data=True):
            if node == query_node['id']:
                similar_nodes.add(node)
                
                # Add all directly adjacent nodes (both predecessors and successors)
                neighbors = set(G.predecessors(node)).union(set(G.successors(node)))
                similar_nodes.update(neighbors)
                
    return list(similar_nodes)

def get_subgraph(G, nodes):
    subgraph = G.subgraph(nodes)
    return subgraph

# Find similar nodes and their adjacent nodes in the graph
similar_and_adjacent_nodes = find_similar_and_adjacent_nodes(G_loaded, query_nodes)

# Extract the subgraph with these nodes
subgraph = get_subgraph(G_loaded, similar_and_adjacent_nodes)

'''
# Retrieve all nodes with their attributes
nodes_data = [{'id': node, 'attributes': subgraph.nodes[node]} for node in subgraph.nodes()]

# Print nodes information
for node_info in nodes_data:
    print(f"Node ID: {node_info['id']}, Attributes: {node_info['attributes']}")

# Retrieve all edges with their attributes
edges_data = [{'from': u, 'to': v, 'attributes': subgraph.edges[u, v]} for u, v in subgraph.edges()]

# Print edges information
for edge_info in edges_data:
    print(f"From: {edge_info['from']}, To: {edge_info['to']}, Attributes: {edge_info['attributes']}")
'''

def generate_summary(subgraph):
    nodes_data = []
    for node in subgraph.nodes:
        node_data = {
            'id': node,
            'type': G_loaded.nodes[node].get('type', 'Unknown'),
            'detailed_type': G_loaded.nodes[node].get('detailed_type', 'Unknown')
        }
        nodes_data.append(node_data)
    
    edges_data = [{'from': u, 'to': v, 'label': G_loaded.edges[u, v].get('label', 'Unknown')} for u, v in subgraph.edges]
    
    summary_query = f"""
    Based on the following nodes and edges, please generate a summary:

    Nodes: {json.dumps(nodes_data, indent=4)}

    Edges: {json.dumps(edges_data, indent=4)}
    """
    
    messages = [
        {"role": "system", "content": """
        A chat between a curious user and an artificial intelligence Assistant. The Assistant generates summaries of the given data.
        """}, 
        {"role": "user", "content": summary_query}
    ]
    
    output = pipe(messages, **generation_args)
    return output[0]['generated_text']

# Generate the summary
summary = generate_summary(subgraph)
#print(summary)

def retrieve_information(query_text):
    # Step 1: Extract entities and relationships from the query
    query_nodes, query_edges = extract_entities_and_relationships(query_text)
    
    # Step 2: Find similar nodes and their adjacent nodes in the knowledge graph
    similar_and_adjacent_nodes = find_similar_and_adjacent_nodes(G_loaded, query_nodes)
    
    # Step 3: Get the subgraph containing these nodes
    subgraph = get_subgraph(G_loaded, similar_and_adjacent_nodes)
    
    # Step 4: Generate a summary based on the subgraph
    summary = generate_summary(subgraph)
    
    return summary

# Example usage
query = "Which of the following would result in Angelman syndrome?"
result_summary = retrieve_information(query)
print(result_summary)
