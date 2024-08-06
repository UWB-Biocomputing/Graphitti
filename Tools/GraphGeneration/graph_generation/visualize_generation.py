# Jasleen Kaur Saini
# Zaina Shaikh
# Graph Generation
# visualize_generation.py

# Purpose: This script visualizes a network graph from a GraphML file. 
# It reads node and edge data from the GraphML file, creates a graph with NetworkX, 
# and generates a plot with Matplotlib. 

# Internal rectangles representing PSAP and Caller regions
# Nodes colored by type (PSAP, Caller, Responder)
# Edges representing connections between nodes
# A global bounding box around the entire network

import matplotlib.pyplot as plt
import networkx as nx
import xml.etree.ElementTree as ET

# Directly specify the file path and node type
graphml_path = "synth_graph.graphml"

# Load data from XML file
# Parse the GraphML file
# Get the root element of the XML tree
tree = ET.parse(graphml_path)  
root = tree.getroot()  

# Initialize the network graph
G = nx.Graph()  

# Define the namespace for the GraphML XML structure
nsmap = {'xmlns': 'http://graphml.graphdrawing.org/xmlns'}

# Prepare data structures for storing node positions, colors, and bounding boxes
node_positions = {}
node_colors = []

# Variables to store global and internal bounding boxes
global_bbox = None
psap_rects = []
caller_rects = []

# Parse the XML to extract bounding box information
for graph_data in root.findall('.//{http://graphml.graphdrawing.org/xmlns}graph'):
    # Extract global bounding box data
    global_bbox_data = graph_data.find('.//{http://graphml.graphdrawing.org/xmlns}data[@key="global_bounding_box"]')
    if global_bbox_data is not None:
        global_bbox = list(map(float, global_bbox_data.text.split(',')))
    
    # Extract internal PSAP region bounding boxes
    psap_rect_data = graph_data.find('.//{http://graphml.graphdrawing.org/xmlns}data[@key="psap_regions"]')
    if psap_rect_data is not None:
        psap_rects = [list(map(float, rect.split(','))) for rect in psap_rect_data.text.split(' | ')]
        
    # Extract internal Caller region bounding boxes
    caller_rect_data = graph_data.find('.//{http://graphml.graphdrawing.org/xmlns}data[@key="caller_regions"]')
    if caller_rect_data is not None:
        caller_rects = [list(map(float, rect.split(','))) for rect in caller_rect_data.text.split(' | ')]

# Parse nodes from the GraphML file
for node in root.findall('.//{http://graphml.graphdrawing.org/xmlns}node'):
    type_element = node.find('.//{http://graphml.graphdrawing.org/xmlns}data[@key="type"]')
    pos_element = node.find('.//{http://graphml.graphdrawing.org/xmlns}data[@key="pos"]')
    if type_element is not None and pos_element is not None:
        node_id = node.get('id')  
        x, y = map(float, pos_element.text.strip('()').split(','))  
        G.add_node(node_id, pos=(x, y))  

        # Determine node color based on type
        if type_element.text == 'PSAP':
            node_colors.append('red') 
        elif type_element.text == 'CALL':
            node_colors.append('blue')  
        elif type_element.text == 'RESP':
            node_colors.append('green')  
        else:
            node_colors.append('gray')  

# Parse edges from the GraphML file
for edge in root.findall('.//{http://graphml.graphdrawing.org/xmlns}edge'):
    edge_source = edge.get('source')  
    edge_target = edge.get('target')  
    if edge_source in G.nodes() and edge_target in G.nodes():
        G.add_edge(edge_source, edge_target)  

# Debugging output
print("Nodes and Attributes:")
for node in G.nodes(data=True):
    print(node)  

print("Edges:")
for edge in G.edges():
    print(edge)  # Print edges

print("Node Positions:")
print(nx.get_node_attributes(G, 'pos'))  

# Plotting
# Increase figure size
fig, ax = plt.subplots(figsize=(12, 12)) 

# Add internal rectangles for PSAP regions
for rect in psap_rects:
    xmin, ymin, width, height = rect
    rect_patch = plt.Rectangle((xmin, ymin), width, height, linewidth=5, edgecolor='coral', facecolor='none')
    ax.add_patch(rect_patch) 

# Add internal rectangles for Caller regions
for rect in caller_rects:
    xmin, ymin, width, height = rect
    rect_patch = plt.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='lightblue', facecolor='none')
    ax.add_patch(rect_patch)   

# Plot the network graph
node_pos = nx.get_node_attributes(G, 'pos')  
nx.draw(G, pos=node_pos, node_size=150, node_color=node_colors, with_labels=False, 
        edge_color='gray', alpha=0.7, width=2, ax=ax)  

# Add the global bounding box 
if global_bbox:
    xmin, ymin, xmax, ymax = global_bbox
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)  

# Create a color legend for the plot
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=20, label='System Perimeter'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='coral', markersize=20, label='PSAP Region'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=20, label='Caller Region'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=20, label='PSAP Node'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=20, label='Caller Node'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=20, label='Responder Node')
]

# Adjust legend location and size
ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, -0.27), fontsize=18)

# Set plot limits to ensure all elements are visible
if global_bbox:
    # Margin Size
    ax.set_xlim(global_bbox[0] - 2, global_bbox[2] + 2)  
    ax.set_ylim(global_bbox[1] - 2, global_bbox[3] + 2)  

plt.title("Graph Generation Visualization", fontsize=24, fontweight='bold')  
# Pad_inches to add more external padding to png
plt.savefig('graph_generation.png', bbox_inches='tight', pad_inches=1.5)  
plt.show()
