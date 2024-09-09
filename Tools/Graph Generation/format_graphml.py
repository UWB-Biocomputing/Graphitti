# Jasleen Kaur Saini
# Zaina Shaikh
# Graph Generation
# format_graphml.py

# Purpose:
# This script reads an existing GraphML file, processes its nodes and edges,
# and outputs a new GraphML file with a modified format. The new format
# includes updated attributes for nodes and uses a specific ID mapping for 
# better organization and simulation testing. 

import xml.etree.ElementTree as ET
import random

# Load the original GraphML file
tree = ET.parse('synth_graph.graphml')
root = tree.getroot()

# Define namespaces for the GraphML format
ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}

# Create a new GraphML structure
new_root = ET.Element('graphml', xmlns='http://graphml.graphdrawing.org/xmlns')

# Define keys for the new GraphML format
keys = [
    {'id': 'segments', 'for': 'node', 'attr.name': 'segments', 'attr.type': 'string'},
    {'id': 'trunks', 'for': 'node', 'attr.name': 'trunks', 'attr.type': 'int'},
    {'id': 'servers', 'for': 'node', 'attr.name': 'servers', 'attr.type': 'int'},
    {'id': 'x', 'for': 'node', 'attr.name': 'x', 'attr.type': 'double'},
    {'id': 'y', 'for': 'node', 'attr.name': 'y', 'attr.type': 'double'},
    {'id': 'type', 'for': 'node', 'attr.name': 'type', 'attr.type': 'string'},
    {'id': 'name', 'for': 'node', 'attr.name': 'name', 'attr.type': 'string'},
    {'id': 'objectID', 'for': 'node', 'attr.name': 'objectID', 'attr.type': 'string'},
]

# Create and add key elements to the new GraphML structure
for key in keys:
    key_elem = ET.Element('key')
    key_elem.set('id', key['id'])
    key_elem.set('for', key['for'])
    key_elem.set('attr.name', key['attr.name'])
    key_elem.set('attr.type', key['attr.type'])
    new_root.append(key_elem)

# Create a new graph element
graph = ET.SubElement(new_root, 'graph', edgedefault='directed')

# Function to generate a large volume of segments data for callers based on their coordinates
def generate_segments(x, y, num_segments=50):
    grid_size = 0.05 
    segments = []
    for _ in range(num_segments):
        x1 = x + (random.random() - 0.5) * grid_size
        y1 = y + (random.random() - 0.5) * grid_size
        x2 = x1 + (random.random() - 0.5) * grid_size
        y2 = y1 + (random.random() - 0.5) * grid_size
        segments.append([(x1, y1), (x2, y2)])
    return segments

# Function to generate random integers within a specified range
def get_random_int(min_val, max_val):
    return random.randint(min_val, max_val)

# Create a mapping from old node IDs to new IDs
node_id_map = {}
new_id_counter = 0

# Process each node in the original GraphML file
for node in root.findall('graphml:graph/graphml:node', ns):
    old_id = node.get('id')
    node_data = {data.get('key'): data.text for data in node.findall('graphml:data', ns)}
    
    # Define new node IDs based on a simple counter
    new_id = str(new_id_counter)
    new_id_counter += 1
    node_id_map[old_id] = new_id
    
    new_node = ET.SubElement(graph, 'node', id=new_id)
    
    # Add data elements to the new node
    if 'type' in node_data:
        node_type = node_data['type']
        if node_type == 'CALL':
            ET.SubElement(new_node, 'data', key='type').text = 'CALR' 
            if 'pos' in node_data:
                x, y = node_data['pos'].strip('()').split(', ')
                segments = generate_segments(float(x), float(y))
                segments_text = ', '.join(
                    f"[{', '.join(f'({x}, {y})' for x, y in segment)}]" for segment in segments
                )
                ET.SubElement(new_node, 'data', key='segments').text = segments_text
            ET.SubElement(new_node, 'data', key='objectID').text = f"CALL_{new_id}@GraphGeneration.gov"
            ET.SubElement(new_node, 'data', key='name').text = "UNKNOWN"
        elif node_type == 'PSAP':
            ET.SubElement(new_node, 'data', key='type').text = node_type
            ET.SubElement(new_node, 'data', key='objectID').text = f"PSAP_{new_id}@GraphGeneration.gov"
            ET.SubElement(new_node, 'data', key='name').text = "UNKNOWN"
            ET.SubElement(new_node, 'data', key='trunks').text = str(get_random_int(5, 10))
            ET.SubElement(new_node, 'data', key='servers').text = str(get_random_int(3, 5))
        elif node_type == 'RESP':
            responder_type = node_data.get('responder_type', 'RESP')
            ET.SubElement(new_node, 'data', key='type').text = responder_type  # Set the correct responder type
            ET.SubElement(new_node, 'data', key='objectID').text = f"{responder_type}_{new_id}@GraphGeneration.gov"
            ET.SubElement(new_node, 'data', key='name').text = "UNKNOWN"
            ET.SubElement(new_node, 'data', key='trunks').text = str(get_random_int(6, 12))
            ET.SubElement(new_node, 'data', key='servers').text = str(get_random_int(3, 6))
    
    if 'pos' in node_data and node_data['type'] != 'CALL':
        x, y = node_data['pos'].strip('()').split(', ')
        ET.SubElement(new_node, 'data', key='x').text = x
        ET.SubElement(new_node, 'data', key='y').text = y

# Collect all edges
edges = []

# Process each edge in the original GraphML file
for edge in root.findall('graphml:graph/graphml:edge', ns):
    source = node_id_map[edge.get('source')]
    target = node_id_map[edge.get('target')]
    edges.append((source, target))
    edges.append((target, source))  # Add the reverse edge

# Sort edges by source ID
edges.sort()

# Add edges to the new graph element
for source, target in edges:
    ET.SubElement(graph, 'edge', source=source, target=target)

# Function to pretty-print XML with indentation for better readability
def pretty_print(elem, level=0):
    indent = "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = "\n" + indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = "\n" + indent
        for elem in elem:
            pretty_print(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = "\n" + indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = "\n" + indent

# Apply pretty-printing to the new XML 
pretty_print(new_root)

# Write the GraphML structure to a file
new_tree = ET.ElementTree(new_root)
new_tree.write('synth_input.graphml', encoding='utf-8', xml_declaration=True)

print("The GraphML has been formatted and saved to 'synth_output2.graphml'.")
