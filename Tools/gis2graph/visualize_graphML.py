import matplotlib.pyplot as plt
import networkx as nx
import xml.etree.ElementTree as ET
#necessary imports to get and graph data

#loads data
tree = ET.parse('graph_files/King_county_NG911.graphml')
root = tree.getroot()
#initialize graph
G = nx.Graph()

# Type attribute key
type_attribute_key = 'type'
# Namespace from root
nsmap = {'xmlns': 'http://graphml.graphdrawing.org/xmlns'}

node_positions = {}
psap_ids = []
psap_coordinates = []
calr_nodes = []

#stores all PSAP nodes
for node in root.findall('.//{http://graphml.graphdrawing.org/xmlns}node'):
    type_element = node.find(f'.//{{{nsmap["xmlns"]}}}data[@key="{type_attribute_key}"]')
    if type_element is not None and type_element.text == 'PSAP':
        node_id = node.get('id')
        psap_ids.append(node_id)

        for data in node.findall(f'.//{{{nsmap["xmlns"]}}}data', namespaces=nsmap):
            if data.get('key') == 'x':
                node_x = float(data.text)
                psap_coordinates.append(node_x)
            elif data.get('key') == 'y':
                node_y = float(data.text)
                psap_coordinates.append(node_y)

        if 'node_x' in locals() and 'node_y' in locals():
            G.add_node(node_id)
            node_positions[node_id] = (node_x, node_y)
            G.nodes[node_id]['pos'] = (node_x, node_y)
            G.nodes[node_id]['color'] = 'cyan'
    #find all EMS nodes
    elif type_element is not None and type_element.text == 'EMS':
        node_id = node.get('id')

        for data in node.findall(f'.//{{{nsmap["xmlns"]}}}data', namespaces = nsmap):
            if data.get('key') == 'x':
                node_x = float(data.text)
            if data.get('key') == 'y':
                node_y = float(data.text)

        if 'node_x' in locals() and 'node_y' in locals():
            G.add_node(node_id)
            node_positions[node_id] = (node_x, node_y)
            G.nodes[node_id]['pos'] = (node_x, node_y)
            G.nodes[node_id]['color'] = 'blue'
    



for node in root.findall('.//{http://graphml.graphdrawing.org/xmlns}node'):
    type_element = node.find(f'.//{{{nsmap["xmlns"]}}}data[@key="{type_attribute_key}"]')
    if type_element is not None and type_element.text == 'CALR':
        node_id = node.get('id')
        calr_nodes.append(node_id)

        segments_data = node.find(f'.//{{{nsmap["xmlns"]}}}data[@key="segments"]')
        if segments_data is not None:
            segments_str = segments_data.text
            segments_str = segments_str.replace("[(", "").replace(")]", "")
            segments_list = segments_str.split("), (")

            # Convert segments into a list of coordinate tuples
            coordinates = [tuple(map(float, segment.split(", "))) for segment in segments_list]

            # Calculate the average position as the node position- this should be the center of the region- can map 
            # onto the graph
            avg_x = sum(coord[0] for coord in coordinates) / len(coordinates)
            avg_y = sum(coord[1] for coord in coordinates) / len(coordinates)

            G.add_node(node_id)
            node_positions[node_id] = (avg_x, avg_y)
            G.nodes[node_id]['pos'] = (avg_x, avg_y)
            G.nodes[node_id]['color'] = 'green' 



for edge in root.findall('.//{http://graphml.graphdrawing.org/xmlns}edge'):
    edge_source = edge.get('source')
    edge_target = edge.get('target')

    # Check if both source and target nodes are in G
    if edge_source in G.nodes() and edge_target in G.nodes():
        G.add_edge(edge_source, edge_target)


fig, ax = plt.subplots()

edge_alpha = 0.4
default_node_color = 'gray' 
node_colors = [G.nodes[node_id].get('color', default_node_color) for node_id in G.nodes()]
nx.draw(G, node_size = 30, pos = node_positions, node_color=node_colors, with_labels=False, alpha = edge_alpha, width = 0.3, ax = ax)# prints graph

label_pos = {k: (v[0], v[1] + 0.01) for k, v in node_positions.items()}  # Adjust label y-coordinates
labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, label_pos, labels=labels, font_size=8)

image = plt.imread('/Users/bennettye/Desktop/Screenshot 2023-08-17 at 2.31.56 PM.png')
image_extent = ax.get_xlim() + ax.get_ylim()

ax.imshow(image, extent=image_extent, aspect='auto', alpha=0.5)

ax.set_axis_on()
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

plt.show()