'''
GETGRAPHEDGES Generate a graphml file from the weights matrix simulation output

This function reads the Graphitti weight matrix output from the
growth simulation and reformats it to graphml for input into the
STDP simulation.

Input:
weights_file - Weight matrix output from Graphitti. The entire path can be used; for example
               '/CSSDIV/research/biocomputing/data/2025/tR_1.0--fE_0.90_10000/weights-epoch-1.xml'
graphml_file - graphml file to add the edges. This is typically the same file 
               used as input to the Graphitti simulation.

Output:
  - weight_graph.graphml

Author: Vanessa Arndorfer (vanessa.arndorfer@gmail.com)
Last updated: 07/01/2025
'''

import numpy as np
import os
import pandas as pd
import sys
import xml.etree.ElementTree as ET 


def xmlToNumpy(node, rows, cols):
    print("Converting xml to matrix for: " + node.tag)

    m = np.zeros(shape=(rows, cols))
    r = 0
    c = 0
    for child in node:
        m[r][c] = float(child.text)
        c += 1

        # new row
        if c == cols:
            r += 1
            c = 0

    return m


def getWeightMatrix(file_name, src_root, weights_root):
    print("Building weights matrix for: " + file_name)

    tree = ET.parse(file_name)
    root = tree.getroot()

    idNum = 0
    srcIdx = root.find(src_root)
    weights = root.find(weights_root)

    rows = 10000
    cols = 200

    srcIdx_np = xmlToNumpy(srcIdx, rows, cols)
    weights_np = xmlToNumpy(weights, rows, cols)

    print("Converting matrices into square format...")
    edge_weights = np.zeros(shape=(rows, rows))

    weight_count = 0

    for r in range(weights_np.shape[0]):
        for c in range(weights_np.shape[1]):
            if weights_np[r][c] != 0:
                weight_count += 1
                src = int(srcIdx_np[r][c])
                edge_weights[src][r] = weights_np[r][c]

    print("Total weighted edges: " + str(weight_count))

    return edge_weights


def getEdgeGraphML(edge_weights, graphml_file, output_dir):
    print("Adding edges to the graph file: " + graphml_file)

    # register xml namespaces
    ET.register_namespace('', "http://graphml.graphdrawing.org/xmlns")
    ET.register_namespace('xsi', "http://www.w3.org/2001/XMLSchema-instance")
    ET.register_namespace('xsi:schemaLocation', "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd")

    tree = ET.parse(graphml_file)
    root = tree.getroot()

    # define weight attribute for edges
    weight_key = ET.Element("key", attrib={"id":"weight", "for":"edge", "attr.name":"weight", "attr.type":"double"})
    root.append(weight_key)

    idNum = 0
    graph = root.find('./{http://graphml.graphdrawing.org/xmlns}graph')

    for r in range(edge_weights.shape[0]):
        for c in range(edge_weights.shape[1]):
            if edge_weights[r][c] != 0:
                edgeId = "e" + str(idNum)
                edge = ET.Element("edge", attrib={"id":edgeId, "source":str(r), "target":str(c)})
                data = ET.SubElement(edge, "data", attrib={"key": "weight"})
                data.text = str(edge_weights[r][c])
                graph.append(edge)
                idNum += 1

    print(str(idNum) + " edges created")
    ET.indent(tree, space="\t", level=0)
    tree.write(output_dir + "/weight_graph.graphml", encoding="utf-8", xml_declaration=True)


if __name__ == "__main__": 
    # execution format: python3.9 ./getGraphEdges.py weights_file graphml_file
    # example: python3.9 ./getGraphEdges.py /CSSDIV/research/biocomputing/data/2025/tR_1.0--fE_0.90_10000/weights-epoch-25.xml configfiles/graphs/fE_0.90_10000.graphml
    weights_file = sys.argv[1]
    graphml_file = sys.argv[2]

    ht = os.path.split(weights_file)
    output_dir = ht[0]
    print("Output dir: " + output_dir)

    df = getWeightMatrix(weights_file, "SourceVertexIndex", "WeightMatrix")
    getEdgeGraphML(df, graphml_file, output_dir)
        