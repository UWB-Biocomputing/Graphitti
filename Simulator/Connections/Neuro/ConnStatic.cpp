#include "ConnStatic.h"
#include "AllEdges.h"
#include "AllNeuroEdges.h"
#include "AllVertices.h"
#include "OperationManager.h"
#include "GraphManager.h"
#include "ParameterManager.h"
#include "ParseParamError.h"
#include "XmlRecorder.h"

#ifdef USE_HDF5
#include "Hdf5Recorder.h"
#endif

#include <algorithm>

ConnStatic::ConnStatic() {
    threshConnsRadius_ = 0;
    connsPerVertex_ = 0;
    rewiringProbability_ = 0;
    radiiSize_ = 0;
}

void ConnStatic::setup() {
    int added = 0;
    AllNeuroEdges &neuroEdges = dynamic_cast<AllNeuroEdges &>(*edges_);
    LOG4CPLUS_INFO(fileLogger_, "Initializing connections");
 
    // Layout and Vertices from the Model
    Layout &layout = Simulator::getInstance().getModel().getLayout();
    AllVertices &vertices = layout.getVertices();
 
    // Initialize GraphManager 
    GraphManager<NeuralVertexProperties> &gm = GraphManager<NeuralVertexProperties>::getInstance();
    // Get the sorted edge list from GraphManager
    auto sorted_edge_list = gm.edgesSortByTarget();
 
    // add sorted edges
    for (auto it = sorted_edge_list.begin(); it != sorted_edge_list.end(); ++it) {
        size_t srcV = gm.source(*it);
        size_t destV = gm.target(*it);
        BGFLOAT weight = gm.weight(*it);
        edgeType type = layout.edgType(srcV, destV);
    
        BGFLOAT dist = layout.dist_(srcV, destV);
        LOG4CPLUS_DEBUG(edgeLogger_, "Source: " << srcV << " Dest: " << destV << " Dist: " << dist);
    
        // Add Edge 
        BGSIZE iEdg = edges_->addEdge(type, srcV, destV, Simulator::getInstance().getDeltaT());
         
        // Store the weight for the edge
        neuroEdges.W_[iEdg] = weight; 

        // no changes in stdp results?   
        // neuroEdges.W_[iEdg] = 0.5;

        // works on all files, except, test-tiny.graphml (works without added edges/weights)

        added++;
    }
    LOG4CPLUS_DEBUG(fileLogger_, "Added connections: " << added);
    // Rewiring if needed
    int nRewiring = static_cast<int>(added * rewiringProbability_);
    LOG4CPLUS_DEBUG(fileLogger_, "Expected rewiring connections: " << nRewiring);
}



void ConnStatic::registerGraphProperties() {
    Connections::registerGraphProperties();

    GraphManager<NeuralVertexProperties> &gm = GraphManager<NeuralVertexProperties>::getInstance();
    gm.registerProperty("source", &NeuralEdgeProperties::source);
    gm.registerProperty("target", &NeuralEdgeProperties::target);
    gm.registerProperty("weight", &NeuralEdgeProperties::weight);
}

void ConnStatic::loadParameters() {
    ParameterManager::getInstance().getBGFloatByXpath("//threshConnsRadius/text()", threshConnsRadius_);
    ParameterManager::getInstance().getIntByXpath("//connsPerNeuron/text()", connsPerVertex_);
    ParameterManager::getInstance().getBGFloatByXpath("//rewiringProbability/text()", rewiringProbability_);
}

void ConnStatic::printParameters() const {
    LOG4CPLUS_DEBUG(fileLogger_, "CONNECTIONS PARAMETERS"
                                 << "\n\tConnections Type: ConnStatic"
                                 << "\n\tConnection radius threshold: " << threshConnsRadius_
                                 << "\n\tConnections per neuron: " << connsPerVertex_
                                 << "\n\tRewiring probability: " << rewiringProbability_);
}
