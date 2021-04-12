/**
* @file Layout911.cpp
* 
* @ingroup Simulator/Layouts/NG911
*
* @brief The Layout class defines the layout of vertices in networks
*/

#include "Layout911.h"
#include "ParameterManager.h"

Layout911::Layout911() {

}

Layout911::~Layout911() {

}

void Layout911::printParameters() const {

}

void Layout911::generateVertexTypeMap(int numVertices) {

}

void Layout911::initStarterMap(const int numVertices) {

}

void Layout911::loadParameters() {
//     // Get the file paths for the Neuron lists from the configuration file
//    string callerFilePath;
//    string psapFilePath;
//    string responderFilePath;
//    if (!ParameterManager::getInstance().getStringByXpath("//LayoutFiles/activeNListFileName/text()",
//                                                          callerFilePath)) {
//       throw runtime_error("In Layout::loadParameters() Endogenously "
//                           "active neuron list file path wasn't found and will not be initialized");
//    }
//    if (!ParameterManager::getInstance().getStringByXpath("//LayoutFiles/inhNListFileName/text()",
//                                                          psapFilePath)) {
//       throw runtime_error("In Layout::loadParameters() "
//                           "Inhibitory neuron list file path wasn't found and will not be initialized");
//    }

//    // Initialize Neuron Lists based on the data read from the xml files
//    if (!ParameterManager::getInstance().getIntVectorByXpath(callerFilePath, "A", endogenouslyActiveNeuronList_)) {
//       throw runtime_error("In Layout::loadParameters() "
//                           "Endogenously active neuron list file wasn't loaded correctly"
//                           "\n\tfile path: " + callerFilePath);
//    }
//    numEndogenouslyActiveNeurons_ = endogenouslyActiveNeuronList_.size();
//    if (!ParameterManager::getInstance().getIntVectorByXpath(psapFilePath, "I", inhibitoryNeuronLayout_)) {
//       throw runtime_error("In Layout::loadParameters() "
//                           "Inhibitory neuron list file wasn't loaded correctly."
//                           "\n\tfile path: " + psapFilePath);
//    }
}