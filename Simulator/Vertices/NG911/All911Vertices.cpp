/**
 * @file All911Vertices.cpp
 * 
 * @ingroup Simulator/Vertices/NG911
 *
 * @brief A container of all 911 vertex data
 */

#include "All911Vertices.h"
#include "ParameterManager.h"


All911Vertices::All911Vertices() {
    callNum_ = nullptr; 
}

All911Vertices::~All911Vertices() {
    if (size_ != 0) {
        delete[] callNum_; 
    }
    callNum_ = nullptr; 
}

void All911Vertices::setupVertices() {
    AllVertices::setupVertices();

    callNum_ = new int[size_];
    fill_n(callNum_, size_, 0);    

}

void All911Vertices::createAllVertices(Layout *layout) {
    /* set their specific types */
    for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {  
        // set the vertex info for vertices
        createVertex(i, layout);
    }
}

void All911Vertices:: createVertex(int index, Layout *layout) {
    callNum_[index] = rng.inRange(callNumRange_[0], callNumRange_[1]);
    
    // ******WORKING*****
    // if this is a psap node, 
        // then populate me with a random num based on my pop. 
    if(layout->vertexTypeMap_[index] == PSAP) {

    }


}

void All911Vertices::loadParameters() {
    ParameterManager::getInstance().getIntByXpath("//CallNum/min/text()", callNumRange_[0]);
    ParameterManager::getInstance().getIntByXpath("//CallNum/max/text()", callNumRange_[1]);
}


void All911Vertices::printParameters() const {

}

string All911Vertices::toString(const int index) const {
    return nullptr; // Change this
}

#if !defined(USE_GPU)

void All911Vertices::advanceVertices(AllEdges &edges, const EdgeIndexMap *edgeIndexMap) {

}

#endif