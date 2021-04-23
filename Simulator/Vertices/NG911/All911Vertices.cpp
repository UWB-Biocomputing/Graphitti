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
    CallNum_ = nullptr; 
}

All911Vertices::~All911Vertices() {
    if (size_ != 0) {
        delete[] CallNum_; 
    }
    CallNum_ = nullptr; 
}

void All911Vertices::setupVertices() {
    AllVertices::setupVertices();

    CallNum_ = new int[size_];
    fill_n(CallNum_, size_, 0);    
    // take call num
    // assign random # of dispatchers depending on zone (minimum = 1) 
    // 



}

void All911Vertices::createAllVertices(Layout *layout) {
    /* set their specific types */
    for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {  
        // set the vertex info for vertices
        createVertex(i, layout);
    }
}

void All911Vertices:: createVertex(int index, Layout *layout) {
   // CallNum_[index] = static_cast<int>(rng.inRange(CallNumRange_[0], CallNumRange_[1]));
   CallNum_[index] = rng.inRange(CallNumRange_[0], CallNumRange_[1]);

}

void All911Vertices::loadParameters() {
    ParameterManager::getInstance().getIntByXpath("//CallNum/min/text()", CallNumRange_[0]);
    ParameterManager::getInstance().getIntByXpath("//CallNum/max/text()", CallNumRange_[1]);
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