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
    // CallNum_ = NULL; 
}

All911Vertices::~All911Vertices() {
    // if (size_ != 0) {
    //     delete[] CallNum_; 
    // }
    // CallNum_ = NULL; 
}

void All911Vertices::setupVertices() {
    AllVertices::setupVertices();

    // CallNum_ = new BGFLOAT[size_];

}

void All911Vertices::createAllVertices(Layout *layout) {
    /* set their specific types */
    // for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {  
    //     // set the vertex info for vertices
    //     createVertex(i, layout);
    // }
}

void All911Vertices:: createVertex(int index, Layout *layout) {
//    CallNum_[index] = rng.inRange(CallNumRange_[0], CallNumRange_[1]);
}

void All911Vertices::loadParameters() {
    ParameterManager::getInstance().getBGFloatByXpath("//CallNum/min/text()", CallNumRange_[0]);
    ParameterManager::getInstance().getBGFloatByXpath("//CallNum/max/text()", CallNumRange_[1]);
}


void All911Vertices::printParameters() const {

}

string All911Vertices::toString(const int index) const {
    return nullptr; // Change this
}

void All911Vertices::advanceVertices(IAllEdges &edges, const EdgeIndexMap *edgeIndexMap) {

}