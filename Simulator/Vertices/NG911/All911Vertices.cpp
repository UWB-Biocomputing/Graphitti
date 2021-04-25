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
    dispNum_ = nullptr; 
    respNum_ = nullptr;
}

All911Vertices::~All911Vertices() {
    if (size_ != 0) {
        delete[] callNum_;
        delete[] dispNum_;
        delete[] respNum_;
    }

    callNum_ = nullptr;
    dispNum_ = nullptr;
    respNum_ = nullptr;
}

// Allocate memory for all class properties
void All911Vertices::setupVertices() {
    AllVertices::setupVertices();

    callNum_ = new int[size_];
    dispNum_ = new int[size_];
    respNum_ = new int[size_];

    // Populate arrays with 0
    fill_n(callNum_, size_, 0);
    fill_n(dispNum_, size_, 0);
    fill_n(respNum_, size_, 0);
}

// Generate callNum_ and dispNum_ for all caller and psap nodes
void All911Vertices::createAllVertices(Layout *layout) {
    vector<int> psapList;
    vector<int> respList;
    psapList.clear();
    respList.clear();

    int callersPerQuadrant[] = {0, 0, 0, 0};
    int respPerQuadrant[] = {0, 0, 0, 0};

    for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {  
        // Create all callers
        if (layout->vertexTypeMap_[i] == CALR) {
            callNum_[i] = rng.inRange(callNumRange_[0], callNumRange_[1]);
            callersPerQuadrant[quadrant(i)] += callNum_[i];
        }

        // Find all PSAPs
        if(layout->vertexTypeMap_[i] == PSAP) {
            psapList.push_back(i);
        }

        // Find all resps
        if(layout->vertexTypeMap_[i] == RESP) {
            respList.push_back(i);
            respPerQuadrant[quadrant(i)] += 1;
        }
    }

    // Create all psaps
    // Dispatchers in a psap = [callers in the quadrant * k] + some randomness
    for (int i = 0; i < psapList.size(); i++) {
        int psapQ = quadrant(i);
        int dispCount = (callersPerQuadrant[psapQ] * dispNumScale_) + rng.inRange(-5, 5);
        if (dispCount < 1) { dispCount = 1; }
        dispNum_[psapList[i]] = dispCount;
    }

    // Create all responders
    // Responders in a node = [callers in the quadrant * k]/[number of responder nodes] + some randomness
    for (int i = 0; i < respList.size(); i++) {
        int respQ = quadrant(respList[i]);
        int respCount = (callersPerQuadrant[respQ] * respNumScale_)/respPerQuadrant[respQ] + rng.inRange(-5, 5);
        if (respCount < 1) { respCount = 1; }
        respNum_[respList[i]] = respCount;
    }
}

// Get the quadrant of the vertex
// Only built for 10x10 grid
// See: https://docs.google.com/spreadsheets/d/1DqP8sjkfJ_pkxtETzuEdoVZbWOGu633EMQAeShe5k68/edit?usp=sharing
int All911Vertices::quadrant(int index) {
    return (index%10 >= 5) + 2*(index < 50);
}

void All911Vertices::loadParameters() {
    ParameterManager::getInstance().getIntByXpath("//CallNum/min/text()", callNumRange_[0]);
    ParameterManager::getInstance().getIntByXpath("//CallNum/max/text()", callNumRange_[1]);
    ParameterManager::getInstance().getBGFloatByXpath("//DispNumScale/text()", dispNumScale_);
    ParameterManager::getInstance().getBGFloatByXpath("//RespNumScale/text()", respNumScale_);
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