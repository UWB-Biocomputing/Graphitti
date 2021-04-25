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
}

All911Vertices::~All911Vertices() {
    if (size_ != 0) {
        delete[] callNum_;
        delete[] dispNum_;
    }

    callNum_ = nullptr;
    dispNum_ = nullptr;
}

// Allocate memory for all class properties
void All911Vertices::setupVertices() {
    AllVertices::setupVertices();

    callNum_ = new int[size_];
    dispNum_ = new int[size_];

    fill_n(callNum_, size_, 0);
    fill_n(dispNum_, size_, 0);
}

// Generate callNum_ and dispNum_ for all caller and psap nodes
// TODO: Create responder nodes
void All911Vertices::createAllVertices(Layout *layout) {
    vector<int> psapList;
    psapList.clear();

    for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {  
        // Create all callers
        if (layout->vertexTypeMap_[i] == CALR) {
            callNum_[i] = rng.inRange(callNumRange_[0], callNumRange_[1]);
        }

        // Find all PSAPs
        if(layout->vertexTypeMap_[i] == PSAP) {
            psapList.push_back(i);
        }
    }

    // Create all psaps
    for (int i = 0; i < psapList.size(); i++) {
        dispNum_[psapList[i]] = generateDispatcherCount(psapList[i], layout);
    }
}

// Generate a dispatcher count based on the callers in the PSAPs jurisdiction
int All911Vertices::generateDispatcherCount(int index, Layout *layout) {
    int psapQ = quadrant(index);
    int callerCount = 0;

    // Calculate total callers under this quadrant
    for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {  
        if (quadrant(i) == psapQ) {
            callerCount += callNum_[i];
        }
    }

    // Scale factor & create some randomness
    callerCount = (callerCount * dispNumScale_) + rng.inRange(-5, 5);
    if (callerCount < 1) { callerCount = 1; }

    return callerCount;
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