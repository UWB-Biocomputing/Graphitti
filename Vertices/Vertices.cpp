//
// Created by chris on 6/22/2020.
//

#include "Vertices.h"
#include "OperationManager.h"


using namespace std;

Vertices::Vertices() {
    // func = name of variable,,, dummy name
    auto func = std::bind(&Vertices::allocateMemory, this);
    //get instance of operation manager by putting 
    // register function we just made
    // want to bind the method with the object we just created (func)
    // we need both the method and object bound together to make it the object within the list. 
    OperationManager::getInstance()->registerOperation(Operations::op::allocateMemory, func);
}

void Vertices::allocateMemory() {
}

// Default constructor
AllNeurons::AllNeurons() : 
        size(0), 
        nParams(0)
{
    summation_map = NULL;
}

AllNeurons::~AllNeurons()
{
    freeResources();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllNeurons::allocateMemory()   //setupNeurons (old name)
{
    // new code chris taught me! 
    size = Simulator::getInstance()->getTotalNeurons();
    // TODO: Rename variables for easier identification
    summation_map = new BGFLOAT[size];

    for (int i = 0; i < size; ++i) {
        summation_map[i] = 0;
    }

    // old code: sim_info->pSummationMap = summation_map;
    Simulator::getInstance->setSummationMap(summation_map);
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllNeurons::cleanupNeurons()
{
    freeResources();
}

/*
 *  Deallocate all resources
 */
void AllNeurons::freeResources()
{
    if (size != 0) {
        delete[] summation_map;
    }
        
    summation_map = NULL;

    size = 0;
}
