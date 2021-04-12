/**
 * @file Xml911Recorder.cpp
 * 
 * @ingroup Simulator/Recorders
 *
 * @brief Header file for Xml911Recorder.h
 *
 * The Xml911Recorder provides a mechanism for recording vertex's layout,
 * and compile history information on xml file:
 *     -# vertex's locations, and type map,
 */

#include "Xml911Recorder.h"


Xml911Recorder::Xml911Recorder() {

}

Xml911Recorder::~Xml911Recorder() {

}

/// Init radii and rates history matrices with default values
void Xml911Recorder::initDefaultValues() {
    
}

/// Init radii and rates history matrices with current radii and rates
void Xml911Recorder::initValues() {
    
}

/// Get the current radii and rates vlaues
void Xml911Recorder::getValues() {
    
}

/// Compile history information in every epoch
///
/// @param[in] vertices   The entire list of vertices.
void Xml911Recorder::compileHistories(IAllVertices &vertices) {
    
}

/// Writes simulation results to an output destination.
///
/// @param  vertices the Neuron list to search from.
void Xml911Recorder::saveSimData(const IAllVertices &vertices) {
    
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void Xml911Recorder::printParameters() {
    
}