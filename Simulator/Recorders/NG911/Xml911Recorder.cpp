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
#include "All911Vertices.h"
#include "Connections911.h"
// #include "Global.h"

/// Init radii and rates history matrices with default values
void Xml911Recorder::initDefaultValues()
{
}

/// Init radii and rates history matrices with current radii and rates
void Xml911Recorder::initValues()
{
}

/// Get the current radii and rates vlaues
void Xml911Recorder::getValues()
{
}

/// Compile history information in every epoch
///
/// @param[in] vertices   The entire list of vertices.
void Xml911Recorder::compileHistories(AllVertices &vertices)
{
}

/// Writes simulation results to an output destination.
///
/// @param  vertices the Vertex list to search from.
void Xml911Recorder::saveSimData(const AllVertices &vertices)
{
   auto &conns = Simulator::getInstance().getModel().getConnections();
   Connections911 &conns911 = dynamic_cast<Connections911 &>(conns);
   All911Vertices &all911Vertices = dynamic_cast<All911Vertices &>(
      Simulator::getInstance().getModel().getLayout().getVertices());

   // create Vertex Types matrix
   VectorMatrix oldTypes(MATRIX_TYPE, MATRIX_INIT, 1, Simulator::getInstance().getTotalVertices(),
                         EXC);
   VectorMatrix vertexTypes(MATRIX_TYPE, MATRIX_INIT, 1,
                            Simulator::getInstance().getTotalVertices(), EXC);

   for (int i = 0; i < Simulator::getInstance().getTotalVertices(); i++) {
      vertexTypes[i] = Simulator::getInstance().getModel().getLayout().vertexTypeMap_[i];
      oldTypes[i] = conns911.oldTypeMap_[i];
   }

   // Write XML header information:
   resultOut_ << "<?xml version=\"1.0\" standalone=\"no\"?>\n"
              << "<!-- State output file for the 911 systems modeling-->\n";
   //stateOut << version; TODO: version
   auto &layout = Simulator::getInstance().getModel().getLayout();

   // Write the core state information:
   resultOut_ << "<SimState>\n";
   resultOut_ << "   " << layout.xloc_.toXML("xloc") << endl;
   resultOut_ << "   " << layout.yloc_.toXML("yloc") << endl;
   resultOut_ << "   " << oldTypes.toXML("vertexTypesPreEvent") << endl;
   resultOut_ << "   " << vertexTypes.toXML("vertexTypesPostEvent") << endl;

   // Write the number of trunks (lines) and servers
   resultOut_ << vectorToXML(all911Vertices.numTrunks_, "numTrunks") << endl;
   resultOut_ << vectorToXML(all911Vertices.numServers_, "numServers") << endl;

   // Write call information
   resultOut_ << vectorToXML(all911Vertices.droppedCalls_, "droppedCalls") << endl;
   resultOut_ << vectorToXML(all911Vertices.receivedCalls_, "receivedCalls") << endl;
   resultOut_ << vector2dToXML(all911Vertices.beginTimeHistory_, "BeginTimeHistory", "vertex")
              << endl;
   resultOut_ << vector2dToXML(all911Vertices.answerTimeHistory_, "AnswerTimeHistory", "vertex")
              << endl;
   resultOut_ << vector2dToXML(all911Vertices.endTimeHistory_, "EndTimeHistory", "vertex") << endl;
   resultOut_ << vector2dToXML(all911Vertices.wasAbandonedHistory_, "WasAbandonedHistory", "vertex")
              << endl;

   // Write system utilization and queue length histories
   resultOut_ << vector2dToXML(all911Vertices.queueLengthHistory_, "queueLengthHistory", "vertex")
              << endl;
   resultOut_ << vector2dToXML(all911Vertices.utilizationHistory_, "utilizationHistory", "vertex")
              << endl;

   // Print out deleted edges and vertices:
   resultOut_ << "   " << conns911.erasedVerticesToXML() << endl;
   resultOut_ << "   " << conns911.changedEdgesToXML(false) << endl;
   resultOut_ << "   " << conns911.changedEdgesToXML(true) << endl;

   // write time between growth cycles
   resultOut_
      << "   <Matrix name=\"Tsim\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">"
      << endl;
   resultOut_ << "   " << Simulator::getInstance().getEpochDuration() << endl;
   resultOut_ << "</Matrix>" << endl;

   // write simulation end time
   resultOut_
      << "   <Matrix name=\"simulationEndTime\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">"
      << endl;
   resultOut_ << "   " << g_simulationStep * Simulator::getInstance().getDeltaT() << endl;
   resultOut_ << "</Matrix>" << endl;
   resultOut_ << "</SimState>" << endl;
}

///  Prints out all parameters to logging file.
///  Registered to OperationManager as Operation::printParameters
void Xml911Recorder::printParameters()
{
}